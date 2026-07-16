#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "flag_gems/backend_utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {
// utils::cdiv returns int; provide an int64_t variant for large N/D.
inline int64_t cdiv_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// C++ bypasses @libtuner/@triton.heuristics (the JIT wrapper is unwrapped to the
// raw kernel), so the tile config must be supplied here. Backend is selected by
// compile macro (CMakeLists.txt: -DFLAGGEMS_BACKEND=... -> FLAGGEMS_USE_*):
//   - Ascend (FLAGGEMS_USE_NPU): port of the heuristic formulas in
//     runtime/backend/_ascend/heuristics_config_utils.py (acnnl cdist/lp_norm_v2
//     informed: large contiguous MTE2 loads, UB budget BLOCK_M*BLOCK_D<=8192).
//   - NVIDIA / others: D-bucketed table (middle of the libtune bands in
//     runtime/backend/_nvidia/tune_configs.yaml).
#if defined(FLAGGEMS_USE_NPU)
constexpr int64_t kTileDCap = 8192;      // Ascend: cap BLOCK_D / split BLOCK_SIZE at 8192
constexpr int64_t kUBBudget = 8192;      // Ascend UB budget: BLOCK_M * BLOCK_D <= 8192 elems
#else
constexpr int64_t kTileDCap = 4096;      // H20: BLOCK_D / split BLOCK_SIZE cap
#endif

struct TileCfg {
  int64_t block_m;
  int64_t block_d;
  unsigned int num_warps;
};
TileCfg pick_tile(int64_t D) {
#if defined(FLAGGEMS_USE_NPU)
  // Mirrors _ascend pairwise_distance_heur_block_d / _block_m:
  //   BLOCK_D = min(next_pow2(D), 8192)
  //   BLOCK_M = max(1, min(32, 8192 // BLOCK_D))
  int64_t bd = std::min(utils::next_power_of_2(D), kTileDCap);
  int64_t bm = std::max(int64_t(1), std::min(int64_t(32), kUBBudget / bd));
  return {bm, bd, 8};
#else
  if (D <= 8)    return {16, utils::next_power_of_2(D), 4};   // block_d = 1/2/4/8
  if (D <= 64)   return {16, 64, 4};
  if (D <= 256)  return {8, 256, 4};
  if (D < 1024)  return {8, 512, 8};
  if (D < 4096)  return {4, 1024, 8};
  return {4, 4096, 8};
#endif
}
// Split-K BLOCK_SIZE cap. NOTE: both the NVIDIA and Ascend Python wrappers pin
// this to 4096 (Ascend PAIRWISE_SPLIT_MAX_BLOCK = 4096); the _ascend heuristic
// file's split_block (8192) is not what the wrapper actually uses. Keep 4096 to
// mirror runtime behavior. (kSplitBlockCap is intentionally decoupled from
// kTileDCap, which is the per-row BLOCK_D cap.)
constexpr int64_t kSplitBlockCap = 4096;
#if defined(FLAGGEMS_USE_NPU)
// Ascend wrapper splits only for very large D (single kernel is faster below
// 512K -- the 2nd launch + mid-buffer GM round-trip isn't worth it).
constexpr int64_t kAscendSplitDThreshold = 524288;
#endif

// Launch the two-stage split-K pair. HAS_P selects the general-p variant
// (which threads `p` between eps/MID_SIZE in stage-1 and out/MID_SIZE in stage-2).
template <bool HAS_P>
void launch_split_k(const std::string &src,
                    backend::RawStreamType raw,
                    const at::Tensor &a,
                    const at::Tensor &b,
                    const at::Tensor &out,
                    int64_t N,
                    int64_t D,
                    float eps,
                    float p,
                    int64_t MID_SIZE,
                    int64_t BLOCK_SIZE,
                    int64_t BLOCK_MID,
                    const std::string &k1_name,
                    const std::string &k2_name) {
  at::Tensor mid =
      at::empty({N, MID_SIZE}, at::TensorOptions().device(a.device()).dtype(at::kFloat));
  const unsigned int num_warps = 4;
  const unsigned int num_stages = 2;
  const TritonJITFunction &k1 = TritonJITFunction::get_instance(src, k1_name);
  const TritonJITFunction &k2 = TritonJITFunction::get_instance(src, k2_name);
  if constexpr (HAS_P) {
    // general_1: x1,x2,mid,D,eps,p,MID_SIZE,BLOCK_SIZE
    k1(raw,
       (unsigned int)N,
       (unsigned int)MID_SIZE,
       1u,
       num_warps,
       num_stages,
       a,
       b,
       mid,
       D,
       eps,
       p,
       MID_SIZE,
       BLOCK_SIZE);
    // general_2: mid,out,p,MID_SIZE,BLOCK_MID
    k2(raw, (unsigned int)N, 1u, 1u, num_warps, num_stages, mid, out, p, MID_SIZE, BLOCK_MID);
  } else {
    // p2/p1/p0/max/min _1: x1,x2,mid,D,eps,MID_SIZE,BLOCK_SIZE
    k1(raw,
       (unsigned int)N,
       (unsigned int)MID_SIZE,
       1u,
       num_warps,
       num_stages,
       a,
       b,
       mid,
       D,
       eps,
       MID_SIZE,
       BLOCK_SIZE);
    // _2: mid,out,MID_SIZE,BLOCK_MID
    k2(raw, (unsigned int)N, 1u, 1u, num_warps, num_stages, mid, out, MID_SIZE, BLOCK_MID);
  }
}
}  // namespace

// pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> ||x1 - x2 + eps||_p
// Drives the existing Triton kernels in ops/pairwise_distance.py; replicates the
// Python wrapper's dispatch (per-row vs split-K, p fast paths, D==0 guard).
at::Tensor pairwise_distance(const at::Tensor &x1,
                             const at::Tensor &x2,
                             double p /* = 2.0 */,
                             double eps /* = 1e-6 */,
                             bool keepdim /* = false */) {
  TORCH_CHECK(x1.dim() >= 1 && x2.dim() >= 1, "pairwise_distance: input must have >= 1 dim");

  std::vector<at::Tensor> bc = at::broadcast_tensors({x1, x2});
  at::Tensor a = bc[0].contiguous();
  at::Tensor b = bc[1].contiguous();

  int64_t D = a.size(-1);
  const float p_f = static_cast<float>(p);
  const float eps_f = static_cast<float>(eps);

  // D == 0: torch returns 0 for finite p; inf/-inf have no identity over an empty
  // reduction and torch raises. Also avoids the split-K plumbing dividing by 0.
  if (D == 0) {
    TORCH_CHECK(std::isinf(p) == false,
                "pairwise_distance cannot compute the inf/-inf norm on an empty "
                "reduction dimension (no identity element)");
    std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
    out_shape.pop_back();
    at::Tensor out = at::zeros(out_shape, a.options());
    return keepdim ? out.unsqueeze(-1) : out;
  }

  int64_t N = a.numel() / D;
  std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
  out_shape.pop_back();
  at::Tensor out = at::empty(out_shape, a.options());
  if (keepdim) out = out.unsqueeze(-1);

  // split-K plumbing. BLOCK_SIZE cap is 4096 on both backends (kSplitBlockCap).
  // The split-K TRIGGER differs by backend (mirrors the Python wrappers):
  //   - Ascend (_ascend/ops/pairwise_distance.py): split only when D is very
  //     large (D >= 524288); the single kernel wins below that.
  //   - NVIDIA (ops/pairwise_distance.py): split when few rows (N <= 128) and
  //     D is large enough to split (MID_SIZE >= 2).
  int64_t BLOCK_SIZE = std::min(utils::next_power_of_2(D), kSplitBlockCap);
  int64_t MID_SIZE = cdiv_i64(D, BLOCK_SIZE);
  int64_t BLOCK_MID = utils::next_power_of_2(MID_SIZE);
#if defined(FLAGGEMS_USE_NPU)
  bool use_split_k = (D >= kAscendSplitDThreshold);
#else
  bool use_split_k = (N <= 128) && (MID_SIZE >= 2);
#endif

  const std::string src =
      (utils::get_flag_gems_src_path() / "ops" / "pairwise_distance.py").string();

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  auto launch_per_row = [&](const std::string &kname, bool has_p) {
    TileCfg t = pick_tile(D);
    const TritonJITFunction &f = TritonJITFunction::get_instance(src, kname);
    unsigned int grid_x = static_cast<unsigned int>(cdiv_i64(N, t.block_m));
    if (has_p) {
      // general: x1,x2,out,N,D,eps,p,BLOCK_M,BLOCK_D
      f(raw_stream, grid_x, 1u, 1u, t.num_warps, 2u, a, b, out, N, D, eps_f, p_f, t.block_m, t.block_d);
    } else {
      // p2/p1/p0/max/min: x1,x2,out,N,D,eps,BLOCK_M,BLOCK_D
      f(raw_stream, grid_x, 1u, 1u, t.num_warps, 2u, a, b, out, N, D, eps_f, t.block_m, t.block_d);
    }
  };

  if (p == 2.0) {
    if (!use_split_k) {
      launch_per_row("pairwise_distance_p2_kernel", false);
    } else {
      launch_split_k<false>(src, raw_stream, a, b, out, N, D, eps_f, p_f, MID_SIZE, BLOCK_SIZE,
                            BLOCK_MID, "pairwise_distance_p2_kernel_1", "pairwise_distance_p2_kernel_2");
    }
  } else if (p == 1.0) {
    if (!use_split_k) {
      launch_per_row("pairwise_distance_p1_kernel", false);
    } else {
      launch_split_k<false>(src, raw_stream, a, b, out, N, D, eps_f, p_f, MID_SIZE, BLOCK_SIZE,
                            BLOCK_MID, "pairwise_distance_p1_kernel_1", "pairwise_distance_p1_kernel_2");
    }
  } else if (p == 0.0) {
    if (!use_split_k) {
      launch_per_row("pairwise_distance_p0_kernel", false);
    } else {
      launch_split_k<false>(src, raw_stream, a, b, out, N, D, eps_f, p_f, MID_SIZE, BLOCK_SIZE,
                            BLOCK_MID, "pairwise_distance_p0_kernel_1", "pairwise_distance_p0_kernel_2");
    }
  } else if (std::isinf(p) && p > 0) {
    if (!use_split_k) {
      launch_per_row("pairwise_distance_max_kernel", false);
    } else {
      launch_split_k<false>(src, raw_stream, a, b, out, N, D, eps_f, p_f, MID_SIZE, BLOCK_SIZE,
                            BLOCK_MID, "pairwise_distance_max_kernel_1", "pairwise_distance_max_kernel_2");
    }
  } else if (std::isinf(p) && p < 0) {
    if (!use_split_k) {
      launch_per_row("pairwise_distance_min_kernel", false);
    } else {
      launch_split_k<false>(src, raw_stream, a, b, out, N, D, eps_f, p_f, MID_SIZE, BLOCK_SIZE,
                            BLOCK_MID, "pairwise_distance_min_kernel_1", "pairwise_distance_min_kernel_2");
    }
  } else {
    if (!use_split_k) {
      launch_per_row("pairwise_distance_general_kernel", true);
    } else {
      launch_split_k<true>(src, raw_stream, a, b, out, N, D, eps_f, p_f, MID_SIZE, BLOCK_SIZE,
                           BLOCK_MID, "pairwise_distance_general_kernel_1", "pairwise_distance_general_kernel_2");
    }
  }

  return out;
}

}  // namespace flag_gems
