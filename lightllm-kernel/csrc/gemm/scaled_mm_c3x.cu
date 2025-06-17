#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12000

  #include "scaled_mm_c3x_sm90_fp8_dispatch.cuh"
  #include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace lightllm {
namespace ops {

using namespace lightllm;
/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm90a (Hopper) or later.
*/

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm90_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  
}

void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias,
                            c10::optional<torch::Tensor> const& ls) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias && ls) {
    TORCH_CHECK(bias->dtype() == c.dtype(),
                "currently bias dtype must match output dtype ", c.dtype());
    TORCH_CHECK(ls->dtype() == c.dtype(),
                "currently ls dtype must match output dtype ", c.dtype());
    return cutlass_scaled_mm_sm90_epilogue<c3x::ScaledEpilogueBiasLs>(
        c, a, b, a_scales, b_scales, *bias, *ls);
  } else if (bias) {
    TORCH_CHECK(bias->dtype() == c.dtype(),
                "currently bias dtype must match output dtype ", c.dtype());
    return cutlass_scaled_mm_sm90_epilogue<c3x::ScaledEpilogueBias>(
        c, a, b, a_scales, b_scales, *bias);
  } else if (ls) {
    TORCH_CHECK(ls->dtype() == c.dtype(),
                "currently ls dtype must match output dtype ", c.dtype());
    return cutlass_scaled_mm_sm90_epilogue<c3x::ScaledEpilogueLs>(
        c, a, b, a_scales, b_scales, *ls);
  } else {
    return cutlass_scaled_mm_sm90_epilogue<c3x::ScaledEpilogue>(
        c, a, b, a_scales, b_scales);
  }
}

} // namespace ops
} // namespace lightllm

#endif
