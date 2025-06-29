import unittest
import torch
from lightllm_kernel.ops import cutlass_scaled_mm_bias_ls
from lightllm.common.vllm_kernel import _custom_ops as ops
from test.utils import benchmark, error


def torch_cutlass_scale_gemm_with_ls(x_q, w_q_t, x_scale, w_scale, out_dtype=torch.bfloat16, bias=None, ls=None):
    y_pred_tmp = ops.cutlass_scaled_mm(x_q, w_q_t, x_scale, w_scale, out_dtype=out_dtype, bias=bias)
    y_pred = y_pred_tmp * ls
    return y_pred


class TestQuantBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.tokens = [128, 1024, 13325]
        self.hiddenDims = [256, 512, 1024, 3200]
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def test_accuracy(self):
        """Test the accuracy of cutlass_scaled_mm_bias_ls"""
        for token in self.tokens:
            for hiddenDim in self.hiddenDims:
                with self.subTest(shape=[token, hiddenDim]):
                    M, N, K = token, 3 * hiddenDim, hiddenDim

                    input = torch.randn(size=[M, K], device=self.device, dtype=self.dtype)
                    x_q, x_scale = ops.scaled_fp8_quant(input, scale=None, scale_ub=None, use_per_token_if_dynamic=True)

                    # 生成权重张量w_q（N×K），转置后为K×N（列优先）
                    weight = torch.randn(size=[N, K], device=self.device, dtype=self.dtype)
                    w_q, w_scale = ops.scaled_fp8_quant(
                        weight, scale=None, scale_ub=None, use_per_token_if_dynamic=True
                    )

                    # 转置，w_q_t为列优先
                    w_q_t = w_q.t()
                    assert w_q_t.stride(0) == 1, "权重转置后步幅需列优先"

                    y_pred = torch.empty((M, N), dtype=input.dtype, device=input.device)
                    bias = torch.randn(size=[N], device=self.device, dtype=torch.bfloat16)
                    ls = torch.randn(size=[N], device=self.device, dtype=torch.bfloat16)

                    cutlass_scaled_mm_bias_ls(y_pred, x_q, w_q_t, x_scale, w_scale, bias=bias, ls=ls)
                    y_real = torch_cutlass_scale_gemm_with_ls(
                        x_q, w_q_t, x_scale, w_scale, out_dtype=torch.bfloat16, bias=bias, ls=ls
                    )

                    self.assertTrue(
                        error(y_pred, y_real) < 0.01,
                        f"Accuracy test failed for size {token}, {hiddenDim}. y_pred={y_pred}, y_real={y_real}",
                    )

    def test_performance(self):
        """Test the performance of cutlass_scaled_mm_bias_ls"""
        for token in self.tokens:
            for hiddenDim in self.hiddenDims:
                with self.subTest(shape=[token, hiddenDim]):
                    M, N, K = token, 3 * hiddenDim, hiddenDim

                    input = torch.randn(size=[M, K], device=self.device, dtype=self.dtype) - 0.5
                    x_q, x_scale = ops.scaled_fp8_quant(input, scale=None, scale_ub=None, use_per_token_if_dynamic=True)

                    # 生成权重张量w_q（N×K），转置后为K×N（列优先）
                    weight = torch.randn(size=[N, K], device=self.device, dtype=self.dtype) - 0.5
                    w_q, w_scale = ops.scaled_fp8_quant(
                        weight, scale=None, scale_ub=None, use_per_token_if_dynamic=True
                    )

                    bias = torch.randn(size=[N], device=self.device, dtype=torch.bfloat16)
                    ls = torch.randn(size=[N], device=self.device, dtype=torch.bfloat16)
                    # 转置，w_q_t为列优先
                    w_q_t = w_q.t()
                    assert w_q_t.stride(0) == 1, "权重转置后步幅需列优先"

                    y_pred = torch.empty((M, N), dtype=input.dtype, device=input.device)
                    shape = [[token, hiddenDim]]
                    tflops = 2 * token * (3 * hiddenDim) * hiddenDim / 1024 ** 4
                    benchmark(
                        cutlass_scaled_mm_bias_ls,
                        shape,
                        tflops,
                        100,
                        y_pred,
                        x_q,
                        w_q_t,
                        x_scale,
                        w_scale,
                        bias=bias,
                        ls=ls,
                    )
                    benchmark(
                        torch_cutlass_scale_gemm_with_ls,
                        shape,
                        tflops,
                        100,
                        x_q,
                        w_q_t,
                        x_scale,
                        w_scale,
                        out_dtype=torch.bfloat16,
                        bias=bias,
                        ls=ls,
                    )  # 无bias 495GB/s, 有bias 482GB/s


if __name__ == "__main__":
    unittest.main()
