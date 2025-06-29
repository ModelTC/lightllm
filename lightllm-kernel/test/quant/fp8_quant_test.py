import unittest
import torch
from lightllm.common.vllm_kernel import _custom_ops as ops
from lightllm_kernel.ops import per_token_quant_bf16_fp8
from test.utils import benchmark, error


class TestQuantBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.tokens = [1024, 13325]
        self.hiddenDims = [3, 256, 511, 1023, 1024, 1025, 1032, 3200, 3201, 3208, 12800]
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def test_accuracy(self):
        """Test the accuracy of per_token_quant"""
        for token in self.tokens:
            for hiddenDim in self.hiddenDims:
                with self.subTest(shape=[token, hiddenDim]):
                    input = torch.rand(size=[token, hiddenDim], device=self.device, dtype=self.dtype) - 0.5
                    y_real, scales_real = ops.scaled_fp8_quant(
                        input.contiguous(), scale=None, use_per_token_if_dynamic=True
                    )
                    y_pred, scales_pred = per_token_quant_bf16_fp8(input)
                    self.assertTrue(
                        error(scales_real, scales_pred) < 0.01,
                        f"Accuracy test failed for size {token}, {hiddenDim}."
                        f"scales_real={scales_real}, scales_pred={scales_pred}",
                    )
                    self.assertTrue(
                        error(y_real, y_pred) < 0.01,
                        f"Accuracy test failed for size {token}, {hiddenDim}. y_real={y_real}, y_pred={y_pred}",
                    )

    def test_performance(self):
        """Test the performance of per_token_quant"""
        for token in self.tokens:
            for size in self.hiddenDims:
                with self.subTest(shape=[token, size]):
                    input = torch.rand(size=[token, size], device=self.device, dtype=self.dtype) - 0.5
                    shape = [[token, size]]
                    tflops = token * size / 1024 ** 4
                    benchmark(per_token_quant_bf16_fp8, shape, tflops, 100, input)
                    benchmark(ops.scaled_fp8_quant, shape, tflops, 100, input, None, True)


if __name__ == "__main__":
    unittest.main()
