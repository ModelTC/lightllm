import unittest
import torch
from lightllm.models.vit.triton_kernel.gelu_vit import gelu_fwd
from lightllm_kernel.ops import per_token_quant_bf16_fp8, gelu_per_token_quant_bf16_fp8
from test.utils import benchmark, error


def gelu_quant(x):
    y = gelu_fwd(x)
    return per_token_quant_bf16_fp8(y)


class TestGeluQuantBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.tokens = [13325]
        self.hiddenDims = [3200, 4800, 12800, 511, 1032, 1023, 1025]
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def test_accuracy(self):
        """Test the accuracy of gelu_per_token_quant"""
        for token in self.tokens:
            for hiddenDim in self.hiddenDims:
                with self.subTest(shape=[token, hiddenDim]):
                    input = torch.normal(
                        mean=0.0, std=10, size=[token, hiddenDim], device=self.device, dtype=self.dtype
                    )

                    y_real, scales_real = gelu_quant(input)
                    y_pred, scales_pred = gelu_per_token_quant_bf16_fp8(input)

                    self.assertTrue(
                        error(scales_real, scales_pred) < 0.01,
                        f"Accuracy test failed for size {token}, {hiddenDim}. "
                        f"scales_real={scales_real}, scales_pred={scales_pred}",
                    )
                    self.assertTrue(
                        error(y_real, y_pred) < 0.01,
                        f"Accuracy test failed for size {token}, {hiddenDim}." f"y_real={y_real}, y_pred={y_pred}",
                    )

    def test_performance(self):
        """Test the performance of gelu_per_token_quant using benchmark."""
        for token in self.tokens:
            for size in self.hiddenDims:
                with self.subTest(shape=[token, size]):
                    input = torch.rand(size=[token, size], device=self.device, dtype=self.dtype) - 0.5
                    shape = [[token, size]]
                    tflops = 0.0
                    benchmark(gelu_per_token_quant_bf16_fp8, shape, tflops, 100, input)
                    benchmark(gelu_quant, shape, tflops, 100, input)


if __name__ == "__main__":
    unittest.main()
