import unittest
import torch
from lightllm_kernel.ops import pre_tp_norm_bf16
from test.utils import benchmark, error


def pre_tp_norm(input):
    input = input.to(torch.float32)
    tp_variance = input.pow(2).sum(-1, keepdim=False)
    return tp_variance


class TestPreTpNormBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.batchs = [1024, 13325]
        self.sizes = [1024, 1025, 1032, 3200, 3201, 3208, 12800]
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def test_accuracy(self):
        for batch in self.batchs:
            for size in self.sizes:
                with self.subTest(shape=[batch, size]):
                    X = torch.rand(size=[batch, size], device=self.device, dtype=self.dtype) - 0.5

                    y_real = pre_tp_norm(X)
                    y_pred = pre_tp_norm_bf16(X)
                    self.assertTrue(
                        error(y_pred, y_real) < 0.01,
                        f"Accuracy test failed for size {batch}, {size}. y_real={y_real}, y_pred={y_pred}",
                    )

    def test_performance(self):
        for batch in self.batchs:
            for size in self.sizes:
                with self.subTest(shape=[batch, size]):
                    X = torch.rand(size=[batch, size], device=self.device, dtype=self.dtype) - 0.5
                    # W = torch.rand(size=[size], device=self.device, dtype=self.dtype) - 0.5

                    shape = [[batch, size], [size], [batch, size]]
                    tflops = 0.0
                    benchmark(pre_tp_norm_bf16, shape, tflops, 100, X)
                    benchmark(pre_tp_norm, shape, tflops, 100, X)


if __name__ == "__main__":
    unittest.main()
