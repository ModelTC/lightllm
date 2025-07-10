import unittest
import torch
from lightllm_kernel.ops import rmsnorm_bf16
from test.utils import benchmark, error


class TestRmsNormBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.batchs = [1024, 13325]
        self.sizes = [1024, 1025, 1032, 3200, 3201, 3208, 12800]
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def test_accuracy(self):
        """Test the accuracy of rmsnorm against torch.rmsnorm."""
        for batch in self.batchs:
            for size in self.sizes:
                with self.subTest(shape=[batch, size]):
                    X = torch.rand(size=[batch, size], device=self.device, dtype=self.dtype) - 0.5
                    W = torch.rand(size=[size], device=self.device, dtype=self.dtype) - 0.5

                    y_real = torch.nn.functional.rms_norm(X, (size,), W)
                    y_pred = rmsnorm_bf16(X, W)
                    self.assertTrue(
                        error(y_pred, y_real) < 0.01,
                        f"Accuracy test failed for size {batch}, {size}. y_real={y_real}, y_pred={y_pred}",
                    )
                    print(f"{error(y_pred, y_real) = }")

    def test_performance(self):
        """Test the performance of rmsnorm using benchmark."""
        for batch in self.batchs:
            for size in self.sizes:
                with self.subTest(shape=[batch, size]):
                    X = torch.rand(size=[batch, size], device=self.device, dtype=self.dtype) - 0.5
                    W = torch.rand(size=[size], device=self.device, dtype=self.dtype) - 0.5

                    shape = [[batch, size], [size], [batch, size]]
                    tflops = 0.0
                    benchmark(rmsnorm_bf16, shape, tflops, 100, X, W)
                    benchmark(torch.nn.functional.rms_norm, shape, tflops, 100, X, (size,), W)


if __name__ == "__main__":
    unittest.main()
