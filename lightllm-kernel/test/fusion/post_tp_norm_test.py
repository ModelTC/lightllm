import unittest
import torch
from lightllm_kernel.ops import post_tp_norm_bf16
from test.utils import benchmark, error


def post_tp_norm(input, weight, tp_variance, embed_dim, eps):
    input = input.to(torch.float32)
    variance = tp_variance / embed_dim
    variance = variance.unsqueeze(-1)
    input = input * torch.rsqrt(variance + eps)
    out = weight * input.to(torch.bfloat16)
    return out


class TestPostTpNormBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.batchs = [1024, 13325]
        self.sizes = [1024, 1025, 1032, 3200, 3201, 3208, 12800]
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.embed_dim = 3200
        self.eps = 1e-6

    def test_accuracy(self):
        for batch in self.batchs:
            for size in self.sizes:
                with self.subTest(shape=[batch, size]):
                    X = torch.rand(size=[batch, size], device=self.device, dtype=self.dtype) - 0.5
                    W = torch.rand(size=[size], device=self.device, dtype=self.dtype) - 0.5
                    V = torch.rand(size=[batch], device=self.device, dtype=torch.float32)

                    y_real = post_tp_norm(X, W, V, self.embed_dim, self.eps)
                    y_pred = post_tp_norm_bf16(X, W, V, self.embed_dim, self.eps)
                    self.assertTrue(
                        error(y_pred, y_real) < 0.01,
                        f"Accuracy test failed for size {batch}, {size}. y_real={y_real}, y_pred={y_pred}",
                    )

    def test_performance(self):
        for batch in self.batchs:
            for size in self.sizes:
                with self.subTest(shape=[batch, size]):
                    X = torch.rand(size=[batch, size], device=self.device, dtype=self.dtype) - 0.5
                    W = torch.rand(size=[size], device=self.device, dtype=self.dtype) - 0.5
                    V = torch.rand(size=[batch], device=self.device, dtype=torch.float32)

                    shape = [[batch, size], [size], [batch, size]]
                    tflops = 0.0
                    benchmark(post_tp_norm_bf16, shape, tflops, 100, X, W, V, self.embed_dim, self.eps)
                    benchmark(post_tp_norm, shape, tflops, 100, X, W, V, self.embed_dim, self.eps)


if __name__ == "__main__":
    unittest.main()
