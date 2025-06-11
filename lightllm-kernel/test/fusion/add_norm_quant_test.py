import unittest
import torch
from lightllm_kernel.ops import add_norm_quant_bf16_fp8
from lightllm.common.vllm_kernel import _custom_ops as ops
from test.utils import benchmark, error


def torch_add_norm_quant_bf16_fp8(X, R, W, eps=1e-6):
    N = X.size(1)
    # 1. Add residual
    X = X.add_(R)
    # 2. rmsnorm
    normalized = torch.nn.functional.rms_norm(X, (N,), W, eps=eps)
    # 3. per token quant
    quantized, scales = ops.scaled_fp8_quant(normalized, scale=None, use_per_token_if_dynamic=True)

    return quantized, scales


class TestFusedAddNormQuantBF16(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.batchs = [13]
        self.seqLens = [1025]
        self.embed_dims = [16, 32, 64, 512, 1024, 3200, 4096, 12800, 24, 511, 513, 1023, 1025, 1032, 4097]
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.eps = 1e-6

    def test_accuracy(self):
        """Test the accuracy of FusedAddNormQuant against torch."""
        for batch in self.batchs:
            for seqLen in self.seqLens:
                for embed_dim in self.embed_dims:
                    with self.subTest(shape=[batch, seqLen, embed_dim]):
                        X1 = torch.rand(size=[batch, seqLen, embed_dim], device=self.device, dtype=self.dtype) - 0.5
                        X2 = X1.clone()
                        R1 = torch.rand(size=[batch, seqLen, embed_dim], device=self.device, dtype=self.dtype) - 0.5
                        R2 = R1.clone()
                        W = torch.rand(size=[embed_dim], device=self.device, dtype=self.dtype) - 0.5
                        output_real, scales_real = torch_add_norm_quant_bf16_fp8(
                            X1.reshape(-1, X1.shape[2]), R1.reshape(-1, R1.shape[2]), W, self.eps
                        )
                        output_pred, scales_pred = add_norm_quant_bf16_fp8(
                            X2.reshape(-1, X1.shape[2]), R2.reshape(-1, R2.shape[2]), W, self.eps
                        )

                        self.assertTrue(
                            error(output_real, output_pred) < 0.01,
                            f"Accuracy test failed for size {batch}, {seqLen}, {embed_dim}. "
                            f"output_real={output_real}, output_pred={output_pred}",
                        )
                        self.assertTrue(
                            error(scales_real, scales_pred) < 0.01,
                            f"Accuracy test failed for size {batch}, {seqLen}, {embed_dim}. "
                            f"scales_real={scales_real}, scales_pred={scales_pred}",
                        )

    def test_performance(self):
        """Test the performance of FusedAddNormQuant using benchmark."""
        for batch in self.batchs:
            for seqLen in self.seqLens:
                for embed_dim in self.embed_dims:
                    with self.subTest(shape=[batch, seqLen, embed_dim]):
                        X1 = torch.rand(size=[batch, seqLen, embed_dim], device=self.device, dtype=self.dtype) - 0.5
                        X2 = torch.rand(size=[batch, seqLen, embed_dim], device=self.device, dtype=self.dtype) - 0.5
                        R1 = torch.rand(size=[batch, seqLen, embed_dim], device=self.device, dtype=self.dtype) - 0.5
                        R2 = R1.clone()
                        W = torch.rand(size=[embed_dim], device=self.device, dtype=self.dtype) - 0.5

                        shape = [[batch, seqLen, embed_dim]]
                        tflops = 0.0
                        benchmark(
                            torch_add_norm_quant_bf16_fp8,
                            shape,
                            tflops,
                            100,
                            X1.reshape(-1, X1.shape[2]),
                            R1.reshape(-1, R1.shape[2]),
                            W,
                            self.eps,
                        )
                        benchmark(
                            add_norm_quant_bf16_fp8,
                            shape,
                            tflops,
                            100,
                            X2.reshape(-1, X1.shape[2]),
                            R2.reshape(-1, R2.shape[2]),
                            W,
                            self.eps,
                        )


if __name__ == "__main__":
    unittest.main()
