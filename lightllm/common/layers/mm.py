import abc
import torch
from lightllm.common.layers.quantization import get_quantization_method
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


class MMBase(abc.ABC):
    def __init__(self, mode):
        super().__init__()
        self.quantize_method = get_quantization_method(mode)()

    def preprocess_weight(self, weight, func=None):
        """
        Preprocess the weight tensor.
        Args:.

        Returns:
            torch.Tensor: The preprocessed weight tensor.
        """
        if self.quantize_method is not None:
            return self.quantize_method.quantize(weight)
        if func is not None:
            return func(weight.transpose(0, 1))
        return weight

    @abc.abstractmethod
    def apply(self, input_tensor, weight, bias=None, beta=1, alpha=1, out=None):
        pass


class MM(MMBase):
    def apply(self, input_tensor, weight, bias=None, beta=1, alpha=1, out=None):
        """
        Applies a matrix multiplication or an addmm operation.
        If bias is provided, performs the operation: result = beta * bias + alpha * (input_tensor @ weight).
        Otherwise, performs a standard matrix multiplication.

        Args:
            input_tensor (torch.Tensor): Input tensor for matrix multiplication.
            weight (torch.Tensor): Weight tensor for multiplication.
            bias (torch.Tensor, optional): Bias tensor for addition.
            beta (float, optional): Scaling factor for bias.
            alpha (float, optional): Scaling factor for matrix multiplication result.
            out (torch.Tensor, optional): Output tensor to store the result.

        Returns:
            torch.Tensor: The result of the addmm or mm operation.
        """
        if self.quantize_method is not None:
            return self.quantize_method.apply(input_tensor, weight, bias, out)

        if out is None:
            shape = (input_tensor.shape[0], weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
        if bias is None:
            return torch.mm(input_tensor, weight, out=out)
        # Perform matrix multiplication and addition using torch.addmm.
        return torch.addmm(bias, input_tensor, weight, beta=beta, alpha=alpha, out=out)


# Example usage
if __name__ == "__main__":
    bias = torch.randn(3)
    input_tensor = torch.randn(3, 2)
    weight = torch.randn(2, 3)

    mm = MM()
    result_addmm = mm.apply(input_tensor, weight, bias)
    print("Result of addmm:")
    print(result_addmm)

    input_tensor = torch.randn(3, 3)
    weight = torch.randn(3, 3)
    result_mm = mm.apply(input_tensor, weight)
    print("Result of mm:")
    print(result_mm)
