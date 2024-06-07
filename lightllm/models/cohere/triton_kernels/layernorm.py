import torch

import triton
import triton.language as tl

@torch.no_grad()
def layernorm_forward(x, weight, eps):
    return torch.layer_norm(x, (x.shape[-1],), weight, bias=None, eps=eps)

def mh_layernorm_forward(x, weight, eps):
    # x shape : (bs, seqlen, head, head_dim)
    inp_dtype = x.dtype
    x = x.to(torch.float32)
    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + eps)
    x = weight.to(torch.float32) * x
    return x.to(inp_dtype)


class CohereLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5, bias=False):
        """The hidden size can be a tuple or an int. The tuple is used for QKNorm to normalize across head_dim"""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)


def test():
    hidden_size = 768
    bs = 10
    seqlen = 128
    m = CohereLayerNorm(hidden_size).to(torch.float32)
    for i in range(10):
        x = torch.randn(bs, seqlen, hidden_size, dtype=torch.float32)
        output_1 = m(x)
        output_2 = layernorm_forward(x, m.weight, m.variance_epsilon)
        print(torch.allclose(output_1, output_2, atol=1e-4))
        max_err = torch.max(torch.abs(output_1 - output_2))
        print("max error:", max_err)

    head = 8
    head_dim = 64
    m = CohereLayerNorm((head, head_dim)).to(torch.float32)
    print(m.weight.shape)
    for i in range(10):
        x = torch.randn(bs * seqlen, head, head_dim, dtype=torch.float32)
        output_1 = m(x)
        output_2 = mh_layernorm_forward(
            x.view(bs * seqlen, head, head_dim), m.weight, m.variance_epsilon
                )
        print(torch.allclose(output_1, output_2, atol=1e-4))
        max_err = torch.max(torch.abs(output_1 - output_2))
        print("max error:", max_err)

if __name__ == "__main__": 
    test()
