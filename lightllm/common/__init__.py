import torch
try:
    from numpy import from_dlpack as _np_from_dlpack
    np_from_tensor = _np_from_dlpack
except:
    def np_from_tensor(tensor: torch.Tensor):
        return tensor.cpu().numpy()
