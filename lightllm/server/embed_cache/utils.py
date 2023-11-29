import torch
from io import BytesIO


def tensor2bytes(t):
    t = t.cpu()
    buf = BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    return buf.read()


def bytes2tensor(b):
    return torch.load(BytesIO(b))
