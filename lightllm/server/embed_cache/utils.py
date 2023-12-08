import torch
import numpy as np
from io import BytesIO


def tensor2bytes(t):
    t = t.cpu().numpy().tobytes()
    return t
    # buf = BytesIO()
    # torch.save(t, buf)
    # buf.seek(0)
    # return buf.read()


def bytes2tensor(b):
    return torch.from_numpy(np.frombuffer(b, dtype=np.float16)).cuda()
    # return torch.load(BytesIO(b))
