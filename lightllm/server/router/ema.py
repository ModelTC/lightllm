
import numpy as np

class Ema:

    def __init__(self, init_len, decay=0.9, moving_max_new_tokens=False) -> None:
        self.global_step = 0
        self.decay = decay
        self.moving_len = init_len
        self.moving_max_new_tokens = moving_max_new_tokens

    def udpate(self, update_len):
        self.moving_len = int((1.0 - self.decay) * update_len + self.decay * self.moving_len)
        self.global_step += 1

    def get_moving_len(self):
        return self.moving_len
    
    def get_max_output_len(self, req):
        if self.moving_max_new_tokens:
            return self.get_moving_len()
        else:
            return req.max_output_len

class Clr(Ema):

    def __init__(self, init_len, decay=0.9, moving_max_new_tokens=False) -> None:
        super().__init__(init_len, decay, moving_max_new_tokens)
        self.max_len = 1024
        self.min_len = 512
        self.step_size = 10

    def clr(self, step):
        cycle = np.floor(1+step/(2*self.step_size))
        x = np.abs(step/self.step_size - 2*cycle + 1)
        y = self.min_len + (self.max_len-self.min_len)*np.maximum(0, (1-x))
        return y

    def udpate(self, update_len):
        self.moving_len = self.clr(self.global_step)
        self.global_step += 1