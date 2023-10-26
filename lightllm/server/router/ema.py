class EmaStatistical:

    def __init__(self, init_len, token_ratio, decay=0.9, moving_max_new_tokens=False, **kwargs) -> None:
        self.global_step = 0
        self.decay = decay
        self.scale = 1.
        self.moving_len = init_len
        self.token_ratio = token_ratio
        self.moving_max_new_tokens = moving_max_new_tokens
        self.history = []

    def udpate(self, update_len):
        self.moving_len = int((1.0 - self.decay) * update_len + self.decay * self.moving_len)
        self.history.append(self.moving_len)
        self.global_step += 1
    
    def get_moving_len(self):
        # return self.moving_len * self.scale
        moving_len = sum(self.history) / len(self.history) if len(self.history) > 0 else self.moving_len
        return moving_len * self.scale
    
    def get_max_output_len(self, req, token_traio=0.):
        if not self.moving_max_new_tokens:
            return req.max_output_len
        if token_traio > self.token_ratio:
            return req.max_output_len
        output_len = self.get_moving_len()
        return output_len if output_len <= req.max_output_len else req.max_output_len