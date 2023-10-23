import threading

class ReqIDGenerator:
    def __init__(self):
        self.current = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            self.current += 1
            return self.current