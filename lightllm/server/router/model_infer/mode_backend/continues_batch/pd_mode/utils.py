import threading


def join_if_alive(thread: threading.Thread):
    if thread is not None and thread.is_alive():
        try:
            thread.join()
        except Exception:
            pass
    return
