import os
import time
import threading
import psutil
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def is_process_active(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def check_parent_alive():
    parent_pid = os.getppid()
    while True:
        if not is_process_active(parent_pid):
            logger.warning("parent is dead, kill self")
            os._exit(-1)

        time.sleep(10)
    return


def start_parent_check_thread():
    thread = threading.Thread(target=check_parent_alive)
    thread.start()
