import asyncio
import rpyc
import time
import threading
from typing import Union
from rpyc.utils.classic import obtain
from .metrics import Monitor
from prometheus_client import generate_latest
import multiprocessing.shared_memory as shm


class MetricServer(rpyc.Service):
    def __init__(self, args) -> None:
        super().__init__()
        self.monitor = Monitor(args)
        self.interval = args.push_interval
        data_size = len(generate_latest(self.monitor.registry))
        self.shared_memory = shm.SharedMemory(name="latest_metrics", create=True, size=data_size)

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        self.shared_memory.close()
        self.shared_memory.unlink()
        pass

    def exposed_counter_inc(self, name: str, label: str = None) -> None:
        return self.monitor.counter_inc(name, label)

    def exposed_histogram_observe(self, name: str, value: float, label: str = None) -> None:
        return self.monitor.histogram_observe(name, value, label)

    def exposed_gauge_set(self, name: str, value: float) -> None:
        return self.monitor.gauge_set(name, value)

    def exposed_generate_latest(self) -> None:
        data = generate_latest(self.monitor.registry)
        data_len = len(data)
        if data_len > self.shared_memory.size:
            # 如果不够大，需要重新初始化
            self.shared_memory.close()
            self.shared_memory.unlink()
            self.shared_memory = shm.SharedMemory(name="latest_metrics", create=True, size=data_len)
        self.shared_memory.buf[:data_len] = data
        return

    def push_metrics(self):
        while True:
            self.monitor.push_metrices()
            print("Metrics pushed to Pushgateway")
            time.sleep(self.interval)


def start_metric_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    service = MetricServer(args)
    if args.metric_gateway is not None:
        push_thread = threading.Thread(target=service.push_metrics)
        push_thread.start()  # 启动推送任务线程

    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port)
    pipe_writer.send("init ok")
    t.start()
