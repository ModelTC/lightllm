import asyncio
import threading
import queue
import websockets
from dataclasses import asdict
from lightllm.server.io_struct import UpKVStatus
import json
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class UpStatusManager:
    def __init__(self, args):
        self.args = args
        self.task_queue = queue.Queue(maxsize=1000)
        self.daemon_thread = threading.Thread(target=self.thread_loop, daemon=True)
        self.daemon_thread.start()

    def put_status_task(self, upkv_status: UpKVStatus):
        self.task_queue.put(upkv_status)
        return

    def thread_loop(self):
        asyncio.run(self.loop())

    async def loop(self):
        while True:
            try:
                uri = f"ws://{self.args.pd_master_ip}:{self.args.pd_master_port}/kv_move_status"
                async with websockets.connect(uri) as websocket:
                    while True:
                        loop = asyncio.get_event_loop()
                        upkv_status = await loop.run_in_executor(None, self.task_queue.get)
                        self.task_queue.task_done()
                        await websocket.send(json.dumps(asdict(upkv_status)))
                        logger.info(f"up status: {upkv_status}")
            except Exception as e:
                logger.error(f"connetion to pd_master has error: {str(e)}")
                logger.exception(str(e))
                await asyncio.sleep(10)
                logger.info("reconnection to pd_master")
