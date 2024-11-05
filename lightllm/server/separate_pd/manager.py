import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import fastapi
import uvicorn
import zmq
import zmq.asyncio

from lightllm.server.io_struct import IdleReq, RouterLoadOut, SPDAssignReq, SPDCommitReq, SPDPreCommitReq
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class SPDScheduler:
    def __init__(
        self,
        socket_urls,
        instance_ids,
        detokenization_socket,
        load_sockets,
        feedback_sockets,
        max_tokens,
    ):
        self.args = (
            socket_urls,
            instance_ids,
            detokenization_socket,
            load_sockets,
            feedback_sockets,
            max_tokens,
        )

    def init_async_resources(self):
        socket_urls, instance_ids, detokenization_socket, load_sockets, feedback_sockets, max_tokens = self.args

        context = zmq.asyncio.Context(len(socket_urls) + len(load_sockets) + 1)
        assert (
            len(socket_urls) == len(instance_ids) == len(load_sockets)
        ), f"socket urls {socket_urls} instance ids {instance_ids} load sockets {load_sockets} not match"
        self.instance_id_list = [int(instance_id) for instance_id in instance_ids]
        self.max_tokens_per_instance = max_tokens
        self.socket_idx = -1
        self.socket_dict = {}
        for mid, socket_url in zip(instance_ids, socket_urls):
            socket = context.socket(zmq.PUSH)
            socket.connect(f"tcp://{socket_url}")
            self.socket_dict[mid] = socket
        self.feedback_sockets = {}
        for mid, feedback_socket in zip(instance_ids, feedback_sockets):
            socket = context.socket(zmq.PULL)
            socket.bind(f"tcp://{feedback_socket}")
            self.feedback_sockets[mid] = socket
        self.load_sockets = []
        for load_socket in load_sockets:
            socket = context.socket(zmq.PULL)
            socket.bind(f"tcp://{load_socket}")
            self.load_sockets.append(socket)
        self.detokenization_socket = context.socket(zmq.PULL)
        if detokenization_socket:
            self.detokenization_socket.bind(f"tcp://{detokenization_socket}")
        self.load_lock = asyncio.Lock()
        self.idx2lock = {k: asyncio.Lock() for k in instance_ids}
        self.event_counter = 0
        self.load_list = [0 for _ in range(len(load_sockets))]
        self.model_instance_id_to_idx = {int(mid): idx for idx, mid in enumerate(instance_ids)}

    async def wait_model_ready(self):
        for socket in self.socket_dict.values():
            rec_ans = await self.detokenization_socket.recv_pyobj()
            assert isinstance(rec_ans, IdleReq), f"error recv type {type(rec_ans)}"
            assert rec_ans.dist_type == "decode", f"error dist type {rec_ans.dist_type}"
        logger.info("all decode instances are ready")

    async def get_max_space_instance(self):
        async with self.load_lock:
            min_load = min(self.load_list)
            min_idx = self.load_list.index(min_load)
            min_idx = self.instance_id_list[min_idx]
            max_space = self.max_tokens_per_instance - min_load
            return max_space, min_idx

    async def calculated_req_num(self, cum_tokens):
        max_space, max_idx = await self.get_max_space_instance()
        req_num = len(cum_tokens)
        for idx, tokens in enumerate(cum_tokens):
            if tokens > max_space:
                req_num = idx
        return req_num, max_idx

    async def two_phase_commit(self, src_idx, tgt_idx, commit_id, needed_tokens, req_info):
        pre_commit_req = SPDPreCommitReq(commit_id=commit_id, total_tokens=needed_tokens)
        commit_req = SPDCommitReq(commit_id=commit_id, source_instance=src_idx, req_info=req_info)
        async with self.idx2lock[tgt_idx]:
            self.socket_dict[tgt_idx].send_pyobj(pre_commit_req)
            resp: bool = await self.feedback_sockets[tgt_idx].recv_pyobj()
            if not resp:
                return False
            self.socket_dict[tgt_idx].send_pyobj(commit_req)
            resp: bool = await self.feedback_sockets[tgt_idx].recv_pyobj()
            return True

    async def assign_decode_instance(self, req):
        try:
            req = await req.json()
            req: SPDAssignReq = SPDAssignReq.from_http_obj(req)
        except:
            logger.error(f"error req {req}")
            return fastapi.responses.JSONResponse(status_code=400, content=dict(error="error req"))
        src_model_id = req.model_instance_id
        req_num, tgt_model_id = await self.calculated_req_num(req.cum_token)
        resp = await self.two_phase_commit(src_model_id, tgt_model_id, req.commit_id, req.cum_token[-1], req.req_info)
        json_resp = fastapi.responses.JSONResponse(
            status_code=200, content=dict(req_num=req_num, target_instance_id=tgt_model_id if resp else -1)
        )
        return json_resp

    async def _recv_from_socket(self, socket):
        message = await socket.recv_pyobj()  # 接收来自 socket 的消息
        return message  #

    async def _recv_from_sockets(self, sockets):
        tasks = {asyncio.create_task(self._recv_from_socket(socket)): socket for socket in sockets}
        while True:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                socket = tasks.pop(task)
                message = await task
                yield message
                new_task = asyncio.create_task(self._recv_from_socket(socket))
                tasks[new_task] = socket

    async def pull_load_loop(self):
        async for message in self._recv_from_sockets(self.load_sockets):
            assert isinstance(message, RouterLoadOut), f"error recv type {type(message)}"
            src_idx = message.model_instance_id
            load = message.load
            async with self.load_lock:
                src_idx = self.model_instance_id_to_idx[src_idx]
                self.load_list[src_idx] = load


def start_spd_schedule_process(
    spd_url,
    socket_urls,
    instance_ids,
    detokenization_socket,
    load_sockets,
    feedback_sockets,
    max_tokens,
    pipe_writer,
):
    import setproctitle

    setproctitle.setproctitle("lightllm:spd_schedule")
    scheduler = SPDScheduler(
        socket_urls=socket_urls,
        instance_ids=instance_ids,
        detokenization_socket=detokenization_socket,
        load_sockets=load_sockets,
        feedback_sockets=feedback_sockets,
        max_tokens=max_tokens,
    )
    app = fastapi.FastAPI()

    # lifespan
    @app.on_event("startup")
    async def startup_event():
        scheduler.init_async_resources()
        await scheduler.wait_model_ready()
        asyncio.create_task(scheduler.pull_load_loop())

    @app.post("/")
    async def assign_decode_instance(req: fastapi.Request):
        return await scheduler.assign_decode_instance(req)

    pipe_writer.send("init ok")
    spd_host, spd_port = spd_url.split(":")
    uvicorn.run(app, host=spd_host, port=int(spd_port), log_level="info", loop="uvloop")
