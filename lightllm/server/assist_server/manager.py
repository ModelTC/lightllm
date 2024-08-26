import zmq
import zmq.asyncio
from ..io_struct import Req, BatchTokenIdOut, NormalReq, FinishStatus

import aiohttp
import asyncio
from lightllm.utils.log_utils import init_logger
import json
import copy

logger = init_logger(__name__)


class DispatcherManager:
    def __init__(self, args, assist_router, router_port, detokenization_port):
        self.args = args
        self.dispatch_threshold = args.dispatch_threshold
        self.dispatch_host = args.dispatch_host
        self.dispatch_port = args.dispatch_port

        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{assist_router}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.dispatch_count = 0
        self.token_count = 0

        self.semaphore = asyncio.Semaphore(25)

        return

    async def _handle_dispatch_req(self, req: Req):
        async with self.semaphore:
            _max_new_tokens = req.sample_params.max_new_tokens - req.cur_output_len
            req.sample_params.max_new_tokens = 1
            input_ids = [int(input_id) for input_id in req.prompt_ids[:-1]]
            data = {
                "inputs": input_ids,
                "parameters": req.sample_params.to_dict(),
            }
            req.sample_params.max_new_tokens = _max_new_tokens
            url = f"http://{self.dispatch_host}:{self.dispatch_port}/id_generate"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers={"Content-Type": "application/json"}, data=json.dumps(data)
                ) as response:
                    new_token_id = req.prompt_ids[-1]
                    is_ok = False
                    if response.status == 200:
                        result = await response.text()
                        if "out_token_ids" in result:
                            result = json.loads(result)
                            is_ok = True
                            new_token_id = result["out_token_ids"][0]

                    if not is_ok:
                        logger.error(f"dispatch error, status code: {response.status}")

                    new_token_id = int(new_token_id)
                    req.prompt_ids[-1] = new_token_id
                    req.sample_params.max_new_tokens = _max_new_tokens
                    batch_out = BatchTokenIdOut()
                    if req.sample_params.max_new_tokens > 0:
                        new_req = NormalReq(
                            req.request_id, copy.deepcopy(req.prompt_ids), req.sample_params, req.multimodal_params
                        )
                        batch_out.reqs_infs.append((new_req.request_id, new_token_id, {"id": new_token_id}, 0))
                        self.send_to_router.send_pyobj(new_req)
                        self.send_to_detokenization.send_pyobj(batch_out)
                    else:
                        batch_out.reqs_infs.append(
                            (new_req.request_id, new_token_id, {"id": new_token_id}, FinishStatus.FINISHED_LENGTH)
                        )

    async def loop_for_netio_dispatch_req(self):
        while True:
            recv_req = await self.recv_from_router.recv_pyobj()
            if isinstance(recv_req, Req):
                self.dispatch_count += 1
                self.token_count += recv_req.cur_output_len
                asyncio.create_task(self._handle_dispatch_req(recv_req))
            elif isinstance(recv_req, tuple):
                self.token_count += recv_req[0]
            else:
                assert False, f"Error Req Inf {recv_req}"

            logger.info(f"Dispatch prob: {self.dispatch_count/self.token_count if self.token_count != 0 else 0}")
        return


def start_dispatcher_process(args, assist_router, router_port, detokenization_port, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        dispatcher = DispatcherManager(
            args,
            assist_router=assist_router,
            router_port=router_port,
            detokenization_port=detokenization_port,
        )

    except:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dispatcher.loop_for_netio_dispatch_req())
    return
