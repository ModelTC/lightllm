import copy
import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from ..io_struct import BatchTokenIdOut, IdleReq, ReqDetokenizationState, BatchStrOut, AbortReq, FinishStatus
from ..req_id_generator import convert_sub_id_to_group_id
from typing import Union
from .decode import decode_token
from ..tokenizer import get_tokenizer
import traceback

from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class DeTokenizationManager:
    def __init__(
        self,
        eos_id,
        model_weightdir,
        tokenizor_mode,
        router_to_detokenization_urls,
        detokenization_to_httpserver_url,
        detokenization_to_spd_url,
        trust_remote_code,
    ):
        context = zmq.asyncio.Context(len(router_to_detokenization_urls) + 2)
        self.recv_from_routers = []
        for detokenzation_url in router_to_detokenization_urls:
            recv_from_router = context.socket(zmq.PULL)
            recv_from_router.bind(f"tcp://{detokenzation_url}")
            self.recv_from_routers.append(recv_from_router)

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://{detokenization_to_httpserver_url}")

        if detokenization_to_spd_url is not None:
            self.send_to_spd = context.socket(zmq.PUSH)
            self.send_to_spd.connect(f"tcp://{detokenization_to_spd_url}")

        self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        self.all_special_ids = set(self.tokenizer.all_special_ids)
        self.req_id_to_out = {}
        self.eos_id = eos_id
        self._init_get_token_id_to_token_str()

    def _init_get_token_id_to_token_str(self):
        self.token_id_to_token = {token_id: token for token, token_id in self.tokenizer.get_vocab().items()}
        return

    async def wait_router_ready(self):
        for recv_from_router in self.recv_from_routers:
            recv_ans: IdleReq = await recv_from_router.recv_pyobj()
            assert isinstance(recv_ans, IdleReq), f"error recv type {type(recv_ans)}"
            if recv_ans.dist_type in ["prefill", "normal"]:
                self.send_to_httpserver.send_pyobj(recv_ans)
            elif recv_ans.dist_type == "decode":
                self.send_to_spd.send_pyobj(recv_ans)
            else:
                raise ValueError(f"error dist_type {recv_ans.dist_type}")
        logger.info("all the routers are ready !!!")

    async def _recv_from_socket(self, socket):
        message = await socket.recv_pyobj()  # 接收来自 socket 的消息
        return message  # 每次接收到消息就 yield 出去

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

    async def handle_loop(self):
        try:
            async for recv_obj in self._recv_from_sockets(self.recv_from_routers):
                assert isinstance(
                    recv_obj, (BatchTokenIdOut, ReqDetokenizationState, AbortReq)
                ), f"type is not right {type(recv_obj)}"

                assert isinstance(
                    recv_obj, (BatchTokenIdOut, ReqDetokenizationState, AbortReq)
                ), f"type is not right {type(recv_obj)}"
                if isinstance(recv_obj, ReqDetokenizationState):
                    # 将解序列对象复制 best_of 份， 并为其生成请求id
                    for delta_id in range(recv_obj.best_of):
                        recv_obj.request_id = recv_obj.group_req_id + delta_id
                        self.req_id_to_out[recv_obj.request_id] = copy.deepcopy(recv_obj)

                if isinstance(recv_obj, AbortReq):
                    delete_group_req_id = recv_obj.group_req_id
                    del_sub_req_ids = []
                    for sub_req_id in self.req_id_to_out.keys():
                        if convert_sub_id_to_group_id(sub_req_id) == delete_group_req_id:
                            del_sub_req_ids.append(sub_req_id)
                    try:
                        for del_id in del_sub_req_ids:
                            del self.req_id_to_out[del_id]
                    except:
                        pass

                if isinstance(recv_obj, BatchTokenIdOut):
                    new_batch_str_out = BatchStrOut()
                    for req_id, new_token_id, new_gen_metadata, finish_status in recv_obj.reqs_infs:
                        if req_id not in self.req_id_to_out:
                            continue

                        req_out: ReqDetokenizationState = self.req_id_to_out[req_id]
                        req_out.output_ids.append(new_token_id)
                        new_gen_metadata["special"] = new_token_id in self.all_special_ids
                        new_gen_metadata["count_output_tokens"] = len(req_out.output_ids)
                        req_out.gen_metadata.update(new_gen_metadata)

                        out_text = decode_token(
                            self.tokenizer,
                            req_out,
                            new_token_id,
                            self.eos_id,
                        )

                        if out_text.endswith("\ufffd"):
                            new_text = ""
                        elif "prefix_str" in new_gen_metadata:
                            # 对应 token_healing 的特殊处理
                            token = new_gen_metadata["prefix_str"]
                            token_str = self.tokenizer.convert_tokens_to_string([token])
                            new_text = out_text[len(req_out.output_str) :]
                            logger.info(
                                f"token headling prefix_token_and_str: '{token}':'{token_str}' new_text: '{new_text}'"
                            )
                            assert new_text.startswith(token_str)
                            new_text = new_text[len(token_str) :]
                            req_out.output_str = out_text
                        else:
                            new_text = out_text[len(req_out.output_str) :]
                            req_out.output_str = out_text
                        new_batch_str_out.reqs_infs.append((req_id, new_text, new_gen_metadata, finish_status))
                        if FinishStatus(finish_status).is_finished():
                            try:
                                del self.req_id_to_out[req_id]
                            except:
                                pass
                    self.send_to_httpserver.send_pyobj(new_batch_str_out)
        except Exception as e:
            logger.error(f"detoken process has exception {str(e)}")
            traceback.print_exc()
            pass


def start_detokenization_process(
    args, router_to_detokenization_urls, detokenization_to_httpserver_url, detokenization_to_spd_url, pipe_writer
):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = DeTokenizationManager(
            args.eos_id,
            args.model_dir,
            args.tokenizer_mode,
            router_to_detokenization_urls=router_to_detokenization_urls,
            detokenization_to_httpserver_url=detokenization_to_httpserver_url,
            detokenization_to_spd_url=detokenization_to_spd_url,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise
    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.wait_router_ready())
    loop.run_until_complete(router.handle_loop())
    return
