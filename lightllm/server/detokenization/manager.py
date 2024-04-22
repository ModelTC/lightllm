import copy
import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from ..io_struct import BatchTokenIdOut, ReqDetokenizationState, BatchStrOut, AbortReq, FinishStatus
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
        detokenization_port,
        httpserver_port,
        trust_remote_code,
    ):
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{detokenization_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")

        self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        self.all_special_ids = set(self.tokenizer.all_special_ids)
        self.req_id_to_out = {}
        self.eos_id = eos_id

    async def handle_loop(self):
        while True:
            try:
                recv_obj: Union(
                    BatchTokenIdOut, ReqDetokenizationState, AbortReq
                ) = await self.recv_from_router.recv_pyobj()
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


def start_detokenization_process(args, detokenization_port, httpserver_port, pipe_writer):
    try:
        router = DeTokenizationManager(
            args.eos_id,
            args.model_dir,
            args.tokenizer_mode,
            detokenization_port=detokenization_port,
            httpserver_port=httpserver_port,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise
    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.handle_loop())
    return
