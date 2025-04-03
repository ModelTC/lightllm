import asyncio
import pickle
import websockets
import ujson as json
import socket
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.log_utils import init_logger
from lightllm.server.httpserver.manager import HttpServerManager

logger = init_logger(__name__)

async def timer_log(manager: HttpServerManager):
    while True:
        await asyncio.sleep(30)
        manager.first_time_costs.print_log("mean first cost")
        manager.per_token_costs.print_log("mean per token cost")
    return

async def pd_handle_loop(manager: HttpServerManager):
    """
    pd_handle_loop 主要负责与 pd master 进行注册连接，然后接收pd master发来的请求，然后
    将推理结果转发给 pd master进行处理。
    """
    # 创建转发队列
    manager.forwarding_queue = AsyncQueue()
    # 启动统计信息日志打印任务
    asyncio.create_task(timer_log(manager))
        
    manager.host_ip = get_hostname_ip()
    if manager.host_ip is None:
        manager.host_ip = manager.args.host

    while True:
        forwarding_tokens_task = None
        try:
            uri = f"ws://{manager.args.pd_master_ip}:{manager.args.pd_master_port}/pd_register"
            async with websockets.connect(uri, max_queue=(2048 * 1024, 2048 * 1023)) as websocket:
                
                sock = websocket.transport.get_extra_info("socket")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                args_dict = vars(manager.args)
                args_dict["host"] = manager.host_ip
                # 发送注册信息
                regist_json = {
                    "node_id": manager.args.pd_node_id,
                    "client_ip_port": f"{manager.host_ip}:{manager.args.port}",
                    "mode": manager.pd_mode.value,
                    "start_args": args_dict,
                }

                await websocket.send(json.dumps(regist_json))
                logger.info(f"Sent registration JSON: {regist_json}")

                # 转发token的task
                async def up_tokens_to_pd_master(forwarding_queue: AsyncQueue, websocket):
                    while True:
                        handle_list = await forwarding_queue.wait_to_get_all_data()
                        if handle_list:
                            await websocket.send(pickle.dumps((ObjType.TOKEN_PACKS, handle_list)))

                forwarding_tokens_task = asyncio.create_task(
                    up_tokens_to_pd_master(manager.forwarding_queue, websocket)
                )
                
                # 接收 pd master 发来的请求，并推理后，将生成的token转发回pd master。
                while True:
                    recv_bytes = await websocket.recv()
                    obj = pickle.loads(recv_bytes)
                    if obj[0] == ObjType.REQ:
                        prompt, sampling_params, multimodal_params = obj[1]

                        # 触发推理的task
                        async def pd_process_generate(
                            manager: HttpServerManager, prompt, sampling_params, multimodal_params
                        ):
                            try:
                                async for sub_req_id, request_output, metadata, finish_status in manager.generate(
                                    prompt, sampling_params, multimodal_params, None
                                ):
                                    # p d 模式下，将 token 数据放入到转发队列中, 请求id 小于0的请求是health探测请求，不用转发。
                                    is_health_check_req = sub_req_id < 0
                                    if not is_health_check_req:
                                        await manager.forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))

                            except BaseException as e:
                                logger.error(str(e))

                        asyncio.create_task(pd_process_generate(manager, prompt, sampling_params, multimodal_params))

                    elif obj[0] == ObjType.ABORT:
                        group_req_id = obj[1]
                        await manager.abort(group_req_id)
                    else:
                        logger.error(f"recevie error obj {str(obj)}")

        except Exception as e:
            logger.error("connetion to pd_master has error")
            logger.exception(str(e))
            if forwarding_tokens_task is not None:
                forwarding_tokens_task.cancel()
            await asyncio.sleep(10)
            await manager.forwarding_queue.get_all_data()
            logger.info("reconnection to pd_master")

