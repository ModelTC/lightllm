import asyncio
from typing import Dict, Optional, Tuple
from lightllm.server.core.objs.io_objs import GroupReqObjs
from lightllm.server.req_id_generator import MAX_BEST_OF


class OrderedRequestManager:
    def __init__(self):
        self.pending_requests = {}  # 缓存未按顺序到达的请求: {request_num: request_data}
        self.current_request_num = 0  # 下一个应该处理的请求的 request_num
        self.lock = asyncio.Lock()

    def _convert_group_id_to_request_num(self, group_req_id):
        return group_req_id // MAX_BEST_OF

    async def add_request(self, request_data: GroupReqObjs):
        """添加新请求到缓存"""
        request_num = self._convert_group_id_to_request_num(request_data.group_req_id)
        self.pending_requests[request_num] = request_data

    async def get_next_request(self) -> GroupReqObjs:
        """获取下一个应该处理的请求"""
        while True:
            if self.current_request_num not in self.pending_requests:
                await asyncio.sleep(0.01)
                continue
            else:
                request_data = self.pending_requests.pop(self.current_request_num)
                print(f"Get request {self.current_request_num}")
                self.current_request_num += 1
                return request_data
        return None
