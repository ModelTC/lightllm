from collections import defaultdict
from typing import Dict, List, Any
from torch import Tensor
from dataclasses import dataclass
import queue
import pickle
import time

from lightllm.utils.log_utils import init_logger

from .pd_remote_prefill_obj import (
    RemoteAgent,
    KVMoveRequest,
    PrefillRequest,
    RemotePrefillStatus,
    ThreadSafeDict,
    KVMoveRequestState,
    PageTransferAck,
    RemoteTransferStatusType,
    RemoteTransferType,
    NotificationType,
    Notification,
)


logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._api import nixlBind

    logger.info("Nixl is available")
except ImportError:
    logger.warning("nixl is not installed, which is required for pd disagreggation!!!")
    NixlWrapper = None


@dataclass
class NixlMetadata:
    id: str
    num_tokens: list[int]
    num_pages: list[int]
    agent_metadatas: list[bytes]
    agent_mem_descs: list[bytes]
    agent_page_mem_descs: list[bytes]


class NixlKVTransporter:
    def __init__(self, node_id: int, tp_idx: int):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.nixl_agent = NixlWrapper(self.agent_name, None)

        self.num_layers = -1
        self.num_tokens = -1
        self.num_heads = -1
        self.head_dims = -1
        self.token_len = -1
        self.num_pages = -1
        self.page_size = -1
        self.page_len = -1

        self.reg_desc = None
        self.local_xfer_handles = None
        self.page_reg_desc = None
        self.page_local_xfer_handles = None

        self.remote_agents = defaultdict(list)

        self.inflight_transfers: ThreadSafeDict = ThreadSafeDict()
        self.inflight_page_transfers: ThreadSafeDict = ThreadSafeDict()

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self):
        return self.nixl_agent.get_agent_metadata()

    @property
    def local_mem_desc(self):
        return self.nixl_agent.get_serialized_descs(self.reg_desc)

    @property
    def local_page_mem_desc(self):
        return self.nixl_agent.get_serialized_descs(self.page_reg_desc)

    def get_new_notifs(self):
        return self.nixl_agent.get_new_notifs()

    def _create_xfer_handles(self, reg_desc: nixlBind.nixlRegDList, num_tokens: int, agent_name: str = ""):
        base_addr, _, device_id, _ = reg_desc[0]
        layer_len = num_tokens * self.token_len
        tokens_data = [0] * (self.num_layers * num_tokens)
        idx = 0
        for layer_id in range(self.num_layers):
            for token_id in range(num_tokens):
                tokens_data[idx] = (
                    base_addr + layer_id * layer_len + token_id * self.token_len,
                    self.token_len,
                    device_id,
                )
                idx += 1
        descs = self.nixl_agent.get_xfer_descs(tokens_data, "VRAM", True)
        return self.nixl_agent.prep_xfer_dlist(agent_name, descs, is_sorted=True)

    def register_kv_buffer(self, kv_buffer: Tensor):
        self.num_layers, self.num_tokens, self.num_heads, self.head_dim = kv_buffer.shape
        self.token_len = self.num_heads * self.head_dim * kv_buffer.element_size()

        self.reg_desc = self.nixl_agent.register_memory(kv_buffer)
        self.local_xfer_handles = self._create_xfer_handles(self.reg_desc, self.num_tokens)

    def _create_paged_xfer_handles(self, reg_desc: nixlBind.nixlRegDList, page_num: int, agent_name: str = ""):
        base_addr, _, device_id, _ = reg_desc[0]
        pages_data = []
        for page_id in range(page_num):
            pages_data.append((base_addr + page_id * self.page_len, self.page_len, device_id))
        descs = self.nixl_agent.get_xfer_descs(pages_data, "VRAM", True)
        return self.nixl_agent.prep_xfer_dlist(agent_name, descs, is_sorted=True)

    def register_kv_move_buffer(self, kv_move_buffer: Tensor):
        self.num_pages, self.page_size, _, _, _ = kv_move_buffer.shape
        self.page_len = self.page_size * self.num_layers * self.token_len
        self.page_reg_desc = self.nixl_agent.register_memory(kv_move_buffer)
        self.page_local_xfer_handles = self._create_paged_xfer_handles(self.page_reg_desc, self.num_pages)

    def add_remote_agent(self, remote_agent: NixlMetadata):
        for idx, (agent_metadata, num_tokens, num_pages, agent_mem_desc, agent_page_mem_desc) in enumerate(
            zip(
                remote_agent.agent_metadatas,
                remote_agent.num_tokens,
                remote_agent.num_pages,
                remote_agent.agent_mem_descs,
                remote_agent.agent_page_mem_descs,
            )
        ):
            if self.tp_idx != idx:
                self.remote_agents[remote_agent.id].append(None)
                continue

            peer_name = self.nixl_agent.add_remote_agent(agent_metadata)
            if isinstance(peer_name, bytes):
                peer_name = peer_name.decode()

            self.nixl_agent.send_notif(
                peer_name, Notification(type=NotificationType.REMOTE_MD, data=self.agent_metadata).to_bytes()
            )

            mem_desc = self.nixl_agent.deserialize_descs(agent_mem_desc)
            kv_xfer_handles = self._create_xfer_handles(mem_desc, num_tokens, agent_name=peer_name)

            page_mem_desc = self.nixl_agent.deserialize_descs(agent_page_mem_desc)
            kv_page_xfer_handles = self._create_paged_xfer_handles(page_mem_desc, num_pages, agent_name=peer_name)

            logger.info("Added remote agent %s with mem desc %s", peer_name, page_mem_desc)
            self.remote_agents[remote_agent.id].append(
                RemoteAgent(
                    name=peer_name,
                    kv_mem_desc=mem_desc,
                    num_tokens=num_tokens,
                    kv_xfer_handles=kv_xfer_handles,
                    kv_page_mem_desc=page_mem_desc,
                    num_pages=num_pages,
                    kv_page_xfer_handles=kv_page_xfer_handles,
                )
            )

    def connect_to_remote(self, name: str, remote_md: bytes):
        target = self.nixl_agent.add_remote_agent(remote_md)
        if isinstance(target, bytes):
            target = target.decode()
        assert name == target, "Target name {} does not match remote name {}".format(target, name)

    def _get_token_desc_ids(self, token_ids: List[int], num_tokens: int):
        token_ids_len, idx = len(token_ids), 0
        descs_ids = [0] * (self.num_layers * token_ids_len)
        for layer_id in range(self.num_layers):
            for token_id in token_ids:
                descs_ids[idx] = layer_id * num_tokens + token_id
                idx += 1
        return descs_ids

    def write_blocks(self, request: KVMoveRequest, prefill_request: PrefillRequest, is_finished: bool):
        group_reqeust_id = request.group_req_id
        skip_kv_move_len = prefill_request.data.local_cached_len

        # current kv len is less than remote cached kv len, just skip
        if request.cur_kv_len <= skip_kv_move_len:
            return

        kv_move_start = max(skip_kv_move_len, request.prev_kv_len)
        kv_move_end = request.cur_kv_len

        src_token_ids = request.token_ids[kv_move_start:]
        dst_token_ids = prefill_request.data.token_ids[
            kv_move_start - skip_kv_move_len : kv_move_end - skip_kv_move_len
        ]

        remote_agent: RemoteAgent = self.remote_agents[prefill_request.decode_id][
            self.tp_idx
        ]  # TODO one-one mapping now

        if len(src_token_ids) > 0:
            assert len(src_token_ids) == len(dst_token_ids), (
                f"{len(src_token_ids)} {len(dst_token_ids)} {kv_move_start} "
                f"{kv_move_end} {skip_kv_move_len}, {len(prefill_request.data.token_ids)}"
            )
            src_token_descs = self._get_token_desc_ids(src_token_ids, self.num_tokens)
            dst_token_descs = self._get_token_desc_ids(dst_token_ids, remote_agent.num_tokens)

            src_handle = self.local_xfer_handles
            dst_handle = remote_agent.kv_xfer_handles

            notify_status = (
                RemotePrefillStatus(
                    group_req_id=group_reqeust_id,
                    status=1,
                    chunk_id=prefill_request.transfer_state.current_chunk_id,
                    is_last=is_finished,
                ).serialize()
                if is_finished
                else b""
            )

            handle = self.nixl_agent.make_prepped_xfer(
                "WRITE", src_handle, src_token_descs, dst_handle, dst_token_descs, notify_status
            )

            status = self.nixl_agent.transfer(handle)
            assert status != "ERR"

            if group_reqeust_id not in self.inflight_transfers:
                self.inflight_transfers[group_reqeust_id] = KVMoveRequestState(
                    handles=[], done_handles=[], remote_agent=remote_agent, abort=False, is_last_arrived=False
                )

            self.inflight_transfers[group_reqeust_id].handles.append(handle)

            if is_finished:
                self.inflight_transfers[group_reqeust_id].is_last_arrived = True

            return handle

        return None

    def write_blocks_paged(
        self,
        remote_id: int,
        transfer_pages: List[int],
        receive_pages: List[int],
        notifications: List[RemotePrefillStatus],
    ):
        remote_agent: RemoteAgent = self.remote_agents[remote_id][self.tp_idx]
        src_handle = self.page_local_xfer_handles
        dst_handle = remote_agent.kv_page_xfer_handles
        notify_status = Notification(type=NotificationType.TRANSFER_NOTIFY, data=[n.serialize() for n in notifications])
        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE", src_handle, transfer_pages, dst_handle, receive_pages, notify_status.to_bytes()
        )
        status = self.nixl_agent.transfer(handle)
        assert status != "ERR", f"Transfer failed with status {status} for handle {handle}"
        self.inflight_page_transfers[handle] = (transfer_pages, receive_pages, notifications, remote_agent)

    def send_transfer_notify(self, agent_name: str, acks: List[PageTransferAck]):
        assert len(acks) > 0, "Acks should not be empty"
        acks_noti = Notification(type=NotificationType.TRANSFER_NOTIFY_ACK, data=[ack.serialize() for ack in acks])
        self.nixl_agent.send_notif(agent_name, acks_noti.to_bytes())

    def send_abort_notify(self, remote_id: int, group_req_id: int):
        remote_agent: RemoteAgent = self.remote_agents[remote_id][self.tp_idx]
        notify_status = RemotePrefillStatus(
            group_req_id=group_req_id,
            transfer_type=RemoteTransferType.PAGE_TRANSFER,
            status=RemoteTransferStatusType.FAILED,
            is_last=True,
        )
        self.nixl_agent.send_notif(
            remote_agent.name,
            Notification(type=NotificationType.TRANSFER_NOTIFY, data=[notify_status.serialize()]).to_bytes(),
        )

        if group_req_id in self.inflight_transfers:
            self.inflight_transfers[group_req_id].abort = True

    async def get_done_page_transfers(self):
        done_pages = []
        done_requests = []
        for handle, (transfer_pages, _, notifications, _) in self.inflight_page_transfers.items():
            xfer_state = self.nixl_agent.check_xfer_state(handle)
            if xfer_state == "DONE":
                done_pages.extend(transfer_pages)
                done_requests.extend(
                    [(x.group_req_id, RemoteTransferStatusType.SUCCESS) for x in notifications if x.is_last]
                )
                self.nixl_agent.release_xfer_handle(handle)
                del self.inflight_page_transfers[handle]

            elif xfer_state == "PROC":
                continue
            else:
                logger.warning(f"Transfer failed with state {xfer_state} for handle {handle}")
                done_pages.extend(transfer_pages)
                done_requests.extend([(x.group_req_id, RemoteTransferStatusType.FAILED) for x in notifications])
                self.nixl_agent.release_xfer_handle(handle)
                del self.inflight_page_transfers[handle]

        return done_pages, done_requests

    def get_done_tranfers(self):
        done_req_ids = []
        for req_id, kv_move_state in self.inflight_transfers.items():
            kv_move_state: KVMoveRequestState
            if kv_move_state.abort:
                logger.warning(f"{req_id} Transfer aborted")
                done_req_ids.append((req_id, -1))
                continue

            if not kv_move_state.is_last_arrived:
                continue

            remote_agent: RemoteAgent = kv_move_state.remote_agent

            left_handles = []
            failed = False
            for handle in kv_move_state.handles:
                if failed:
                    left_handles.append(handle)
                    continue

                xfer_state = self.nixl_agent.check_xfer_state(handle)

                if xfer_state == "DONE":
                    kv_move_state.done_handles.append(handle)
                elif xfer_state == "PROC":
                    left_handles.append(handle)
                else:
                    logger.warning(f"{req_id} Transfer failed with state {xfer_state}")
                    failed = True
                    kv_move_state.done_handles.append(handle)
                    notify_failed_status = RemotePrefillStatus(
                        group_req_id=req_id, status=-1, chunk_id=-1, is_last=True
                    )
                    self.nixl_agent.send_notif(remote_agent.name, notify_failed_status.serialize())

            kv_move_state.handles = left_handles

            if failed:
                done_req_ids.append((req_id, -1))
            elif len(left_handles) == 0:
                done_req_ids.append((req_id, 1))

        for req_id, _ in done_req_ids:
            kv_move_state: KVMoveRequestState = self.inflight_transfers[req_id]
            for handle in kv_move_state.handles + kv_move_state.done_handles:
                # release will abort inflight transfer
                self.nixl_agent.release_xfer_handle(handle)

            del self.inflight_transfers[req_id]
        return done_req_ids

    def shutdown(self):
        self.nixl_agent.deregister_memory(self.reg_desc)
        self.nixl_agent.release_dlist_handle(self.local_xfer_handles)
        self.nixl_agent.release_dlist_handle(self.page_local_xfer_handles)
        for id, agents in self.remote_agents.items():
            for agent in agents:
                self.nixl_agent.remove_remote_agent(agent.name)
                self.nixl_agent.release_dlist_handle(agent.kv_xfer_handles)
                self.nixl_agent.release_dlist_handle(agent.kv_page_xfer_handles)

        for handle in self.inflight_page_transfers:
            self.nixl_agent.release_xfer_handle(handle)
