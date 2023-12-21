import uuid
import threading
import dataclasses
from ..interface import CacheManager, CacheManagerFactory
from typing import Union
import torch
import hashlib
from collections import deque
import multiprocessing.shared_memory as shm
from ..utils import get_shm_name_data, get_shm_name_embed, free_shm


@dataclasses.dataclass
class Record(object):
    id: int
    md5sum: str
    ref: int
    data: bool
    embed: bool

@CacheManagerFactory.register("naive")
class InMemoryCache(CacheManager):

    def __init__(self, max_pool_size=1000) -> None:
        self._records = dict()
        self._md5_to_record = dict()
        self.max_pool_size = max_pool_size
        self.pool_size = 0
        self.lock = threading.Lock()

    def add_item(self, md5sum: str, ref: int) -> int:
        with self.lock:
            if md5sum not in self._md5_to_record:
                id = uuid.uuid1()
                id = id.int
                record = Record(id=id, md5sum=md5sum, ref=ref, data=False, embed=False)
                self._records[id] = record
                self._md5_to_record[md5sum] = record
                self.pool_size += 1
                return id
            else:
                record = self._md5_to_record[md5sum]
                record.ref += ref
                return record.id

    def set_item_data(self, id: int) -> None:
        self._records[id].data = True

    def get_item_data(self, id: int) -> bool:
        return self._records[id].data

    def set_item_embed(self, id: int) -> None:
        self._records[id].embed = True

    def get_item_embed(self, id: int) -> bool:
        return self._records[id].embed

    def free_item(self, id: int) -> None:
        with self.lock:
            self._records[id].ref -= 1

    def recycle_item(self, ratio: float) -> None:
        reserved_size = max(0, int(self.max_pool_size * ratio))
        target_delete_size = max(1, self.pool_size - reserved_size)
        with self.lock:
            if self.pool_size >= self.max_pool_size:
                deleted = 0
                keys = list(self._records.keys())
                for id in keys:
                    record = self._records[id]
                    if record.ref <= 0:
                        if record.data:
                            free_shm(get_shm_name_data(id))
                        if record.embed:
                            free_shm(get_shm_name_embed(id))
                        del self._md5_to_record[record.md5sum]
                        del self._records[id] 
                        self.pool_size -= 1
                        deleted += 1
                        if deleted >= target_delete_size:
                            break
