import uuid
import threading
import dataclasses
from ..interface import CacheManager, CacheManagerFactory
from typing import Union
import torch
import hashlib
from collections import deque
import multiprocessing.shared_memory as shm


@dataclasses.dataclass
class Record(object):
    id: int
    data: bytes
    md5sum: str
    embed: bool
    embed_cnt: int
    # data_cnt: int

@CacheManagerFactory.register("naive")
class InMemoryCache(CacheManager):

    def __init__(self, max_pool_size=1000) -> None:
        self._records = dict()
        self._md5_to_record = dict()
        self.max_pool_size = max_pool_size
        self.pool_size = 0
        self.lock = threading.Lock()

    def add_item(self, data: bytes, id: int=None) -> int:
        md5sum = hashlib.md5(data).hexdigest()
        id = self.query_item_uuid(md5sum)
        if id is not None:
            # this data already exists in cache.
            # with self.lock:
                # self._records[id].data_cnt += 1
                # self._records[id].embed_cnt += 1
            return id
        # data doesn't exist, insert it and return id.
        id = uuid.uuid1()
        id = id.int
        record = Record(id=id, data=data, md5sum=md5sum, embed=False, embed_cnt=0)
        self._records[id] = record
        self._md5_to_record[md5sum] = record
        return id

    def query_item_uuid(self, md5sum) -> Union[int, None]:
        with self.lock:
            record = self._md5_to_record.get(md5sum, None)
            if record is None:
                return None
            self._records[record.id].embed_cnt += 1
            return record.id
    
    def get_item_data(self, id: int) -> bytes:
        # with self.lock:
        return self._records[id].data

    def query_available_size(self):
        # with self.lock:
        return self.max_pool_size - self.pool_size

    def free_item(self, id: int):
        with self.lock:
            self._records[id].embed_cnt -= 1    
            # print("free: ", id, self._records[id].embed_cnt) 
            return self.pool_size

    def set_item_embed(self, id: int):
        with self.lock:
            self._records[id].embed = True
            self._records[id].embed_cnt += 1
            # self._records[id].data_cnt -= 1
            self.pool_size += 1 

    def get_item_embed(self, id: int) -> bool:
        # with self.lock:
        if id not in self._records:
            return False
        return self._records[id].embed

    def recycle_item(self):
        with self.lock:
            if self.pool_size >= self.max_pool_size:
                old_keys = list(self._records.keys())
                for id in old_keys:
                    if self._records[id].embed_cnt == 0 and self._records[id].embed: #and self._records[id].data_cnt ==0:
                        if self._records[id].embed:
                            shared_memory = shm.SharedMemory(name=str(id))
                            shared_memory.close()
                            shared_memory.unlink()    
                        del self._md5_to_record[self._records[id].md5sum]
                        del self._records[id] 
                    self.pool_size -= 1