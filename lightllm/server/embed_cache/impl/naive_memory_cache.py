import uuid
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

@CacheManagerFactory.register("naive")
class InMemoryCache(CacheManager):

    def __init__(self, pool_size=10) -> None:
        self._records = dict()
        self._md5_to_record = dict()
        self.pool_size = pool_size
        self.record_order = deque()

    def add_item(self, data: bytes, id: int=None) -> int:
        md5sum = hashlib.md5(data).hexdigest()
        id = self.query_item_uuid(md5sum)
        if id is not None:
            # this data already exists in cache.
            return id
        # data doesn't exist, insert it and return id.
        id = uuid.uuid1()
        id = id.int
        record = Record(id=id, data=data, md5sum=md5sum, embed=False)
        self._records[id] = record
        self._md5_to_record[md5sum] = record
        self.record_order.append(id)
        # if len(self.record_order) > self.pool_size:
        #     # Remove the earliest added entry from the cache
        #     oldest_id = self.record_order.popleft()
        #     record = self._records[oldest_id]
        #     uid = record.id
        #     if record.embed:
        #         shared_memory = shm.SharedMemory(name=str(uid))
        #         shared_memory.close()
        #         shared_memory.unlink()
        #     del self._md5_to_record[record.md5sum]
        #     del self._records[oldest_id]
        return id

    def query_item_uuid(self, md5sum) -> Union[int, None]:
        record = self._md5_to_record.get(md5sum, None)
        if record is None:
            return None
        return record.id
    
    def get_item_data(self, id: int) -> bytes:
        return self._records[id].data

    def set_item_embed(self, id: int):
        self._records[id].embed = True

    def get_item_embed(self, id: int, debug=False) -> bool:
        if debug:
            print(self._records.keys())
        return self._records[id].embed

    def recycle_item(self, id: int):
        record = self._records.get(id, None)
        if record is None:
            return
        # md5sum = record.md5sum
        # shared_memory = shm.SharedMemory(name=str(id))
        # shared_memory.close()
        # shared_memory.unlink()
        del self._records[id]
        del self._md5_to_record[md5sum]
