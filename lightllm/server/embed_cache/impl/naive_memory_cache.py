import uuid
import dataclasses
from ..interface import CacheManager, CacheManagerFactory
from typing import Union
import torch
import hashlib

@dataclasses.dataclass
class Record(object):
    id: uuid.UUID
    data: bytes
    md5sum: str
    embed: torch.Tensor

@CacheManagerFactory.register("naive")
class InMemoryCache(CacheManager):

    def __init__(self) -> None:
        self._records = dict()
        self._md5_to_record = dict()

    def add_item(self, data: bytes, id: uuid.UUID=None) -> uuid.UUID:
        md5sum = hashlib.md5(data).hexdigest()
        id = self.query_item_uuid(md5sum)
        if id is not None:
            # this data already exists in cache.
            return id
        # data doesn't exist, insert it and return id.
        id = uuid.uuid1()
        record = Record(id=id, data=data, md5sum=md5sum, embed=None)
        self._records[id] = record
        self._md5_to_record[md5sum] = record
        return id

    def query_item_uuid(self, md5sum) -> Union[uuid.UUID, None]:
        record = self._md5_to_record.get(md5sum, None)
        if record is None:
            return None
        return record.id

    def get_item_data(self, id: uuid.UUID) -> bytes:
        return self._records[id].data

    def set_item_embed(self, id: uuid.UUID, embed: bytes):
        record = self._records[id]
        record.embed = embed

    def get_item_embed(self, id: uuid.UUID) -> bytes:
        record = self._records[id]
        if not record:
            return None
        return record.embed

    def recycle_item(self, id: uuid.UUID):
        record = self._records.get(id, None)
        if record is None:
            return
        md5sum = record.md5sum
        del self._records[id]
        del self._md5_to_record[md5sum]