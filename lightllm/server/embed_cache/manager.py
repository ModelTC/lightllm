import asyncio
# asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
import rpyc
import uuid
from typing import Union

from .interface import CacheManager

class CacheServer(rpyc.Service):
    def __init__(self, manager_impl: CacheManager) -> None:
        super().__init__()
        self._impl = manager_impl

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_add_item(self, data: bytes) -> uuid.UUID:
        id = self._impl.add_item(data)
        assert isinstance(id, uuid.UUID)
        return id

    def exposed_query_item_uuid(self, md5sum) -> Union[uuid.UUID, None]:
        id = self._impl.query_item_uuid(md5sum)
        print(id)
        assert isinstance(id, uuid.UUID)
        return id

    def exposed_get_item_data(self, id: uuid.UUID) -> bytes:
        return self._impl.get_item_data(id=id)

    def exposed_set_item_embed(self, id: uuid.UUID, embed: bytes):
        return self._impl.set_item_embed(id=id, embed=embed)

    def exposed_get_item_embed(self, id: uuid.UUID) -> bytes:
        return self._impl.get_item_embed(id=id)

    def exposed_recycle_item(self, id: uuid.UUID):
        return self._impl.recycle_item(id)

def start_cache_manager(port: int):
    from .interface import CacheManagerFactory
    manager_cls = CacheManagerFactory.get_impl("naive")
    manager = manager_cls()
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(service, port=port)
    t.start()

if __name__ == "__main__":
    start_cache_manager(2233)