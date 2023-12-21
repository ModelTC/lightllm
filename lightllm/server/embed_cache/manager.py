import asyncio
# asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
import rpyc
import uuid
from typing import Union

from .interface import CacheManager
from rpyc.utils.classic import obtain


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

    def exposed_add_item(self, md5sum: str, ref: int) -> int:
        md5sum = obtain(md5sum)
        ref = obtain(ref)
        id = self._impl.add_item(md5sum, ref)
        assert isinstance(id, int)
        return id

    def exposed_set_item_data(self, id: int) -> None:
        id = obtain(id)
        return self._impl.set_item_data(id=id)

    def exposed_get_item_data(self, id: int) -> bool:
        id = obtain(id)
        return self._impl.get_item_data(id=id)

    def exposed_set_item_embed(self, id: int) -> None:
        id = obtain(id)
        return self._impl.set_item_embed(id=id)

    def exposed_get_item_embed(self, id: int) -> bool:
        id = obtain(id)
        return self._impl.get_item_embed(id=id)

    def exposed_free_item(self, id: int) -> None:
        id = obtain(id)
        return self._impl.free_item(id)

    def exposed_recycle_item(self, ratio: float):
        ratio = obtain(ratio)
        return self._impl.recycle_item(ratio)

def start_cache_manager(port: int, pipe_writer):
    from .interface import CacheManagerFactory
    manager_cls = CacheManagerFactory.get_impl("naive")
    manager = manager_cls()
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(service, port=port)
    pipe_writer.send('init ok')
    t.start()

if __name__ == "__main__":
    start_cache_manager(2233)
