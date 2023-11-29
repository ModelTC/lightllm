from lightllm.server.embed_cache.interface import CacheManagerFactory
from lightllm.server.embed_cache.manager import CacheServer
from rpyc.utils.server import ThreadedServer

port = 10004
print("start_cache_manager:")

manager_cls = CacheManagerFactory.get_impl("naive")
manager = manager_cls()
service = CacheServer(manager)
t = ThreadedServer(service, port=port)
t.start()

print("end_cache_manager:")
