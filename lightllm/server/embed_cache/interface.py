import uuid
from typing import Union

class CacheManager(object):
    ''' Defines the interface of embedding cache manager.
    '''
    def __init__(self) -> None:
        pass

    def add_item(self, data: bytes) -> uuid.UUID:
        pass

    def query_item_uuid(self, md5sum) -> Union[uuid.UUID, None]:
        ''' Query the uuid of certain data with its md5sum, if not exists, return None.
        '''
        pass

    def get_item_data(self, id: uuid.UUID) -> bytes:
        pass

    def set_item_embed(self, id: uuid.UUID, embed: bytes):
        pass

    def get_item_embed(self, id: uuid.UUID) -> bytes:
        pass

    def recycle_item(self, id: uuid.UUID):
        pass

class CacheManagerFactory(object):
    _impls = dict()

    @classmethod
    def register(cls, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in cls._impls:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            cls._impls[key] = value
            return value

        if callable(target):            # 如果传入的目标可调用，说明之前没有给出注册名字，我们就以传入的函数或者类的名字作为注册名
            return add_register_item(target.__name__, target)
        else:                           # 如果不可调用，说明额外说明了注册的可调用对象的名字
            return lambda x : add_register_item(target, x)
    
    @classmethod
    def get_impl(cls, name: str):
        return cls._impls[name]