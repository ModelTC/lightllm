import rpyc
import os
import PIL
import torch
from lightllm.server.embed_cache.utils import tensor2bytes, bytes2tensor


client = rpyc.connect("localhost", 10004)

dirname = os.path.dirname(__file__)
data = open(os.path.join(dirname, "AthensAcropolis.jpg"), "rb").read()
# add data to cache.
id = client.root.add_item(data)

# query data by md5sum, check the id is same.
import hashlib
md5 = hashlib.md5(data).hexdigest()
id_queried = client.root.query_item_uuid(md5)
assert id == id_queried, "Queried uuid differs from original id"

# get data back and check the data is same.
data_retrive = client.root.get_item_data(id)
assert data_retrive == data, "the data retrived is different from inserted"

# insert embed
embed = torch.rand(32, 10240).to(torch.float32)
print(embed)
embed_bytes = tensor2bytes(embed)

client.root.set_item_embed(id, embed_bytes)
print("set", id, embed.shape, embed.dtype, embed.device)

# get embed and check embed is the same.
embed_retrived = client.root.get_item_embed(id)
embed_retrived = bytes2tensor(embed_retrived)
assert torch.allclose(embed, embed_retrived), "the embed retrived is different from inserted"
# print("end with sleep 100")
# import time
# time.sleep(20)
