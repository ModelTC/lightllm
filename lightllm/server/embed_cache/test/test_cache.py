import rpyc
import os
import PIL

client = rpyc.connect("localhost", 2233)

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
import torch
embed = torch.rand(32, 10240).to(torch.float32)
client.root.set_item_embed(id, embed)

# get embed and check embed is the same.
embed_retrived = client.root.get_item_embed(id)
assert torch.allclose(embed, embed_retrived), "the embed retrived is different from inserted"