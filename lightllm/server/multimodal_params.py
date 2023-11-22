"""Multimodal parameters for text generation."""
from typing import List
import os
import requests
from io import BytesIO
from PIL import Image
import base64
# import hashlib
# import rpyc
import uuid


class ImageItem:

    def __init__(self, **kwargs):
        _type, _data = kwargs["type"], kwargs["data"]
        img_data = self.read(_type, _data)
        # the unique id for the image 
        self.uuid = self.uuid(img_data)
        # where should the image fill into the text embeds
        self.offset = -1
        # the length of the image embeds
        self.length = -1

    def read(self, _type, _data):
        try:
            if _type == "url":
                timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
                ret = requests.get(_data, timeout=timeout)
                img_data = ret.content
            elif _type == "base64":
                img_data = base64.b64decode(_data)
            else:
                raise ValueError(f"cannot read image which type is {_type}!")
    
            # check if valid image bytes
            image = Image.open(BytesIO(img_data))
            print(f"succeed to read image {_type} size={image.size}")
            return img_data
    
        except Exception as e:
            raise ValueError(f"Failed to read image type={_type}, data[:100]={_data[:100]}: {e}!")

    def uuid(self, img_data):
        # client = rpyc.connect("localhost", 2233)
        # image_uuid = client.root.add_item(img_data)
        image_uuid = uuid.uuid1()
        return image_uuid

    def to_dict(self):
        ret = {}
        ret["uuid"] = self.uuid
        ret["offset"] = self.offset
        ret["length"] = self.length
        return ret


class MultimodalParams:

    def __init__(
        self,
        images: List[dict] = [],
    ) -> None:
        self.images = [ImageItem(**i) for i in images]
        return

    def fill_offset_length_for_images(self, offsets, lengths):
        i_num, o_num, l_num = len(self.images), len(offsets), len(lengths)
        assert i_num == o_num, "Invalid image offsets: {} vs {}".format(i_num, o_num)
        assert i_num == l_num, "Invalid image lengths: {} vs {}".format(i_num, l_num)
        for i, o, l in zip(self.images, offsets, lengths):
            i.offset = o
            i.length = l 
    
    def verify(self):
        return

    def to_dict(self):
        ret = {}
        ret["images"] = [i.to_dict() for i in self.images]
        return ret
