import os
import sys
import unittest
from multimodal_infer import test_multimodal_inference 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestLlavaInfer(unittest.TestCase):

    def test_llava_infer(self):
        from lightllm.models.llava.model import LlavaTpPartMulitModal
        test_multimodal_inference(world_size=1, 
                             model_dir="/path/to/llava-v1.5-7b", 
                             model_class=LlavaTpPartMulitModal, 
                             batch_size=20, 
                             input_len=1024, 
                             output_len=1024,
                             mode=[],
                             multimodal_kwargs={
                                 "image_path": "/path/to/image",
                                 "offset": 10,
                                 "pad_len": 575, 
                             })
        return


if __name__ == '__main__':
    unittest.main()
