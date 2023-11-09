import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlamaInfer(unittest.TestCase):

    def test_llama_infer(self):
        from lightllm.models.llama.model import LlamaTpPartModel
        test_model_inference(world_size=1, 
                             model_dir="/nvme/baishihao/llama-7b", 
                             model_class=LlamaTpPartModel, 
                             batch_size=16, 
                             input_len=688, 
                             output_len=100,
                             mode=[])
        return


if __name__ == '__main__':
    unittest.main()