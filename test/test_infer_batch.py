from lightllm.server.router.model_infer.infer_batch import InferBatch, MemoryManager
from lightllm.common.infer_utils import init_bloc
import unittest
import torch
import numpy as np
import functools

test_targets = ['cpu',]

if torch.cuda.is_available():
    test_targets.append('cuda')


class TestInferBatch(unittest.TestCase):

    def test_init(self):
        for target in test_targets:
            reqs = [self.get_test_request(request_id=0, length=233),
                    self.get_test_request(request_id=4, length=234),]
            mgr = MemoryManager(size=2048 * 50, dtype=torch.float16,
                                head_dim=23, head_num=23, layer_num=23, device=target)
            batch = InferBatch.init_batch(
                batch_id=0, requests=reqs, dtype=torch.float16, device=target, mem_manager=mgr, vocab_size=233)
            self.assertEqual(2, len(batch.all_input_ids))
            self.assertEqual(233, len(batch.all_input_ids[0]))
            self.assertEqual(234, len(batch.all_input_ids[1]))
            self.assertEqual(0, batch.batch_id)
            self.assertEqual({0: 0, 4: 1}, batch.requests_idx_mapping)
            self.assertEqual(233+234, batch.input_ids.shape[0])
            self.assertEqual([233, 234], batch.input_lengths)
            self.assertEqual(233 + 234, batch.nopad_total_token_num)
            self.assertEqual(234, batch.nopad_max_len_in_batch)
            self.assertEqual([0, 233], list(batch.nopad_b_start_loc))
            self.assertEqual([233, 234], list(batch.nopad_b_seq_len))

    def test_filter(self):
        reqs = [
            TestInferBatch.get_test_request(request_id=x, length=y) for x, y in [
                (2, 232),
                (0, 230),
                (1, 231),
                (3, 233),
                (4, 234),
                (5, 235),
            ]
        ]
        for target in test_targets:
            get_batch = functools.partial(self.get_test_batch, device=target)
            if "all":
                batch = get_batch(reqs)
                self.assertEqual(batch, batch.filter([2, 0, 1, 3, 4, 5]))
                self.assertEqual(batch, batch.filter(["LOL"] * len(batch)))
            if "zero":
                batch = get_batch(reqs)
                with self.assertRaisesRegex(ValueError, "Batch must have at least one request"):
                    batch.filter([])
            if "[2]":
                batch = get_batch(reqs)
                batch = batch.filter([2])
                self.assertEqual(1, len(batch.all_input_ids))
                self.assertEqual(232, len(batch.all_input_ids[0]))
                self.assertEqual({2: 0}, batch.requests_idx_mapping)
                self.assertEqual(232, batch.input_ids.shape[0])
                self.assertEqual([232,], batch.input_lengths)
                self.assertEqual(232, batch.nopad_total_token_num)
                self.assertEqual(232, batch.nopad_max_len_in_batch)
                self.assertEqual([0,], list(batch.nopad_b_start_loc))
                self.assertEqual([232,], list(batch.nopad_b_seq_len))
            if "[5, 0]":
                batch = get_batch(reqs)
                batch = batch.filter([5, 0])
                self.assertEqual(2, len(batch.all_input_ids))
                self.assertEqual(235, len(batch.all_input_ids[0]))
                self.assertEqual(230, len(batch.all_input_ids[1]))
                self.assertEqual({5: 0, 0: 1}, batch.requests_idx_mapping)
                self.assertEqual(230 + 235, batch.input_ids.shape[0])
                self.assertEqual([235, 230,], batch.input_lengths)
                self.assertEqual(235 + 230, batch.nopad_total_token_num)
                self.assertEqual(235, batch.nopad_max_len_in_batch)
                self.assertEqual([0, 235], list(batch.nopad_b_start_loc))
                self.assertEqual([235, 230,], list(batch.nopad_b_seq_len))

    def test_merge(self):
        reqs1 = [
            TestInferBatch.get_test_request(request_id=x, length=y) for x, y in [
                (5, 235),
                (2, 232),
                (0, 230),
            ]
        ]
        reqs2 = [
            TestInferBatch.get_test_request(request_id=x, length=y) for x, y in [
                (3, 233),
                (1, 231),
            ]
        ]
        for target in test_targets:
            get_batch = functools.partial(self.get_test_batch, device=target)
            batch1 = get_batch(reqs1, batch_id=233)
            batch2 = get_batch(reqs2, batch_id=234)
            batch = InferBatch.merge(batch1, batch2)
            self.assertEqual(5, len(batch.all_input_ids))
            self.assertEqual(235, len(batch.all_input_ids[0]))
            self.assertEqual(232, len(batch.all_input_ids[1]))
            self.assertEqual(230, len(batch.all_input_ids[2]))
            self.assertEqual(233, len(batch.all_input_ids[3]))
            self.assertEqual(231, len(batch.all_input_ids[4]))
            self.assertEqual({5: 0, 2: 1, 0: 2, 3: 3, 1: 4},
                             batch.requests_idx_mapping)
            lst = [235, 232, 230, 233, 231]
            self.assertEqual(sum(lst), batch.input_ids.shape[0])
            self.assertEqual(lst, batch.input_lengths)
            self.assertEqual(sum(lst), batch.nopad_total_token_num)
            self.assertEqual(max(lst), batch.nopad_max_len_in_batch)
            self.assertEqual(
                [0] + list(np.array(lst[:-1]).cumsum()), list(batch.nopad_b_start_loc))
            self.assertEqual(lst, list(batch.nopad_b_seq_len))

    @staticmethod
    def get_test_request(*, request_id, length):
        return {
            'request_id': request_id,
            'input_id': np.array([233] * length, dtype=np.int32),
            'sampling_param': {
            },
        }

    @staticmethod
    def get_test_batch(reqs, device, batch_id=0):
        mgr = MemoryManager(size=2048 * 50, dtype=torch.float16,
                            head_dim=23, head_num=23, layer_num=23, device=device)
        batch = InferBatch.init_batch(
            batch_id=batch_id, requests=reqs, dtype=torch.float16, device=device, mem_manager=mgr, vocab_size=233)
        prefill_mem_index = mgr.alloc(batch.nopad_total_token_num)
        init_bloc(batch.nopad_b_loc, batch.nopad_b_seq_len,
                  batch.nopad_max_len_in_batch, prefill_mem_index)
        return batch


if __name__ == '__main__':
    unittest.main()
