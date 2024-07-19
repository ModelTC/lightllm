import torch
import torch.distributed as dist
from .radix_cache import RadixCache, TreeNode
from .shared_arr import SharedInt
from lightllm.common.mem_manager import MemoryManager
import threading
import queue

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FakeTreeNode:
    """
    主要用于在Sorted Tree中排序搜索
    """

    def __init__(self, ref_count, child_num, time_id):
        self.data = (ref_count, child_num, time_id)

    def get_compare_key(self):
        return self.data


class SwapManager(threading.Thread):
    def __init__(self, gpu_radix_cache: RadixCache, cpu_radix_cache: RadixCache, unique_name: str) -> None:
        super().__init__()
        self.gpu_radix_cache = gpu_radix_cache
        self.cpu_radix_cache = cpu_radix_cache
        # 创建进程间通信组
        self.dist_group = dist.new_group(backend="gloo")
        # 最新更新radix tree 中的叶节点的时间id, 只会更新 radix_cache 中时间id > latest_move_time_id 的节点
        # 减少判断的时间量
        self.latest_move_time_id = 0

        # 每次最大的移动数据的长度
        self.max_move_token_num = 512

        # 多进程共享的全局共享变量，多个共享进程使用一个标识移动任务的开始和结束, 0 为结束， 1 为开始
        self.shared_task_mark = SharedInt(f"{unique_name}_swap_task_mark")
        self.shared_task_mark.set_value(0)

        self.cuda_stream = torch.cuda.Stream()

        self.task_start_event = queue.Queue(maxsize=1)
        self.task_end_event = queue.Queue(maxsize=1)
        self.daemon = True
        self.start()
        return

    def move_from_gpu_to_cpu(self):
        for leaf_tree_node in self.gpu_radix_cache.evict_tree_set.irange(
            minimum=FakeTreeNode(0, 0, self.latest_move_time_id), maximum=None, inclusive=(True, True)
        ):
            if not leaf_tree_node.is_leaf():
                break
            # 如果遇到根节点，没有数据可以复制直接
            if leaf_tree_node == self.gpu_radix_cache.root_node:
                continue
            dist.barrier(group=self.dist_group)
            # 插入检查退出点的代码。
            if self._check_move_task_finished():
                break
            self.latest_move_time_id = leaf_tree_node.time_id
            leaf_tree_node: TreeNode = leaf_tree_node
            all_key, all_value = leaf_tree_node.get_all_key_value_buffers()
            _, prefix_len, _ = self.cpu_radix_cache.match_prefix(all_key, update_refs=False)
            move_key = all_key[prefix_len:]
            move_value = all_value[prefix_len:]

            success_move_token_size = 0
            # 初始化的prefix_len 的长度只是占位，在插入到cpu radix cache的过程中不会真正使用
            cpu_values = [torch.empty((prefix_len,), dtype=torch.int64, device="cpu")]
            total_prefix_len = prefix_len

            while len(move_key) != 0:
                dist.barrier(group=self.dist_group)
                # 分块进行 move 操作
                cur_move_key = move_key[0 : self.max_move_token_num]
                cur_move_value = move_value[0 : self.max_move_token_num]
                is_success, move_token_size, cpu_value_index = self._gpu_to_cpu_copy(cur_move_key, cur_move_value)
                assert move_token_size <= self.max_move_token_num
                total_prefix_len += move_token_size
                success_move_token_size += move_token_size
                if move_token_size > 0:
                    cpu_values.append(cpu_value_index)
                move_key = move_key[move_token_size:]
                move_value = move_value[move_token_size:]
                dist.barrier(group=self.dist_group)

                if (not is_success) or (self._check_move_task_finished()):  # 插入检查退出点的代码。
                    break

            # 将移动成功的数据插入到cpu的radix cache中
            if success_move_token_size != 0:
                self.cpu_radix_cache.insert(all_key[0:total_prefix_len], torch.cat(cpu_values)[0:total_prefix_len])
                logger.info(f"from gpu to cpu token_num: {success_move_token_size}")

            dist.barrier(group=self.dist_group)
            # 插入检查退出点的代码。
            if self._check_move_task_finished():
                break

    def _gpu_to_cpu_copy(self, cur_move_key, cur_move_value):
        """
        返回False表示移动失败。反之即成功,
        第二个返回值代表移动的token数量
        """
        with torch.cuda.StreamContext(self.cuda_stream):
            assert len(cur_move_key) == len(cur_move_value)
            move_token_size = len(cur_move_key)
            if move_token_size == 0:
                return False, 0, None
            if (
                self.cpu_radix_cache.mem_manager.can_use_mem_size + self.cpu_radix_cache.can_released_token_num()
                < move_token_size
            ):
                return False, 0, None
            # 准备移动
            self.cpu_radix_cache.free_radix_cache_to_get_enough_token(move_token_size)
            dest_mem_index = self.cpu_radix_cache.mem_manager.alloc(move_token_size)
            self.gpu_radix_cache.mem_manager.copy_to_mem_manager(
                cur_move_value, self.cpu_radix_cache.mem_manager, dest_mem_index
            )
            self.cuda_stream.synchronize()  # 等待复制完成
            return True, move_token_size, dest_mem_index.detach().cpu()

    def cpu_to_gpu_copy(self, cpu_mem_index, gpu_mem_index):
        self.cpu_radix_cache.mem_manager.copy_to_mem_manager(
            cpu_mem_index, self.gpu_radix_cache.mem_manager, gpu_mem_index
        )
        return

    def mark_swap_task_start(self):
        dist.barrier(group=self.dist_group)
        self.shared_task_mark.set_value(1)
        self.task_start_event.put(None)
        return

    def wait_swap_task_finished(self):
        self.shared_task_mark.set_value(0)
        self.task_end_event.get()
        return

    def _check_move_task_finished(self):
        """
        检查点函数，用于检查是否外部调用了 mark_swap_task_finished 想让任务结束
        """
        return self.shared_task_mark.get_value() == 0

    def run(self):
        # 新线程需要重新设置当强设备
        import torch.distributed as dist

        tp_rank = dist.get_rank()
        torch.cuda.set_device(tp_rank)
        while True:
            self.task_start_event.get()
            self.move_from_gpu_to_cpu()
            self.task_end_event.put(None)
