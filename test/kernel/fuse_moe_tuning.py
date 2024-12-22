import os
import torch
import torch.multiprocessing as mp
from lightllm.common.fused_moe.better_fused_moe import fused_experts_impl
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@torch.no_grad()
def test_kernel(expert_num: int, m: int, n: int, k: int, topk: int, dtype: torch.dtype, test_count: int = 20, **config):
    input_tuples = []
    for _ in range(test_count):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((expert_num, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((expert_num, k, n), device="cuda", dtype=dtype) / 10

        rnd_logics = torch.randn(m, expert_num, device="cuda")
        topk_values, topk_ids = torch.topk(rnd_logics, topk, dim=1)

        topk_weights = torch.randn((m, topk), device="cuda", dtype=dtype) / 10
        input_tuples.append((a, w1, w2, topk_ids, topk_weights))

    fused_experts_impl(a, w1, w2, topk_weights, topk_ids, inplace=True, **config)

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for index in range(test_count):
            a, w1, w2, topk_ids, topk_weights = input_tuples[index]
            fused_experts_impl(a, w1, w2, topk_weights, topk_ids, inplace=True, **config)

    # graph.replay()
    import time

    torch.cuda.synchronize()
    start = time.time()
    graph.replay()
    torch.cuda.synchronize()

    cost_time = (time.time() - start) * 1000

    logger.info(str(config))
    logger.info(f"bf16 {m} cost time: {cost_time} ms")
    return cost_time


def worker(
    expert_num: int,
    m: int,
    n: int,
    k: int,
    topk: int,
    dtype: torch.dtype,
    test_count: int,
    test_configs,
    queue,
):
    try:
        for index in range(len(test_configs)):
            cost_time = test_kernel(
                expert_num=expert_num,
                m=m,
                n=n,
                k=k,
                topk=topk,
                dtype=dtype,
                test_count=test_count,
                **test_configs[index],
            )
            queue.put(cost_time)  # Put result in queue
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as ex:
        logger.error(str(ex))
        logger.exception(str(ex))
        import sys

        sys.exit(-1)
        pass


def get_test_configs():
    # all_configs = []
    for num_stages in [
        1,
        2,
        3,
    ]:
        for GROUP_SIZE_M in [
            1,
            2,
            4,
        ]:
            for num_warps in [4, 8, 16]:
                for BLOCK_SIZE_M in [
                    16,
                    32,
                    64,
                ]:
                    for BLOCK_SIZE_N in [16, 32, 64, 128, 256]:
                        for BLOCK_SIZE_K in [16, 32, 64, 128, 256]:
                            t_config = {
                                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                "GROUP_SIZE_M": GROUP_SIZE_M,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                            yield t_config
                            # all_configs.append(t_config)

    # import random
    # random.shuffle(all_configs)
    # for t_config in all_configs:
    #     yield t_config


def tuning_configs(
    expert_num: int,
    m: int,
    n: int,
    k: int,
    topk: int,
    dtype: torch.dtype,
    test_count: int = 20,
):

    best_config, best_cost_time = None, 10000000
    queue = mp.Queue()
    test_configs = []
    for t_config in get_test_configs():
        test_configs.append(t_config)
        if len(test_configs) < 256:
            continue

        p = mp.Process(
            target=worker,
            args=(
                expert_num,
                m,
                n,
                k,
                topk,
                dtype,
                test_count,
                test_configs,
                queue,
            ),
        )
        p.start()
        p.join()
        while len(test_configs) != 0:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs[0]
                    best_cost_time = cost_time
                    logger.info(f"cur best : {best_config} {best_cost_time}")
                del test_configs[0:1]
            except:
                del test_configs[0:16]
                logger.info(f"cur best : {best_config} {best_cost_time}")
                break

    while len(test_configs) != 0:
        p = mp.Process(
            target=worker,
            args=(
                expert_num,
                m,
                n,
                k,
                topk,
                dtype,
                test_count,
                test_configs,
                queue,
            ),
        )
        p.start()
        p.join()

        try:
            cost_time = queue.get_nowait()
            logger.info(f"get {test_configs[0]} cost_time: {cost_time}")
            if cost_time < best_cost_time:
                best_config = test_configs[0]
                best_cost_time = cost_time
                logger.info(f"cur best : {best_config} {best_cost_time}")
            del test_configs[0:1]
        except:
            del test_configs[0:16]
            logger.info(f"cur best : {best_config} {best_cost_time}")
            break

    logger.info(f"{best_config} best cost: {best_cost_time}")


if __name__ == "__main__":
    tuning_configs(
        expert_num=64,
        m=200,
        n=1408 // 2,
        k=2048,
        topk=6,
        dtype=torch.bfloat16,
        test_count=8,
    )
    pass
