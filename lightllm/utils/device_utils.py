from functools import lru_cache


@lru_cache(maxsize=None)
def get_device_sm_count():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["multiprocessor_count"]


@lru_cache(maxsize=None)
def get_device_sm_regs_num():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["max_num_regs"]


@lru_cache(maxsize=None)
def get_device_sm_shared_mem_num():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["max_shared_mem"]


@lru_cache(maxsize=None)
def get_device_warp_size():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["warpSize"]
