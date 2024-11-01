import torch.distributed as dist

LOCAL_DEVICE_ID = None
LOCAL_DIST_GROUP = None


def set_device_id(device_id):
    global LOCAL_DEVICE_ID
    LOCAL_DEVICE_ID = device_id


def get_device_id():
    return LOCAL_DEVICE_ID


def set_dist_group(dist_group):
    global LOCAL_DIST_GROUP
    LOCAL_DIST_GROUP = dist_group


def local_all_reduce(tensor, op=dist.ReduceOp.SUM):
    dist.all_reduce(tensor, op=op, group=LOCAL_DIST_GROUP, async_op=False)


def local_all_gather(gather_list, tensor):
    dist.all_gather(gather_list, tensor, group=LOCAL_DIST_GROUP, async_op=False)
