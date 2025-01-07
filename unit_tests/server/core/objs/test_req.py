import pytest

from lightllm.server.core.objs.req import Req, NormalReq, TokenHealingReq, SplitFuseReq, SamplingParams


@pytest.fixture
def req():
    req_instance = Req()
    req_instance.init(1, [1, 2, 3], {"max_new_tokens": 1}, None, splitfuse_block_size=128)
    return req_instance


def test_req_init(req):
    assert req.request_id == 1
    assert req.input_len == 3


def test_set_status(req):
    req.req_status.set_status(req.req_status.RUNNING)
    assert req.req_status.is_running()


def test_create_prompt_ids_shm_array(req):
    assert hasattr(req, "shm_prompt_ids")


def test_get_used_tokens(req):
    req.cur_kv_len = 5
    assert req.get_used_tokens() == 5


def test_normal_req_get_tuple_tokens():
    normal_req = NormalReq()
    normal_req.init(1, [1, 2, 3], {"max_new_tokens": 2}, None)
    normal_req.req_status.set_status(normal_req.req_status.RUNNING)
    result = normal_req.get_tuple_tokens(False, 10)
    assert result == (3, 1)  # Adjust expected result based on logic


def test_token_healing_req_post_init():
    token_healing_req = TokenHealingReq()
    token_healing_req.init(1, [1, 2, 3, 4], {"max_new_tokens": 1}, None)
    assert token_healing_req.sample_params.max_new_tokens == 9


def test_split_fuse_req_get_tuple_tokens():
    split_fuse_req = SplitFuseReq()
    split_fuse_req.init(1, [1, 2, 3], {"max_new_tokens": 1}, None, splitfuse_block_size=256)
    split_fuse_req.req_status.set_status(split_fuse_req.req_status.RUNNING)
    result = split_fuse_req.get_tuple_tokens(False, 10)
    assert isinstance(result, tuple)


def test_finish_status(req):
    req.finish_status.set_status(req.finish_status.FINISHED_ABORT)
    assert req.finish_status.is_aborted()


if __name__ == "__main__":
    pytest.main()
