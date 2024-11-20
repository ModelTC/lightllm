import os
from typing import Any, Literal, Optional
import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class LocalProfiler:
    def __init__(self, mode: Literal["torch_profile", "nvtx"], name: Optional[str] = None):
        self.mode: Literal["torch_profile", "nvtx"] = mode
        self.name: Optional[str] = name
        self.active: bool = False
        if self.mode == "torch_profile":
            trace_dir = os.getenv("LIGHTLLM_TRACE_DIR", "./trace")
            self._torch_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,  # additional overhead
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, worker_name=name, use_gzip=True),
            )
            logger.warning(
                "Profiler support (--profiler=XXX) for torch.profile enabled, trace file will been saved to %s",
                trace_dir,
            )
            logger.warning("do not enable this feature in production")
        elif self.mode == "nvtx":
            self._nvtx_toplevel_mark = "LIGHTLLM_PROFILE"
            self._nvtx_toplevel_id = None
            logger.warning(
                """Profiler support (--profiler=XXX) for NVTX enabled, toplevel NVTX mark is %s,
                use it with external profiling tools""",
                self._nvtx_toplevel_mark,
            )
            logger.warning(
                """e.g. nsys profile --capture-range=nvtx --nvtx-capture=%s --trace=cuda,nvtx
                -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 [--other_nsys_options]
                python -m lightllm.server.api_server --profiler=nvtx [--other_lightllm_options]""",
                self._nvtx_toplevel_mark,
            )
        elif self.mode is not None:
            assert False, "invalid profiler mode"

    def start(self):
        if self.active:
            logger.error("profiler already started, ignore")
            return
        logger.warning("Profiler support: profiling start")
        self.active = True
        if self.mode == "torch_profile":
            self._torch_profiler.start()
        elif self.mode == "nvtx":
            self._nvtx_toplevel_id = torch.cuda.nvtx.range_start(self._nvtx_toplevel_mark)

    def stop(self):
        if not self.active:
            logger.error("profiler not started, ignore")
            return
        logger.warning("Profiler support: profiling stop")
        self.active = False
        if self.mode == "torch_profile":
            logger.warning("Profiler support: torch_profiler saving trace file, it might take a while...")
            self._torch_profiler.stop()
            logger.warning("Profiler support: torch_profiler saving done")
        elif self.mode == "nvtx":
            torch.cuda.nvtx.range_end(self._nvtx_toplevel_id)

    def mark_range_start(self, message: str) -> Any:
        "return the handle of the range, to be used in mark_range_end()"
        if self.active:
            # only support for NVTX mode
            if self.mode == "nvtx":
                return torch.cuda.nvtx.range_start(message)

    def mark_range_end(self, handle: Any):
        if self.active:
            # only support for NVTX mode
            if self.mode == "nvtx":
                return torch.cuda.nvtx.range_end(handle)
