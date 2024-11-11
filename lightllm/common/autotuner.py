from __future__ import annotations

from functools import wraps
import builtins
import os
import time
import inspect
from typing import Dict
from tqdm import tqdm

from triton.testing import do_bench, do_bench_cudagraph
from triton.runtime.jit import KernelInterface
from triton.runtime.errors import OutOfResources
import triton

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def closest_power_of_2(n):
    n = int(n)
    # 对于小于等于 1 的情况，直接返回 1
    if n <= 1:
        return 1
    # 使用位运算查找最接近的 2 的幂
    lower = 1 << (n.bit_length() - 1)
    upper = lower << 1
    return lower if (n - lower) < (upper - n) else upper


def get_str(name_list, value_list):
    return ",".join([f"{name}={value}" for (name, value) in zip(name_list, value_list)])


class Autotuner(KernelInterface):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=25,
        rep=100,
        use_cuda_graph=False,
    ):
        if not configs:
            self.configs = [triton.Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = {}
        self.arg_names = arg_names

        # Reset to zero or restore values
        self.reset_idx = []
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]
        self.restore_idx = []
        if restore_value is not None:
            self.restore_idx = [arg_names.index(k) for k in restore_value]

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda args, reset_only=False: 0
        self.post_hook = lambda args, exception: 0
        if pre_hook:
            self.pre_hook = pre_hook
        elif len(self.reset_idx) > 0 or len(self.restore_idx) > 0:

            def _pre_hook(args, reset_only=False):
                for i in self.reset_idx:
                    args[i].zero_()
                if not reset_only:
                    self.restore_copies = [args[i].clone() for i in self.restore_idx]

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
        elif len(self.restore_idx) > 0:

            def _post_hook(args, exception):
                for i, j in enumerate(self.restore_idx):
                    args[j].copy_(self.restore_copies[i])
                self.restore_copies = []

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)

        self.fn = fn
        self.fn_name = f"{os.path.relpath(fn.__module__)}.{fn.__name__}"
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.num_warmups = warmup
        self.num_reps = rep
        import torch

        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()

    def _bench(self, *args, config, **meta):
        from triton.compiler.errors import CompileTimeAssertionFailure

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.all_kwargs().keys()
        if conflicts:
            # raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
            #                  " Make sure that you don't re-define auto-tuned symbols.")
            meta = {k: v for k, v in meta.items() if k not in conflicts}

        conflicts = meta.keys() & config.all_kwargs().keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )

        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(args, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(args, exception=None)

        try:
            if self.use_cuda_graph:
                import torch

                with torch.cuda.stream(torch.cuda.Stream()):
                    bench_res = do_bench_cudagraph(kernel_call, rep=self.num_reps, return_mode="median")
                return bench_res
            return do_bench(kernel_call, warmup=self.num_warmups, rep=self.num_reps, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure):
            return float("inf") if self.use_cuda_graph else [float("inf"), float("inf"), float("inf")]

    def run(self, *args, **kwargs):
        if os.environ.get("ENABLE_AUTOTUNE", "0") == "0":
            return self.fn.run(*args, **kwargs)

        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            _args_name = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
                    _args_name.append(name)
            key_list = [_args[i] for i in self.key_idx]
            key = tuple(key_list)
            if key not in self.cache:
                _args_name = []
                for name in self.arg_names:
                    if name in all_args:
                        _args_name.append(name)
                name_list = [_args_name[i] for i in self.key_idx]
                used_cached_result = False
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in tqdm(self.configs, desc=f"Tuning {self.fn_name}::{get_str(name_list, key_list)}")
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config

        conflicts = kwargs.keys() & config.all_kwargs().keys()
        kwargs = {k: v for k, v in kwargs.items() if k not in conflicts}

        if not used_cached_result:
            logger.debug(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})

        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )

        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(
                self.fn.warmup(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                )
            )
        self.nargs = None
        return ret


def autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=True,
):
    def autotuned(fn):
        return Autotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
        )

    return autotuned
