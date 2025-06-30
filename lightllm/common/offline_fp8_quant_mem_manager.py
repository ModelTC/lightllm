import os
import json
import torch
import torch.distributed as dist
from lightllm.utils.envs_utils import get_kv_quant_calibration_inference_count
from lightllm.utils.envs_utils import get_kv_quant_calibration_warmup_count
from lightllm.utils.dist_utils import get_global_rank
from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)

from .mem_manager import MemoryManager


class OfflineFP8QuantMemManager(MemoryManager):
    def __init__(
        self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9, is_export_mode=False
    ):
        # 这里用uint8存储量化后的kv，方便兼容各种torch算子。fp8量化目前采用离线方案，kv_buffer不存储scale
        super().__init__(
            size, dtype if is_export_mode else torch.uint8, head_num, head_dim, layer_num, always_copy, mem_fraction
        )

        self.qmin = torch.finfo(torch.float8_e4m3fn).min
        self.qmax = torch.finfo(torch.float8_e4m3fn).max
        self.model_arch = get_model_architectures(get_env_start_args().model_dir)
        self.layer_num = layer_num
        self.head_num = head_num
        self.total_head_num = head_num * dist.get_world_size() if dist.is_initialized() else head_num
        self.scales_shape = [layer_num, 2 * head_num]
        self.scales = None
        self.scales_list = []
        self.abs_max = None
        self.warmup_counts = get_kv_quant_calibration_warmup_count()
        self.inference_counts = get_kv_quant_calibration_inference_count()
        self.count = 0
        self.enable_calib = False
        self.is_export_mode = is_export_mode
        if is_export_mode:
            self.abs_max = torch.zeros(self.scales_shape, dtype=torch.float32, device="cuda")
        elif get_env_start_args().kv_quant_calibration_config_path is not None:
            logger.info(
                f"kv_quant_calibration_config_path {get_env_start_args().kv_quant_calibration_config_path} is set, "
                "will load kv quant calibration config"
            )
            if os.path.exists(get_env_start_args().kv_quant_calibration_config_path):
                with open(get_env_start_args().kv_quant_calibration_config_path, "r") as f:
                    cfg = json.load(f)

                if cfg["architectures"] != self.model_arch:
                    raise ValueError(
                        f"architectures {cfg['architectures']} in config "
                        f"not match current model_arch {self.model_arch}"
                    )
                if cfg["num_layers"] != layer_num:
                    raise ValueError(
                        f"num_layers {cfg['num_layers']} in config " f"not match current layer_num {layer_num}"
                    )
                if cfg["num_head"] != self.total_head_num:
                    raise ValueError(
                        f"num_head {cfg['num_head']} in config "
                        f"not match current model head num {self.total_head_num}"
                    )
                if get_env_start_args().enable_fa3:
                    if cfg["quant_type"] != "per_head":
                        raise ValueError(f"quant type {cfg['num_head']} in config not match fa3 backend")
                else:
                    raise ValueError("only support per_head quant type for fa3 backend, use --enable_fa3 in start args")

                self.qmin = cfg["qmin"]
                self.qmax = cfg["qmax"]
                self.scales_shape = cfg["scales_shape"]

                full_scales_list = cfg["scales"]
                self.scales_list = full_scales_list
                self.scales = torch.tensor(self.scales_list, dtype=torch.float32, device="cuda").view(self.scales_shape)
                if dist.is_initialized() and dist.get_world_size() > 1:
                    half_head = self.total_head_num // 2
                    start_head = dist.get_rank() * head_num
                    end_head = start_head + head_num
                    k_scales = self.scales[:, start_head:end_head].contiguous()
                    v_scales = self.scales[:, start_head + half_head : end_head + half_head].contiguous()
                    current_scales = torch.cat((k_scales, v_scales), dim=-1)

                    self.scales_list = current_scales.tolist()
                    self.scales = current_scales
            else:
                raise FileNotFoundError(
                    f"kv_quant_calibration_config {get_env_start_args().kv_quant_calibration_config_path} not found"
                )
        else:
            logger.warning("scales is None, no kv_quant_calibration_config_path be set")

    def enable_calibration(self):
        assert (
            get_env_start_args().enable_fa3
        ), "Calibration is only supported in fa3 backend, use --enable_fa3 in start args"
        assert self.is_export_mode, "Calibration is only supported in export mode"
        assert get_env_start_args().disable_cudagraph, "Calibration is not supported in cudagraph mode"
        logger.info("Enable kv cache calibration, will collect kv cache data for quantization calibration")
        self.enable_calib = True

    def update_calibration_data(self, kv_buffer: torch.Tensor, layer_index: int):
        if not self.enable_calib or self.count >= self.warmup_counts + self.inference_counts:
            return

        if self.abs_max is not None and self.count >= self.warmup_counts:
            kv_max = kv_buffer.abs().amax(dim=(0, 2)).to(torch.float32)
            self.abs_max[layer_index] = torch.maximum(self.abs_max[layer_index], kv_max)
            if self.count == self.warmup_counts + self.inference_counts - 1 and layer_index == self.layer_num - 1:
                final_abs_max = self.abs_max
                if dist.is_initialized() and dist.get_world_size() > 1:
                    k_max, v_max = torch.chunk(self.abs_max, 2, dim=-1)
                    k_max = k_max.contiguous()
                    v_max = v_max.contiguous()
                    gathered_k_max = [torch.zeros_like(k_max) for _ in range(dist.get_world_size())]
                    gathered_v_max = [torch.zeros_like(v_max) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_k_max, k_max, group=None, async_op=False)
                    dist.all_gather(gathered_v_max, v_max, group=None, async_op=False)
                    k_max = torch.cat(gathered_k_max, dim=-1)
                    v_max = torch.cat(gathered_v_max, dim=-1)
                    final_abs_max = torch.cat((k_max, v_max), dim=-1)

                self.scales = final_abs_max / self.qmax
                self.scales = torch.where(self.scales > 0, self.scales, torch.ones_like(self.scales))

                if get_global_rank() == 0:
                    self.abs_max = final_abs_max
                    self._export_calibration_data()

        if layer_index == self.layer_num - 1:
            self.count += 1

    def _export_calibration_data(self):
        cfg = {
            "version": "1.0",
            "architectures": self.model_arch,
            "quant_type": "per_head",
            "qmin": self.qmin,
            "qmax": self.qmax,
            "num_layers": self.layer_num,
            "num_head": self.total_head_num,
            "scales_shape": list(self.abs_max.shape),
            "scales": self.scales.cpu().numpy().tolist(),
        }
        with open("./kv_cache_calib.json", "w") as f:
            json.dump(cfg, f, indent=4)
        logger.info(
            f"Export kv cache calibration data to kv_cache_calib.json, "
            f"architectures: {self.model_arch}, "
            f"qmin: {self.qmin}, qmax: {self.qmax}, "
            f"total heads: {self.total_head_num}, "
            f"scales_shape: {list(self.abs_max.shape)}, "
        )
