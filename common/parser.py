import dataclasses
import operator
import re

import torch
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def _validate_cfg(cfg: OmegaConf) -> None:
	"""Validate resolved experiment settings and the selected CUDA device."""
	for key in ("steps", "batch_size", "horizon", "eval_freq", "eval_episodes"):
		value = cfg.get(key, None)
		if value is None:
			continue
		if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
			raise ValueError(f"Configuration '{key}' must be a positive integer, got {value!r}.")

	device_name = str(cfg.get("device", "cuda:0"))
	try:
		device = torch.device(device_name)
	except (TypeError, RuntimeError) as exc:
		raise ValueError(f"Invalid configured device {device_name!r}.") from exc
	if device.type != "cuda":
		raise ValueError(f"Training and evaluation require a CUDA device, got {device_name!r}.")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available; install a CUDA-enabled PyTorch build.")
	if device.index is not None and device.index >= torch.cuda.device_count():
		raise ValueError(
			f"Configured device {device_name!r} is unavailable; "
			f"found {torch.cuda.device_count()} CUDA device(s)."
		)
	cfg.device = str(device)


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.fullmatch(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					left, operation, right = int(match.group(1)), match.group(2), int(match.group(3))
					if operation == "/" and right == 0:
						raise ValueError(f"Invalid zero divisor in configuration expression {v!r}.")
					operations = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
					result = operations[operation](left, right)
					cfg[k] = int(result) if isinstance(result, float) and result.is_integer() else result
		except:
			pass

	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
		if cfg.task == 'mt30' and cfg.model_size == 19:
			cfg.latent_dim = 512 # This checkpoint is slightly smaller

	# Multi-task
	cfg.multitask = cfg.task in TASK_SET.keys()
	if cfg.multitask:
		cfg.task_title = cfg.task.upper()
		# Account for slight inconsistency in task_dim for the mt30 experiments
		cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
	else:
		cfg.task_dim = 0
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

	_validate_cfg(cfg)
	return cfg_to_dataclass(cfg)
