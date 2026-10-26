# lr_modules.py
import torch
import torch.nn as nn
import random


# --- 1. The Model ---
class LayerRegressionMLP(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim=2048):
		super().__init__()
		# [cite_start]Paper proposes 2 hidden layers [cite: 217]
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, x):
		return self.net(x)


# --- 2. The Automator ---
class LRAutomator:
	"""
	Handles Layer Discovery and Configuration.
	Supports both RANDOM selection (Training) and FIXED restoration (Testing).
	"""

	def __init__(self, model, input_size=None):
		self.model = model
		self.device = next(model.parameters()).device
		self.layer_info = []

		# Auto-detect input size
		if input_size is None:
			if hasattr(model, 'default_cfg') and 'input_size' in model.default_cfg:
				self.input_size = (1, *model.default_cfg['input_size'])
			else:
				self.input_size = (1, 3, 224, 224)
		else:
			self.input_size = input_size

	def _discover_layers(self):
		hooks = []
		self.layer_info = []

		def probe_hook(name):
			def hook(model, input, output):
				# Calculate flattened dimension for slicing logic
				flat_dim = output.flatten(start_dim=1).shape[1]
				self.layer_info.append({
					'name': name,
					'flat_dim': flat_dim
				})

			return hook

		for name, module in self.model.named_modules():
			# Universal check for weighted layers (Conv, Linear, etc.)
			if hasattr(module, 'weight') and len(list(module.children())) == 0:
				hooks.append(module.register_forward_hook(probe_hook(name)))

		# Dummy pass to map model structure
		dummy = torch.zeros(self.input_size).to(self.device)
		with torch.no_grad():
			self.model(dummy)

		for h in hooks: h.remove()

	def generate_config(self, num_layers=3, fixed_layers=None):
		"""
		Args:
			num_layers (int): How many layers to pick (if random).
			fixed_layers (list): List of layer names to force (if restoring).
		"""
		if not self.layer_info:
			self._discover_layers()

		selected_indices = []

		# MODE A: Restore specific layers (Testing/Inference)
		if fixed_layers is not None:
			print(f"Restoring configuration for {len(fixed_layers)} layers...")
			for target_name in fixed_layers:
				found = False
				for idx, info in enumerate(self.layer_info):
					if info['name'] == target_name:
						selected_indices.append(idx)
						found = True
						break
				if not found:
					print(f"WARNING: Layer {target_name} not found in model.")

		# [cite_start]MODE B: Random Selection (Training) [cite: 575]
		else:
			n = len(self.layer_info)
			lower = int(n / 5)
			upper = int(4 * n / 5)
			# Map valid candidates
			candidate_indices = list(range(lower, upper))

			if len(candidate_indices) < num_layers:
				selected_indices = candidate_indices
			else:
				selected_indices = sorted(random.sample(candidate_indices, num_layers))

		# Build the Configuration Dictionary
		config = {
			'selected_names': [],
			'slices': [],
			'mlp_input_dim': 0
		}

		print(f"LR Config: {len(selected_indices)} layers selected.")
		for idx in selected_indices:
			info = self.layer_info[idx]
			name = info['name']
			total_dim = info['flat_dim']

			# [cite_start]Universal Slicing: Middle 60% [cite: 218]
			start = int(total_dim * 0.20)
			end = int(total_dim * 0.80)

			# Closure to lock in start/end values
			slicer = lambda x, s=start, e=end: x.flatten(start_dim=1)[:, s:e]

			config['selected_names'].append(name)
			config['slices'].append(slicer)
			config['mlp_input_dim'] += (end - start)

			print(f" - {name}: Dim {total_dim} -> Slice {start}:{end}")

		return config


# --- 3. The Saver ---
class UniversalActivationSaver:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		self.activations = {}
		self.hooks = []

	def _get_hook(self, name):
		def hook(model, input, output):
			self.activations[name] = output.detach()

		return hook

	def __enter__(self):
		self.activations = {}
		for name, module in self.model.named_modules():
			if name in self.config['selected_names']:
				self.hooks.append(module.register_forward_hook(self._get_hook(name)))
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		for h in self.hooks: h.remove()
		self.hooks = []

	def clear(self):
		"""Releases stored activations to free GPU memory."""
		self.activations = {}

	def get_features(self, mask=None):
		"""
		Returns the concatenated, sliced vector 'v'.
		Args:
			mask (Tensor, optional): Boolean mask to filter samples (e.g. correct only).
									 If None, returns full batch.
		"""
		parts = []
		for name, slicer in zip(self.config['selected_names'], self.config['slices']):
			if name not in self.activations:
				raise RuntimeError(f"Layer {name} captured no data.")

			data = self.activations[name]

			# Apply Mask if provided (Training behavior)
			if mask is not None:
				data = data[mask]

			# Apply Slicing
			sliced = slicer(data)
			parts.append(sliced)

		return torch.cat(parts, dim=1)
