import torch
import torch.nn as nn
import torch.optim as optim
import timm
import dataloader as dt
from lr_modules import LayerRegressionMLP, LRAutomator, UniversalActivationSaver


def train():
	# --- Configuration ---
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	model_name = 'resnet50'  # Change to 'resnet50', 'deit_base_patch16_224', etc.
	batch_size = 100
	epochs = 1
	lr = 0.0003
	save_path = f'lr-models/lr_detector_{model_name}.pt'

	print(f">>> Initializing Training for {model_name} on {device}")

	# --- 1. Load Target Model ---
	# We load the model in eval mode because we are not training the backbone
	target_model = timm.create_model(model_name, pretrained=True).to(device)
	target_model.eval()

	# --- 2. Automate Layer Selection ---
	print("\n>>> Running Layer Automation...")
	automator = LRAutomator(target_model)

	# [cite_start]Generate a RANDOM configuration (Algorithm 1 from paper) [cite: 575]
	# This picks 3 layers from the "middle 60%" of the network
	lr_config = automator.generate_config(num_layers=3)

	# --- 3. Initialize Detector ---
	# Target dim: The size of the feature vector a_{n-1} (e.g., 2048 for Inception/ResNet)
	target_dim = target_model.num_features

	detector = LayerRegressionMLP(
		input_dim=lr_config['mlp_input_dim'],
		output_dim=target_dim,
		hidden_dim = 2048  #[cite_start] Per paper [cite: 217]
	).to(device)

	optimizer = optim.Adam(detector.parameters(), lr=lr)
	loss_fn = nn.MSELoss()

	# --- 4. Prepare Data & Saver ---
	if model_name == 'inception_v3':
		resize_dim = 299
	else:
		resize_dim = 224
	train_loader = dt.get_loader(split='train', resize_dim=resize_dim, shuffle=True, batch=batch_size)
	saver = UniversalActivationSaver(target_model, lr_config)

	# --- 5. Training Loop ---
	print("\n>>> Starting Training Loop...")
	detector.train()

	for e in range(epochs):
		total_loss = 0
		total_samples = 0

		for i, (inputs, labels) in enumerate(train_loader):
			inputs = inputs.to(device).float()
			labels = labels.to(device).long()

			optimizer.zero_grad()

			# --- A. Forward Pass (Target Model) ---
			# Use no_grad() to save massive amounts of memory on the backbone
			with torch.no_grad():
				with saver:
					# Get features and logits
					# forward_features -> Raw spatial maps (for some models)
					# forward_head -> Pooled features or logits
					raw_features = target_model.forward_features(inputs)
					logits = target_model.forward_head(raw_features)

					_, preds = torch.max(logits, 1)

					# [cite_start]Filter: Train ONLY on correctly classified samples [cite: 168]
					correct_mask = (preds == labels)
					num_correct = correct_mask.sum().item()

				# If batch has no correct samples, skip
				if num_correct == 0:
					saver.clear()
					continue

				# --- B. Extract Vectors ---
				# 1. Input Vector 'v': Sliced early layers
				# We pass the mask to filter internally in the saver
				v_vector = saver.get_features(mask=correct_mask)

				# 2. Target Vector 'a_{n-1}': Deep feature vector
				# pre_logits=True gives us the feature vector before classification
				target_vector = target_model.forward_head(raw_features, pre_logits=True)
				target_vector = target_vector[correct_mask].detach()

			# --- C. Clear Memory ---
			# Crucial: Delete the heavy feature maps before the backward pass
			saver.clear()

			# --- D. Update Detector ---
			# Forward pass on the lightweight MLP
			mlp_out = detector(v_vector)

			# MSE Loss between predicted features and actual features
			loss = loss_fn(mlp_out, target_vector)
			loss.backward()
			optimizer.step()

			total_loss += loss.item() * num_correct
			total_samples += num_correct

			if i % 10 == 0:
				print(f"Epoch {e} | Batch {i} | Correct: {num_correct} | Loss: {loss.item():.5f}")

	# --- 6. Save Checkpoint ---
	print("\n>>> Saving Checkpoint...")
	checkpoint = {
		'model_state_dict': detector.state_dict(),
		'config': {
			'selected_layers': lr_config['selected_names'],  # CRITICAL: Save names for test.py
			'input_dim': lr_config['mlp_input_dim'],
			'target_dim': target_dim
		}
	}
	torch.save(checkpoint, save_path)
	print(f"Saved to {save_path}")


if __name__ == "__main__":
	train()
