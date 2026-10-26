import torch
import torch.nn as nn
import timm
import dataloader as dt
import utils
from lr_modules import LayerRegressionMLP, LRAutomator, UniversalActivationSaver


def test():
	# --- Configuration ---
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	model_name = 'resnet50'
	# Ensure this matches the filename you saved in train.py
	load_path = f'lr-models/lr_detector_{model_name}.pt'

	print(f">>> Initializing Testing for {model_name} on {device}")

	# --- 1. Load Target Model ---
	target_model = timm.create_model(model_name, pretrained=True).to(device)
	target_model.eval()

	# --- 2. Load Checkpoint & Restore Config ---
	print(f">>> Loading Checkpoint from {load_path}...")
	try:
		checkpoint = torch.load(load_path)
	except FileNotFoundError:
		print("Error: Checkpoint not found. Run train.py first.")
		return

	saved_config = checkpoint['config']
	saved_layers = saved_config['selected_layers']
	print(f"    Restoring layers: {saved_layers}")

	# Initialize Automator and Force Fixed Layers
	automator = LRAutomator(target_model)
	# [cite_start]This ensures exact match with training config [cite: 575]
	lr_config = automator.generate_config(fixed_layers=saved_layers)

	# --- 3. Initialize Detector ---
	detector = LayerRegressionMLP(
		input_dim=saved_config['input_dim'],
		output_dim=saved_config['target_dim'],
		hidden_dim=2048
	).to(device)

	detector.load_state_dict(checkpoint['model_state_dict'])
	detector.eval()

	# --- 4. Prepare Data & Saver ---
	loss_fn = nn.MSELoss()
	saver = UniversalActivationSaver(target_model, lr_config)

	# Test Loader (Batch size 1 is fine for detailed analysis)
	if model_name == 'inception_v3':
		resize_dim = 299
	else:
		resize_dim = 224
	test_loader = dt.get_loader(split='test', resize_dim=resize_dim, shuffle=False, batch=1)

	# Specific indices to test
	img_indexes = [0, 500, 1500, 7500, 9500]

	print("\n>>> Starting Evaluation...")
	print(f"{'Index':<6} | {'Status':<10} | {'Clean MSE':<10} | {'Adv MSE':<10} | {'Detection'}")
	print("-" * 65)

	for idx in img_indexes:
		img, lbl = test_loader.dataset[idx]
		img = torch.unsqueeze(img.to(torch.float32).to(device), dim=0)
		lbl = torch.tensor([lbl], device=device, dtype=torch.long)

		# --- A. Clean Pass ---
		with torch.no_grad():
			with saver:
				# 1. Get raw spatial features FIRST (e.g., [1, 2048, 7, 7])
				raw_spatial_clean = target_model.forward_features(img)

				# 2. Get Pooled Features (a_n-1) for your Detector (pre_logits=True)
				# Output shape: [1, 2048]
				feat_clean = target_model.forward_head(raw_spatial_clean, pre_logits=True)

				# 3. Get Logits for Accuracy Check (default behavior)
				# We pass 'raw_spatial_clean' here, NOT 'feat_clean'
				logits = target_model.forward_head(raw_spatial_clean)
				_, pre = torch.max(logits, 1)

				# Get 'v' vector
				v_clean = saver.get_features(mask=None)
			saver.clear()

		# Detector Prediction
		pred_clean = detector(v_clean)
		# [cite_start]Compare prediction against actual features [cite: 126, 168]
		mse_clean = loss_fn(pred_clean, feat_clean).item()

		# --- B. Adversarial Pass ---
		# Generate attack (e.g., PGD)
		adv_img = utils.get_attack(img, lbl, target_model, 'pgd')

		with torch.no_grad():
			with saver:
				# 1. Get raw spatial features
				raw_spatial_adv = target_model.forward_features(adv_img)

				# 2. Get Pooled Features for Detector
				feat_adv = target_model.forward_head(raw_spatial_adv, pre_logits=True)

				# 3. Get Logits for Accuracy Check
				# FIX: Pass raw_spatial_adv, not feat_adv
				logits_adv = target_model.forward_head(raw_spatial_adv)

				_, pred_label = torch.max(logits_adv, 1)
				attack_success = (pred_label != lbl).item()

				v_adv = saver.get_features(mask=None)
			saver.clear()

		# Detector Prediction
		pred_adv = detector(v_adv)
		# [cite_start]Expected: High MSE for adversarial samples [cite: 121]
		mse_adv = loss_fn(pred_adv, feat_adv).item()

		# --- C. Result ---
		status = "Success" if attack_success else "Failed"
		# Simple threshold logic: if Error increases significantly, it's an attack
		# In practice, you'd calculate a threshold over the validation set.
		is_detected = mse_adv > (mse_clean * 1.5)
		det_str = "DETECTED" if is_detected else "Missed"

		print(f"{idx:<6} | {status:<10} | {mse_clean:.5f}    | {mse_adv:.5f}    | {det_str}")


if __name__ == "__main__":
	test()
