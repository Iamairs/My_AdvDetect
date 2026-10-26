from attacks import *

def get_attack(inputs, labels, model, attack_type, eps=8/255):
	adv_inputs = None
	if attack_type == 'pgd':
		atk = PGD(model, eps=eps, alpha=1 / 255, steps=10, random_start=True)
		adv_inputs = atk(inputs, labels)
	if attack_type == 'bim':
		atk = BIM(model, eps=eps, alpha=2 / 255, steps=10)
		adv_inputs = atk(inputs, labels)
	if attack_type == 'pif':
		atk = PIFGSM(model, num_iter_set=10)
		adv_inputs = atk(inputs, labels)
	if attack_type == 'vmi':
		atk = VMIFGSM(model, eps=eps, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
		adv_inputs = atk(inputs, labels)
	if attack_type == 'vni':
		atk = VNIFGSM(model, eps=eps, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2)
		adv_inputs = atk(inputs, labels)
	if attack_type == 'apgd':
		atk = APGD(model, norm='Linf', eps=eps, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
		adv_inputs = atk(inputs, labels)
	if attack_type == 'anda':
			adv_inputs = anda_attack.attack(inputs, labels, model)

	return adv_inputs

