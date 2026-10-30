from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from advdetect.datasets.cifar import make_loader
from advdetect.utils.artifacts import save_json
from research.mechanisms.common import StageBackbone, confidence_scores, distance_profile, fit_profile, margin


class ActivationDataset(Dataset):
    def __init__(self, path, shape, labels, logits):
        self.maps = np.memmap(path, mode='r', dtype='float32', shape=shape)
        self.labels, self.logits = labels, logits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.maps[i], copy=True)), self.labels[i], self.logits[i]


@torch.no_grad()
def activation_cache(view, dataset, config, stages, device, directory, name):
    """在独立运行目录流式存储训练激活，避免30轮重复前半网络前向。"""
    streams, shapes = [], []
    labels, logits_all = [], []
    total = 0
    for images, y, _ in make_loader(dataset, config):
        logits, maps = view.forward(images.to(device))
        if not streams:
            for s in stages:
                shape = (len(dataset), *maps[s-1].shape[1:])
                path = directory / f'{name}_stage{s}.float32.bin'
                streams.append(np.memmap(path, mode='w+', dtype='float32', shape=shape))
                shapes.append(shape)
        for stream, s in zip(streams, stages):
            stream[total:total+len(y)] = maps[s-1].cpu().numpy()
        total += len(y)
        labels.append(y); logits_all.append(logits.cpu())
    for stream in streams:
        stream.flush()
    del streams
    labels, logits_all = torch.cat(labels), torch.cat(logits_all)
    return [ActivationDataset(directory / f'{name}_stage{s}.float32.bin', shape, labels, logits_all)
            for s, shape in zip(stages, shapes)]


def matched_drop(h, mean, selected_gate, random_gate, alpha, eps=1e-8):
    delta = h - mean
    selected_energy = (alpha * selected_gate * delta).flatten(1).norm(dim=1)
    random_delta = random_gate * delta
    random_norm = random_delta.flatten(1).norm(dim=1)
    valid = random_norm > eps
    intensity = (selected_energy / (random_norm + eps)).clamp(0, 1)
    # Invalid controls remain in the system and are explicitly diagnosed.
    intensity = torch.where(valid, intensity, torch.zeros_like(intensity))
    h_random = h - intensity[:, None, None, None] * random_delta
    realized = intensity * random_norm
    error = (realized-selected_energy).abs() / selected_energy.clamp_min(eps)
    return h_random, intensity, error, valid


class EvidenceInterventionDetector:
    modes = ('paired', 'keep', 'drop', 'no_control', 'random_pair', 'msp', 'margin')
    def __init__(self, model, config, device, nclasses=10):
        self.model, self.cfg, self.device = model, config, device
        self.nclasses, self.view = nclasses, StageBackbone(model)
        self.stages = config.get('stages', [3])
        self.theta = nn.ParameterList()
        self.means, self.permutations, self.profiles = [], [], {}

    def _branches(self, h, full, classes, stage_index, random=False, all_features=True):
        s = self.stages[stage_index]
        mean = self.means[stage_index].to(h)
        gates = self.theta[stage_index].sigmoid()
        perms = self.permutations[stage_index].to(h.device)
        base = gates[classes]
        if random:
            base = base.gather(1, perms[classes, 0])
        gate = base[:, :, None, None]
        keep_h = mean + gate * (h-mean)
        alpha = float(self.cfg.get('alpha', .5))
        drop_h = h - alpha * gate * (h-mean)
        # No in-place intervention: every branch starts from the original h.
        logits = self.view.resume(torch.cat([keep_h, drop_h], 0), s)
        keep, drop = logits.chunk(2)
        temperature = float(self.cfg.get('temperature', 1))
        u = margin(keep, classes, temperature)
        full_margin = margin(full, classes, temperature)
        v = full_margin - margin(drop, classes, temperature)
        control_effect, errors, valid_controls = [], [], []
        for r in range(int(self.cfg.get('random_controls', 2))):
            offset = r + 1 if random else r
            b = gates[classes].gather(1, perms[classes, offset])[:, :, None, None]
            random_h, _, error, valid = matched_drop(h, mean, gate, b, alpha)
            response_logits = self.view.resume(random_h, s)
            control_effect.append(full_margin-margin(response_logits, classes, temperature))
            errors.append(error); valid_controls.append(valid)
        relative = v-torch.stack(control_effect).mean(0)
        return {'u': u, 'v': v, 'r': relative, 'keep_logits': keep, 'drop_logits': drop,
                'energy_error': torch.stack(errors, 1), 'valid_control': torch.stack(valid_controls, 1)}

    def responses(self, images, classes=None, include_random=True):
        full, maps = self.view.forward(images)
        classes = full.argmax(1) if classes is None else classes
        main, randoms = [], []
        for i, stage in enumerate(self.stages):
            main.append(self._branches(maps[stage-1], full, classes, i))
            if include_random:
                randoms.append(self._branches(maps[stage-1], full, classes, i, random=True))
        values = {
            'paired': torch.stack([v for d in main for v in (d['u'], d['r'])], 1),
            'keep': torch.stack([d['u'] for d in main], 1),
            'drop': torch.stack([d['v'] for d in main], 1),
            'no_control': torch.stack([v for d in main for v in (d['u'], d['v'])], 1),
        }
        if include_random:
            values['random_pair'] = torch.stack([v for d in randoms for v in (d['u'], d['r'])], 1)
        diagnostics = {'energy_error': torch.cat([d['energy_error'] for d in main], 1),
                       'valid_control': torch.cat([d['valid_control'] for d in main], 1),
                       'keep_prediction': main[0]['keep_logits'].argmax(1),
                       'drop_effect': main[0]['v'], 'relative_effect': main[0]['r']}
        return full, classes, values, diagnostics

    def _gate_loss(self, h, y, full, i, kappa):
        gate = self.theta[i].sigmoid()[y, :, None, None]
        mean = self.means[i].to(h)
        keep_h = mean + gate * (h-mean)
        drop_h = h - float(self.cfg.get('alpha', .5))*gate*(h-mean)
        keep, drop = self.view.resume(torch.cat([keep_h, drop_h]), self.stages[i]).chunk(2)
        temperature = float(self.cfg.get('temperature', 1))
        effect = F.relu(margin(drop, y, temperature)-margin(full, y, temperature)+kappa[y])
        budget = (gate.flatten(1).mean(1)-float(self.cfg.get('q', .5))).square().mean()
        loss = F.cross_entropy(keep, y) + float(self.cfg['lambda_drop'])*effect.mean() + float(self.cfg['lambda_budget'])*budget
        if float(self.cfg.get('lambda_nonclass', 0)):
            other = (y + 1) % self.nclasses
            g_other = self.theta[i].sigmoid()[other, :, None, None]
            logits_other = self.view.resume(h-float(self.cfg['alpha'])*g_other*(h-mean), self.stages[i])
            loss = loss + float(self.cfg['lambda_nonclass'])*F.kl_div(
                logits_other.log_softmax(1), full.softmax(1), reduction='batchmean')
        return loss, keep

    def fit_gates(self, gate_data, select_data, full_config, directory):
        train_sets = activation_cache(self.view, gate_data, full_config, self.stages, self.device, directory, 'gate')
        val_sets = activation_cache(self.view, select_data, full_config, self.stages, self.device, directory, 'select')
        generator = torch.Generator().manual_seed(int(full_config['seed']) + 30)
        histories = []
        for i, (train, val) in enumerate(zip(train_sets, val_sets)):
            # Streaming channel mean avoids a full activation-array copy.
            total = torch.zeros(train.maps.shape[1], dtype=torch.float64)
            for start in range(0, len(train), 256):
                total += torch.from_numpy(np.array(train.maps[start:start+256])).double().mean((2, 3)).sum(0)
            self.means.append((total/len(train)).float().to(self.device)[None, :, None, None])
            q = float(self.cfg.get('q', .5))
            self.theta.append(nn.Parameter(torch.full((self.nclasses, train.maps.shape[1]), math.log(q/(1-q)), device=self.device)))
            full_margin = margin(train.logits, train.labels, float(self.cfg.get('temperature', 1)))
            kappa = torch.tensor([.25 * full_margin[(train.labels == c) & (full_margin > 0)].median().item()
                                  if ((train.labels == c) & (full_margin > 0)).any() else 0.
                                  for c in range(self.nclasses)], device=self.device)
            weights = 1 / torch.bincount(train.labels, minlength=self.nclasses).float()[train.labels]
            sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=generator)
            loader = DataLoader(train, batch_size=int(self.cfg['train_batch_size']), sampler=sampler, num_workers=0)
            val_loader = DataLoader(val, batch_size=int(self.cfg['train_batch_size']), num_workers=0)
            optimizer = torch.optim.Adam([self.theta[i]], lr=float(self.cfg['learning_rate']))
            history, best, best_theta = [], None, None
            for epoch in range(int(self.cfg['epochs'])):
                total_loss = total_grad = seen = 0
                for h, y, full in loader:
                    h, y, full = h.to(self.device), y.to(self.device), full.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._gate_loss(h, y, full, i, kappa)
                    loss.backward()
                    grad_norm = self.theta[i].grad.norm().item()
                    optimizer.step()
                    seen += len(y); total_loss += loss.item()*len(y); total_grad += grad_norm
                correct = retained = val_n = 0; val_loss = 0.
                with torch.no_grad():
                    for h, y, full in val_loader:
                        h, y, full = h.to(self.device), y.to(self.device), full.to(self.device)
                        loss, keep = self._gate_loss(h, y, full, i, kappa)
                        good = full.argmax(1).eq(y)
                        correct += good.sum().item(); retained += (keep.argmax(1).eq(y) & good).sum().item()
                        val_n += len(y); val_loss += loss.item()*len(y)
                retention = retained/max(correct, 1)
                row = {'stage': self.stages[i], 'epoch': epoch+1, 'train_loss': total_loss/seen,
                       'select_loss': val_loss/val_n, 'keep_retention_on_correct': retention,
                       'gate_mean': self.theta[i].sigmoid().mean().item(), 'gradient_norm_mean': total_grad/len(loader)}
                # Predeclared: prioritize normal retention >=95%, then validation loss.
                key = (max(0., .95-retention), val_loss/val_n)
                if best is None or key < best:
                    best, best_theta = key, self.theta[i].detach().clone()
                history.append(row)
                print(f'gate {row}', flush=True)
                save_json(directory / f'gate_history_stage{self.stages[i]}.json', history)
                torch.save({'epoch': epoch+1, 'theta': self.theta[i].detach().cpu(), 'optimizer': optimizer.state_dict()},
                           directory / f'gate_progress_stage{self.stages[i]}.pt')
            self.theta[i].data.copy_(best_theta)
            perms = torch.stack([torch.stack([torch.randperm(train.maps.shape[1], generator=generator)
                                              for _ in range(int(self.cfg.get('random_controls', 2))+1)])
                                 for _ in range(self.nclasses)])
            self.permutations.append(perms.to(self.device)); histories.append(history)
        self.theta.requires_grad_(False)
        return {'history': histories, 'gate_means': [p.sigmoid().mean(1).tolist() for p in self.theta],
                'checkpoint_selection': 'dev_select normal retention shortfall, then normal objective'}

    @torch.no_grad()
    def fit_profiles(self, dataset, full_config):
        collected = {k: [] for k in self.modes if k not in ('msp', 'margin')}
        predictions, labels = [], []
        diagnostics = []
        for images, y, _ in make_loader(dataset, full_config):
            _, cp, values, diag = self.responses(images.to(self.device))
            predictions.append(cp.cpu()); labels.append(y)
            for mode in collected:
                collected[mode].append(values[mode].cpu())
            diagnostics.append({k: v.cpu() for k, v in diag.items()})
        cp, labels = torch.cat(predictions), torch.cat(labels)
        self.profiles = {k: fit_profile(torch.cat(v), cp, labels, self.nclasses,
                                       float(self.cfg.get('covariance_beta', .1)), float(self.cfg.get('covariance_ridge', 1e-5)))
                         for k, v in collected.items()}
        diag = {k: torch.cat([d[k] for d in diagnostics]) for k in diagnostics[0]}
        per_class = []
        for c in range(self.nclasses):
            mask = cp.eq(labels) & labels.eq(c)
            retention = diag['keep_prediction'][mask].eq(labels[mask]).float().mean().item() if mask.any() else None
            per_class.append({'class': c, 'correct_count': int(mask.sum()), 'keep_retention': retention,
                              'mean_drop_effect': diag['drop_effect'][mask].mean().item() if mask.any() else None,
                              'mean_relative_effect': diag['relative_effect'][mask].mean().item() if mask.any() else None})
        return {'per_class': per_class, 'fraction_classes_keep_drop_le_5pp': sum(
                    d['keep_retention'] is not None and d['keep_retention'] >= .95 for d in per_class)/self.nclasses,
                'energy_match_le_15pct': (diag['energy_error'] <= .15).float().mean().item(),
                'invalid_control_fraction': (~diag['valid_control']).float().mean().item(),
                'profile_counts': self.profiles['paired']['counts'], 'profile_fallback': self.profiles['paired']['fallback']}

    def scores(self, images, classes=None, modes=None):
        modes = self.modes if modes is None else modes
        if all(m in ('msp', 'margin') for m in modes):
            result = confidence_scores(self.model(images))
        else:
            full, cp, values, _ = self.responses(images, classes, include_random='random_pair' in modes)
            result = confidence_scores(full)
            result.update({k: distance_profile(values[k], cp, self.profiles[k])
                           for k in modes if k not in ('msp', 'margin')})
        return {k: result[k] for k in modes}

    def score(self, images, classes=None, mode='paired'):
        return self.scores(images, classes, [mode])[mode]

    def state(self):
        return {'theta': [p.detach().cpu() for p in self.theta], 'means': [v.cpu() for v in self.means],
                'permutations': [v.cpu() for v in self.permutations], 'profiles': self.profiles}

    def restore(self, state):
        self.theta = nn.ParameterList([nn.Parameter(v.to(self.device), requires_grad=False) for v in state['theta']])
        self.means = [v.to(self.device) for v in state['means']]
        self.permutations = [v.to(self.device) for v in state['permutations']]
        self.profiles = state['profiles']
