"""只读取现有 P1 产物，生成中文结果表与可导出机制图。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path.insert(0, str(ROOT/'src'))
from advdetect.utils.config import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    directory = resolve_path(args.run_dir)
    metrics = json.loads((directory/'ordinary_metrics.json').read_text(encoding='utf-8'))
    refinement = None
    if (directory/'baseline_refinement.json').is_file():
        refinement = json.loads((directory/'baseline_refinement.json').read_text(encoding='utf-8'))
        metrics['shared'].update(refinement['metrics'])
    lines = ['# 两个方案 P0/P1 探索性验证', '',
             '固定 CIFAR-10/CIFAR ResNet18，1,000 张正常开发图及其同一批标准 PGD 攻击。', '',
             '**限制**：原分类器训练索引未知；阈值来自独立开发子集的全部正常图；未使用最终校准集或官方测试集。不能解释为论文最终结果。', '',
             '| 方法 | 模式 | AUROC（全正常/成功攻击） | 正常FPR | 成功攻击TPR | JASR0（该普通PGD候选） |',
             '|---|---|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for name, modes in metrics.items():
        for mode, values in modes.items():
            if not isinstance(values, dict):
                continue
            keys = ['auroc_successful_only', 'clean_fpr', 'tpr_on_successful_attacks', 'joint_success_rate_on_clean_correct']
            lines.append(f'| {name} | {mode} | '+ ' | '.join(fmt(values[k]) for k in keys)+' |')
    diag_path = directory/'p0_shared_convergence_1000.json'
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding='utf-8'))
        lines += ['', '## 方案一数值诊断', '', '| 迭代数 | 目标均值 | 最大相对FW间隙 | 未达到1e-5的比例 |', '|---|---:|---:|---:|']
        for steps, values in diag.items():
            if steps.isdigit():
                lines.append(f'| {steps} | {values["objective_mean"]:.6g} | {values["relative_fw_gap_max"]:.6g} | {values["not_converged_fraction"]:.3%} |')
        lines += ['', f'共享减独立目标：`{diag["shared_minus_independent"]}`。原始负值没有被截断。']
    adaptive_path = directory/'adaptive_probe.json'
    if adaptive_path.is_file():
        probe = json.loads(adaptive_path.read_text(encoding='utf-8'))
        lines += ['', '## 联合攻击实现探针', '',
                  '参数：`'+json.dumps(probe['configuration'], ensure_ascii=False)+'`。所有模式评价同一候选流并集；此实现探针不足以支持鲁棒性结论。', '',
                  '| 方法 | 模式 | 原模型正确样本数 | 并集JASR0 |', '|---|---|---:|---:|']
        for name, modes in probe['results'].items():
            for mode, values in modes.items():
                lines.append(f'| {name} | {mode} | {values["num_clean_correct"]} | {fmt(values["jasr0_union"])} |')
    gradient_path = directory/'p0_real_network_gradients.json'
    if gradient_path.is_file():
        lines += ['', '## 真实网络方向梯度检查', '', '在独立float64副本上检查同一操作链；实际部署与攻击为float32。完整数据见 `p0_real_network_gradients.json`。']
        for name, value in json.loads(gradient_path.read_text(encoding='utf-8')).items():
            lines += [f'- {name}：梯度最大绝对值 {value["input_gradient_max"]:.6g}；各差分步长的相对误差 '+
                      ', '.join(f'{v["relative_error"]:.4g}' for v in value['directional_checks'])+'。']
    bootstrap = json.loads((directory/'ordinary_paired_bootstrap.json').read_text(encoding='utf-8'))
    if refinement:
        bootstrap['shared'].update(refinement['bootstrap_vs_main'])
        lines += ['', '## 独立/单层基线的高精度复核', '',
                  f'初版1,000次迭代的最深单层QP有25.2%正常样本未达1e-5间隙，故复用同一攻击缓存，以{refinement["solver_steps"]}次迭代重新计算独立/单层基线的开发阈值和分数。主方法未改、攻击未重新生成；上表的两个基线使用这次复核值。', '',
                  '| 数据 | 层 | 最大相对FW间隙 | 未达标比例 |', '|---|---|---:|---:|']
        for split, layers in refinement['convergence'].items():
            for layer, values in layers.items():
                lines.append(f'| {split} | {layer} | {values["relative_fw_gap_max"]:.5g} | {values["not_converged_fraction"]:.3%} |')
    lines += ['', '## 普通PGD下的原图配对比较', '',
              '以下差值为“消融/基线JASR0减主方法JASR0”，正值支持主方法；并非联合攻击结果。']
    for name, comparisons in bootstrap.items():
        for mode, values in comparisons.items():
            if values is not None:
                lo, hi = values['ci95']
                lines.append(f'- {name} vs {mode}：{100*values["difference"]:.2f}个百分点，95%区间[{100*lo:.2f}, {100*hi:.2f}]。')
    latency_path = directory/'main_mode_latency.json'
    if latency_path.is_file():
        lines += ['', '## 单图推理成本', '', '预热后5次计时的中位数，CUDA同步；主模式单独计时，不含全部消融。', '',
                  '| 模式 | 毫秒/图 | 相对分类器 |', '|---|---:|---:|']
        for name, values in json.loads(latency_path.read_text(encoding='utf-8')).items():
            lines.append(f'| {name} | {1000*values["median_batch_seconds"]:.2f} | {values.get("relative_to_classifier", 1):.2f} |')
        lines += ['', '当前Python展开QP的单图延迟明显超出方案一的初始工程目标；方案二也高于约5倍目标。批量吞吐量与单图延迟不能混淆。']
    verification_path = directory/'witness_verification.json'
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        lines += ['', '## 独立绕过见证审计', '', '重新加载权重和阈值，用原始样本ID逐图核验像素范围、Linf预算、真实误分类和真实分数。']
        for name, values in verification['results'].items():
            lines.append(f'- {name}：{values["reverified_success_count"]}/{values["recorded_success_count"]}个已记录绕过通过重新核验；最大Linf={values["max_linf_recorded"]:.10f}。')
    if 'shared' in metrics and 'evidence' in metrics:
        shared = metrics['shared']; evidence = metrics['evidence']
        advantage = shared['independent']['jasr0_union']-shared['shared']['jasr0_union']
        lines += ['', '## 本轮判断', '',
                  f'- 方案一普通PGD下比独立权重少{100*advantage:.2f}个百分点错误接受；打乱列身份使结果变差，提供初步对应关系信号，尚未达到文档建议的5个百分点门槛。',
                  '- 方案二(u,r)主版本弱于(u,v)无随机对照版本；当前数据未支持随机对照作为检测增益来源。',
                  '- 两个主版本均存在小规模联合PGD真实绕过。当前不满足扩展为自适应防御主结论的条件，不据普通PGD的AUROC启动大规模跨数据集实验。',
                  '- 原分类器训练索引未知，开发正常图本轮准确率为100%。必须先重建可核验的分类器训练/开发/校准划分，才能作严格独立数据上的结论。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores = np.load(directory/'scores.npz')
    if 'shared_clean_shared' in scores:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        refined_scores = np.load(directory/'baseline_refinement_scores.npz') if refinement else None
        for prefix, color, label in [('clean', 'C0', 'Normal development'), ('adv', 'C1', 'PGD')]:
            a = scores[f'shared_{prefix}_shared']
            b = refined_scores[f'{prefix}_independent'] if refined_scores is not None else scores[f'shared_{prefix}_independent']
            axes[0].scatter(b, a-b, s=6, alpha=.35, color=color, label=label)
            axes[1].hist(a, bins=40, density=True, alpha=.45, label=label)
        axes[0].set(xlabel='Independent objective', ylabel='Shared - independent')
        axes[1].set(xlabel='Shared objective', ylabel='Density')
        for ax in axes:
            ax.legend()
        fig.tight_layout(); fig.savefig(directory/'shared_mechanism.png', dpi=180); plt.close(fig)
    if 'evidence_clean_paired' in scores:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, mode in zip(axes, ['paired', 'random_pair']):
            for prefix, label in [('clean', 'Normal development'), ('adv', 'PGD')]:
                values = scores[f'evidence_{prefix}_{mode}']
                ax.hist(np.log1p(values), bins=40, density=True, alpha=.45, label=label)
            ax.set(xlabel=f'log(1 + {mode} score)', ylabel='Density'); ax.legend()
        fig.tight_layout(); fig.savefig(directory/'evidence_scores.png', dpi=180); plt.close(fig)
    if (directory/'evidence_responses.npz').is_file():
        responses = np.load(directory/'evidence_responses.npz')
        fig, ax = plt.subplots(figsize=(6, 4))
        for prefix, label in [('clean', 'Normal development'), ('adv', 'PGD')]:
            values = responses[prefix]
            ax.scatter(values[:, 0], values[:, 1], s=6, alpha=.4, label=label)
        ax.set(xlabel='Keep margin u', ylabel='Control-adjusted drop effect r'); ax.legend()
        fig.tight_layout(); fig.savefig(directory/'evidence_responses.png', dpi=180); plt.close(fig)
    print(str(directory/'report.md'))


if __name__ == '__main__':
    main()
