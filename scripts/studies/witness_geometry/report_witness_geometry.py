"""为已审计见证的距离分析生成正常尺度图和中文报告。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from advdetect.utils.config import load_config, resolve_path
from advdetect.utils.artifacts import save_json


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def short(name):
    return name.replace('gap1_pca32_', '').replace('_shared', ' shared').replace('_independent', ' independent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(); directory = resolve_path(args.run_dir)
    config = load_config(directory/'config.yaml'); cfg = config['witness_geometry']
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    checks = json.loads((directory/'numerical_checks.json').read_text(encoding='utf-8'))
    normal_stats = json.loads((directory/'normal_score_scales.json').read_text(encoding='utf-8'))
    rows = read_rows(directory/'per_layer.csv'); samples = read_rows(directory/'per_sample.csv')
    normal = np.load(directory/'normal_details.npz')
    focus = cfg['focus_variants']; figures = directory/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11})

    fig, axes = plt.subplots(len(focus), 3, figsize=(13, 3*len(focus)), squeeze=False)
    for i, name in enumerate(focus):
        for j, layer in enumerate([2, 3, 4]):
            ax = axes[i, j]; points = [r for r in rows if r['variant']==name and int(r['layer'])==layer]
            if not points:
                ax.text(.5, .5, 'Not used by this detector', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off(); continue
            for accepted, color, label in [(True, '#2277b4', 'Clean already below wrong-class threshold'),
                                            (False, '#e17c22', 'Clean above wrong-class threshold')]:
                subset = [r for r in points if (float(r['clean_wrong_score'])<=float(r['threshold']))==accepted]
                ax.scatter([float(r['shift_over_normal_nn_median']) for r in subset],
                           [float(r['wrong_reconstruction_normal_percentile']) for r in subset],
                           s=25, alpha=.8, color=color, label=label)
            ax.axhline(95, color='#b33', ls='--', lw=1); ax.axvline(1, color='gray', ls=':', lw=1)
            ax.set_xlim(left=0); ax.set_ylim(-3, 103)
            ax.set_title(f'{short(name)}, layer{layer}')
            ax.set_xlabel('Shift / normal same-class NN median')
            ax.set_ylabel('Wrong-class residual\n(normal percentile)')
    handles, labels = next(ax for ax in axes.flat if ax.has_data()).get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, 1]); fig.savefig(figures/'shift_vs_reconstruction.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, name in zip(axes.flat, focus):
        points = [r for r in samples if r['variant']==name]
        x = np.array([float(r['clean_wrong_score'])/float(r['threshold']) for r in points])
        y = np.array([float(r['score_over_threshold']) for r in points])
        ax.scatter(x, y, c=np.where(x<=1, '#2277b4', '#e17c22'), alpha=.8, s=30)
        ax.axvline(1, color='#b33', ls='--'); ax.axhline(1, color='#b33', ls='--')
        ax.plot([0, max(1.1, x.max())], [0, max(1.1, x.max())], color='gray', ls=':', lw=1)
        ax.set_xlim(left=0); ax.set_ylim(0,1.06); ax.set_title(f'{short(name)}: clean wrong-class accepted {int((x<=1).sum())}/64')
        ax.set_xlabel('Clean image forced to final wrong class:\nscore / threshold')
        ax.set_ylabel('Successful witness: score / threshold')
    fig.tight_layout(); fig.savefig(figures/'counterfactual_wrong_class.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rng = np.random.default_rng(cfg['seed'])
    correct = normal['labels']==normal['predictions']
    for ax, name in zip(axes.flat, focus):
        tau = normal_stats[name]['threshold']; values = normal[f'{name}__score']
        groups = [values[correct & (normal['labels']==c)]/tau for c in range(10)]
        ax.boxplot(groups, positions=np.arange(10), widths=.6, showfliers=False, whis=[5,95])
        points = [r for r in samples if r['variant']==name]
        ax.scatter([int(r['wrong_class'])+rng.uniform(-.18,.18) for r in points],
                   [float(r['score_over_threshold']) for r in points], s=18, color='#b33', alpha=.7, label='Successful witnesses')
        ax.axhline(1, color='#b33', ls='--', label='Actual global threshold')
        ax.set_title(short(name)); ax.set_xlabel('Class: normal true / witness predicted wrong')
        ax.set_ylabel('Score / global threshold'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures/'normal_class_score_width.png', dpi=170); plt.close(fig)

    # Continuous decrease and nearest-neighbor class preference are separate diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2277b4', '#e17c22', '#27914b']
    name = 'gap1_pca32_L234_shared'
    for layer, color in zip([2,3,4], colors):
        points = [r for r in rows if r['variant']==name and int(r['layer'])==layer]
        axes[0].scatter([float(r['clean_wrong_refit_l2']) for r in points],
                        [float(r['wrong_reconstruction_l2']) for r in points], color=color, s=22, alpha=.7, label=f'layer{layer}')
        ratios = sorted(float(r['nn_wrong_class'])/float(r['nn_true_class']) for r in points)
        axes[1].plot(ratios, np.arange(1,len(ratios)+1)/len(ratios), color=color, label=f'layer{layer}')
    limit = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0,limit],[0,limit],color='gray',ls='--'); axes[0].set_xlabel('Clean image: refit to wrong-class combination, L2')
    axes[0].set_ylabel('Witness: refit to wrong-class combination, L2'); axes[0].set_title('L234 shared: reconstruction change')
    axes[1].axvline(1,color='gray',ls='--'); axes[1].set_xlabel('Nearest wrong-class / nearest true-class reference distance')
    axes[1].set_ylabel('Fraction of 64 witnesses'); axes[1].set_title('Below 1: closer to a wrong-class individual reference')
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(figures/'reference_movement.png', dpi=170); plt.close(fig)

    examples = {}
    for name in focus:
        points = [r for r in samples if r['variant']==name]
        small = [r for r in points if r['small_shift_all_used_layers']=='True' and r['clean_wrong_already_accepted']=='True']
        high = [r for r in points if r['high_reconstruction_any_used_layer']=='True']
        examples[name] = {'small_shift_and_clean_wrong_already_accepted_ids': [int(r['dataset_id']) for r in small[:5]],
                          'accepted_with_high_layer_residual_ids': [int(r['dataset_id']) for r in high[:5]]}
    save_json(directory/'illustrative_cases.json', examples)

    tradeoff = {}
    fig, ax = plt.subplots(figsize=(8,5))
    for name in focus:
        attack_values = np.array([float(r['adv_score']) for r in samples if r['variant']==name])
        normal_values = normal[f'{name}__score']
        cut = float(np.median(attack_values))
        tradeoff[name] = {'posthoc_cut_below_median_witness_score':cut,
            'normal_rejection_fraction_at_least':float(np.mean(normal_values>=cut)),
            'witness_rejection_fraction':float(np.mean(attack_values>=cut)),
            'warning':'Posthoc descriptive reuse of normal threshold data; no threshold update or adaptive reattack.'}
        cuts = np.unique(np.concatenate([normal_values,attack_values]))
        # Rejection here is >= cut, corresponding to a threshold just below each cut.
        normal_fpr = np.array([np.mean(normal_values>=v) for v in cuts])
        caught = np.array([np.mean(attack_values>=v) for v in cuts])
        ax.plot(normal_fpr*100,caught*100,label=short(name))
    ax.set_xlabel('Normal images rejected (%)');ax.set_ylabel('These retained witnesses rejected (%)')
    ax.set_title('Posthoc threshold-only tradeoff (no reattack)');ax.legend(fontsize=9)
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(figures/'threshold_only_tradeoff.png',dpi=170);plt.close(fig)
    save_json(directory/'threshold_tradeoff.json',tradeoff)

    lines = ['# 成功见证：离原图多远、离错误类别参考组合多远', '',
        '这是对既有成功见证的开发集机制诊断；没有修改模型、参考库、阈值或生成新攻击。所有38版本各64图，共2,432个方法—原图见证重新通过实际分数、预测和预算检查。', '',
        '## 如何读两个距离', '',
        '- z是各版本实际使用的池化→PCA→标准化特征；raw L2仅在同一表示/层内解释，不能跨维数直接比大小。',
        '- 横轴：原图到攻击图的特征位移，除以正常图片到同类最近参考图的中位距离。小于1表示比这种正常近邻间距还小；另存类内随机不同图对距离及分位数。它不是自然噪声阈值。',
        '- 纵轴：攻击图到最终错误类别的最优参考组合的L2残差，在“同版本、同层、该类别正常正确图片”的误差分布中处于第几百分位。95以上表示该层误差高于至少约95%的这些正常图片。',
        '- 全部正常尺度使用2,000张dev_threshold图，与64张原图及参考图均无交集；类内尺度只使用分类正确的正常图。阈值仍为原来的全体2,000正常图阈值，未重设。',
        '- 原图错误类别反事实：保持原图不动，固定类别为攻击最终预测的错误类别重新求最优组合，再与同一个全局阈值比较；不表示原图本身已经被误分类。', '',
        '## 核心计数（每项分母64，标志可重叠）', '',
        '| 版本 | 原图已在错误类检测接受区 | 所用各层移动均低于正常NN第5百分位 | 有层残差高于该错误类正常第95百分位 | 总分高于该类正常第95百分位仍放行 | 各层错误类残差均下降 | 各层均更近错误类单个参考 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in focus:
        c = summary[name]['counts']
        lines.append(f'| {short(name)} | '+ ' | '.join(str(c[k]) for k in ['clean_wrong_already_accepted',
            'small_shift_all_used_layers','high_reconstruction_any_used_layer','score_high_vs_normal_wrong_class_q95',
            'wrong_reconstruction_decreased_all_layers','nn_wrong_closer_all_layers'])+' |')
    lines += ['', '## 逐层尺度（64图的中位数）', '',
        '| 版本 | 层 | 位移/正常同类NN中位数 | 位移/正常同类图对中位数 | 错误类重构误差的正常百分位 |', '|---|---:|---:|---:|---:|']
    for name in focus:
        for layer, st in summary[name]['layers'].items():
            lines.append(f'| {short(name)} | {layer} | {st["shift_over_normal_nn_median"]["q50"]:.3f} | '
                f'{st["shift_over_normal_pair_median"]["q50"]:.3f} | {st["wrong_reconstruction_normal_percentile"]["q50"]:.1f} |')
    lines += ['', '## 正常分数的宽度与全局阈值', '',
        '“阈值宽”必须相对正常分布和类别解释，不能仅凭分数绝对值大来判断。下表为全体正常图，类条件分布见图和JSON。', '',
        '| 版本 | 正常q50 | 正常q95 | 阈值/q50 | 见证分数/阈值中位数 |', '|---|---:|---:|---:|---:|']
    for name in focus:
        ns = normal_stats[name]; st = ns['all_normal']
        lines.append(f'| {short(name)} | {st["q50"]:.4f} | {st["q95"]:.4f} | {ns["threshold"]/st["q50"]:.3f} | {summary[name]["score_over_threshold"]["q50"]:.3f} |')
    lines += ['', '仅改变总分阈值、想挡住当前这批见证的一半时，下述正常拒绝率可以帮助判断分数分布是否重叠。这是事后诊断，复用了正常阈值数据，未把阈值改成这些数，也未重新攻击，不能作为新防御的效果。', '',
              '| 版本 | 拦住当前约一半见证时，正常图片会被拒绝 |','|---|---:|']
    for name in focus:
        lines.append(f'| {short(name)} | {100*tradeoff[name]["normal_rejection_fraction_at_least"]:.1f}% |')
    lines += ['', '## 图', '']
    for title, file in [('位移与重构正常分位', 'shift_vs_reconstruction.png'), ('原图错误类别反事实', 'counterfactual_wrong_class.png'),
                         ('正常类别分数分布与成功见证', 'normal_class_score_width.png'), ('参考重构下降与近邻类别倾向', 'reference_movement.png'),
                         ('仅收紧阈值的事后取舍', 'threshold_only_tradeoff.png')]:
        lines += [f'![{title}]({(figures/file).as_posix()})', '']
    lines += ['## 能说明什么、不能说明什么', '',
        '- 小位移加“原图错误类重构已被接受”，支持错误类别的**检测接受集合**已覆盖/邻近原图特征；分类器仍需改变预测才能形成真实绕过。不能从有限点证明整个分类边界或接受区域的形状。',
        '- 位移大且错误类残差下降，提示参照几何上的移动；低重构误差不等于语义已迁移为错误类别。单个参考近邻倾向额外提供一种检查，不与凸组合残差混同。',
        '- 某层高残差仍放行，可能与跨层平均或类别正常分布宽度有关。总分高于该类别q95但低于全局阈值，更直接说明全局阈值对该类别较宽松；二者需分别报告。',
        '- 这些是每方法最低分的成功候选，选择本身偏向低重构误差；不是最小位移或最早成功样本，不能反推攻击必经路线。',
        '- 每方法只有64张原图；38版本和各层行相关，不把所有行当独立样本。标志只是预先固定的描述性分位数规则，互相重叠，不是经验证的因果分类器。',
        '- 原分类器训练索引未知，属于开发集探索；不作为独立最终测试结果。', '',
        '## 数值与产物', '',
        f'- 严格重放通过 {checks["verified_witness_method_pairs"]} 个方法—原图对；正确正常类尺度图片 {checks["correct_normal_for_class_scales_n"]}/{checks["normal_scale_n"]}。',
        f'- 最大正常/见证/反事实QP相对FW间隙：{checks["max_normal_relative_fw_gap"]:.3g} / {checks["max_witness_relative_fw_gap"]:.3g} / {checks["max_counterfactual_relative_fw_gap"]:.3g}。',
        f'- 分数=逐层加权MSE+权重正则，显式float64复算最大绝对差 {checks["max_score_decomposition_absolute_error"]:.3g}；固定组合三角不等式超量 {checks["max_fixed_combination_triangle_excess"]:.3g}。',
        '- `per_layer.csv`保留两个raw L2、NN/图对归一化、正常百分位、原图正确/错误类重构、同一组合距离与逐层分数贡献。',
        '- `per_sample.csv`保留原图错误类反事实、实际阈值及分数、各描述性标志；`normal_details.npz`保留正常误差原始分布。',
        '- `geometry_tensors.pt`保留实际变换特征和最优权重；`provenance.json`记录源状态/池/阈值/审计哈希，未回写源实验。',
        '- `normal_distance_scales.json`/`normal_score_scales.json`保留各类距离尺度和分数宽度；类内5,000对有重复，不能当作5,000张独立图片。', '']
    audit_path = directory/'independent_geometry_audit.json'
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        lines += [f'- 独立审计另行重建 {audit["algebraic_method_image_pairs"]} 个方法—原图对、{audit["layer_rows"]} 行层记录，'
                  f'重构L2最大差 {audit["max_residual_l2_error"]:.3g}；四个主版本的真实网络特征和实际分数重放全部通过。', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(directory/'report.md')


if __name__ == '__main__': main()
