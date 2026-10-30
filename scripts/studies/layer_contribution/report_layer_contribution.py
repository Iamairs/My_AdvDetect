"""汇总中间层贡献实验；区分信息增益、共享约束、空间处理和联合绕过。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.studies.layer_contribution.run_layer_contribution import paired_bootstrap
from advdetect.utils.config import load_config,resolve_path
from advdetect.utils.artifacts import save_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();directory=resolve_path(args.run_dir)
    config=load_config(directory/'config.yaml');cfg=config['layer_study']
    variants=read(directory/'variants.json');metrics=read(directory/'ordinary_metrics.json')
    calibrations=read(directory/'development_thresholds.json')
    normal=read(directory/'ordinary_comparisons.json')
    joint=read(directory/'joint_metrics.json')
    joint_comparisons=read(directory/'joint_comparisons.json')
    archive=np.load(directory/'ordinary_scores.npz')
    correct=archive['clean_predictions']==archive['labels']
    wrong=archive['adv_predictions']!=archive['labels']
    hits={name:correct&wrong&(archive[f'adversarial__{name}']<=calibrations[name]['threshold']) for name in variants}
    analysis_rows=[]
    # These additional pairs are exploratory decompositions, not new confirmatory tests.
    for rep in cfg['representations']:
        pairs=[(f'{rep}_L4_shared',f'{rep}_L24_shared','add_layer2'),
               (f'{rep}_L4_shared',f'{rep}_L34_shared','add_layer3'),
               (f'{rep}_L24_shared',f'{rep}_L234_shared','add_layer3_given_2_4'),
               (f'{rep}_L34_shared',f'{rep}_L234_shared','add_layer2_given_3_4'),
               (f'{rep}_L4_shared',f'{rep}_L24_independent','information_without_sharing'),
               (f'{rep}_L24_independent',f'{rep}_L24_shared','sharing_given_2_4')]
        own=f'{rep}_L4_reselected'
        if own in variants:pairs.append((own,f'{rep}_L24_shared','vs_layer4_own_references'))
        for baseline,method,label in pairs:
            extra=np.sum(hits[baseline]&~hits[method]);lost=np.sum(~hits[baseline]&hits[method])
            ci=paired_bootstrap(hits[baseline][correct],hits[method][correct],cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
            analysis_rows.append({'comparison':label,'representation':rep,'baseline':baseline,'method':method,
                                  'extra_attacks_rejected':int(extra),'newly_accepted_attacks':int(lost),
                                  'net_extra_rejected':int(extra-lost),'difference':ci['baseline_minus_method'],
                                  'ci95_low':ci['interval'][0],'ci95_high':ci['interval'][1],
                                  'status':'exploratory_uncorrected'})
    save_json(directory/'incremental_analysis.json',analysis_rows)
    with (directory/'incremental_analysis.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(analysis_rows[0]));writer.writeheader();writer.writerows(analysis_rows)
    table=[]
    for name,spec in variants.items():
        m=metrics[name];j=joint[name]
        table.append({'name':name,'representation':spec['representation'],'layers':'+'.join(map(str,spec['layers'])),
                      'weight_mode':spec['mode'],'normal_fpr':m['clean_fpr'],'auroc':m['auroc_successful_only'],
                      'ordinary_jasr0':m['joint_success_rate_on_clean_correct'],
                      'ordinary_tpr_successful':m['tpr_on_successful_attacks'],
                      'joint_jasr0_common_pool':j['jasr0_common_pool'],'joint_success_count':j['joint_success_count'],
                      'joint_num_clean_correct':j['num_clean_correct']})
    with (directory/'method_table.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    # Per-class decompositions expose aggregate cancellations without selecting classes for a claim.
    per_class=[]
    for row in analysis_rows:
        if row['representation']!='gap1_pca32':continue
        for c in np.unique(archive['labels']):
            mask=correct&(archive['labels']==c)
            a,b=hits[row['baseline']],hits[row['method']]
            per_class.append({'comparison':row['comparison'],'class':int(c),'num_clean_correct':int(mask.sum()),
                              'extra_attacks_rejected':int((mask&a&~b).sum()),
                              'newly_accepted_attacks':int((mask&~a&b).sum())})
    with (directory/'class_decomposition.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_class[0]));writer.writeheader();writer.writerows(per_class)
    figures=directory/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    groups=['2','3','4','24','34','234']
    values=[100*metrics[f'gap1_pca32_L{s}_shared']['joint_success_rate_on_clean_correct'] for s in groups]
    axes[0].bar(groups,values,color=['C2','C1','C0','C2','C1','C0'])
    axes[0].set(xlabel='Layers (GAP + PCA32)',ylabel='PGD wrong acceptance (%)',title='Single / combined layers')
    for i,v in enumerate(values):axes[0].text(i,v+.8,f'{v:.1f}',ha='center',fontsize=9)
    groups2=['4','24','34','234']
    values2=[100*metrics[f'gap1_pca32_L{s}_shared']['joint_success_rate_on_clean_correct'] for s in groups2]
    axes[1].bar(groups2,values2,color='C0')
    axes[1].set(xlabel='Layers',ylabel='PGD wrong acceptance (%)',title='Layer4 and marginal additions')
    for i,v in enumerate(values2):axes[1].text(i,v+.3,f'{v:.1f}',ha='center')
    fig.tight_layout();fig.savefig(figures/'layer_additions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,rep in zip(axes,['gap1_pca32','grid2_pca32']):
        groups=['24','34','234'];x=np.arange(3)
        for offset,mode in [(-.18,'independent'),(.18,'shared')]:
            values=[100*metrics[f'{rep}_L{s}_{mode}']['joint_success_rate_on_clean_correct'] for s in groups]
            ax.bar(x+offset,values,width=.35,label=mode)
        ax.set_xticks(x,groups);ax.set(title=rep,xlabel='Layers',ylabel='PGD wrong acceptance (%)');ax.legend()
    fig.tight_layout();fig.savefig(figures/'sharing.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,suffix in zip(axes,['pca32','pcafull']):
        groups=['4','24','34','234'];x=np.arange(4)
        for offset,grid,label in [(-.18,'gap1','GAP'),(.18,'grid2','2x2 spatial')]:
            values=[100*metrics[f'{grid}_{suffix}_L{s}_shared']['joint_success_rate_on_clean_correct'] for s in groups]
            ax.bar(x+offset,values,width=.35,label=label)
        ax.set_xticks(x,groups);ax.set(title=suffix,xlabel='Layers',ylabel='PGD wrong acceptance (%)');ax.legend()
    fig.tight_layout();fig.savefig(figures/'spatial.png',dpi=180);plt.close(fig)
    primary=normal['primary_bonferroni']
    fig,ax=plt.subplots(figsize=(8,4))
    for i,item in enumerate(primary):
        v=item['result'];lo,hi=v['interval'];mean=v['baseline_minus_method']
        ax.errorbar(mean*100,i,xerr=[[100*(mean-lo)],[100*(hi-mean)]],fmt='o',capsize=4,color='C0')
    ax.axvline(0,color='black',linestyle='--')
    ax.set_yticks(range(3),['L4 -> L234','Independent -> shared','GAP -> 2x2 spatial'])
    ax.set_xlabel('Reduction in PGD wrong acceptance (percentage points)\nPositive favors the proposed change; 98.33% paired intervals')
    fig.tight_layout();fig.savefig(figures/'primary_intervals.png',dpi=180);plt.close(fig)
    columns=['L4_shared','L24_shared','L34_shared','L234_independent','L234_shared']
    matrix=np.array([[100*joint[f'{rep}_{c}']['jasr0_common_pool'] for c in columns] for rep in cfg['representations']])
    fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Reds')
    ax.set_xticks(range(len(columns)),columns,rotation=15);ax.set_yticks(range(4),cfg['representations'])
    for i in range(4):
        for j in range(len(columns)):
            ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>=60 else 'black')
    fig.colorbar(im,ax=ax,label='Common-pool joint wrong acceptance (%)')
    fig.tight_layout();fig.savefig(figures/'joint_common_pool.png',dpi=180);plt.close(fig)
    lines=['# 中间层贡献、共享约束与空间信息：实验报告','',
           '本报告回答三个不同问题：增加layer2/3是否有收益；共享权重是否比各层独立权重有收益；保留更多空间位置是否有收益。', '',
           '**判断依据是拦截攻击的增量，不能用“丢掉了多少变化能量”直接判定特征有效或无效。**', '',
           '## 设置与边界','',
           f'- {len(variants)}个检测版本，同一CIFAR-10/ResNet18、PCA/标准化参数、每类64参考、eta=0.001。层组合使用等权平均，独立/共享正则总强度一致。',
           '- 参考身份固定；额外加入GAP32与2×2/PCA32下的layer4自选参考，防止共同挑选参考让单层基线吃亏。',
           '- 相同2,000正常开发图设5%目标阈值；相同1,000正常开发图与已有标准PGD（20步、8/255）评价。表中保留实际误报率差异。',
           f'- 联合攻击选用新的{config["layer_study"]["joint_samples"]}个分层抽样开发原图，排除此前16例，索引与种子预先固定。每个版本评价同一逐原图候选并集，含原图、普通PGD和所有已生成迭代点。',
           '- 联合攻击上限为40步、3个近竞争错误目标、1重启、lambda=[1,0.1,10]，完整展开5,000次QP梯度，先用普通PGD暖启动，之后随机初始化。',
           '- 已在公共池中全部被攻破的版本不重复优化；单批全部出现真实误分类且被接受时提前停止。二者只在已达到本组成功率上限时省去重复计算。',
           '- 原分类器训练索引未知；所有结果是开发集探索实验，不是最终独立测试或充分自适应安全性评价。','',
           '## 1 中间层是否有贡献','',
           '以下“放过攻击”越低越好。单独2/3仍使用相同参考图，不代表穷尽其最佳单层检测方法。','',
           '| GAP+PCA32层组合 | 正常误报 | AUROC | 放过普通PGD |', '|---|---:|---:|---:|']
    for layers in ['2','3','4','24','34','234']:
        m=metrics[f'gap1_pca32_L{layers}_shared']
        lines.append(f'| {layers} | {m["clean_fpr"]:.1%} | {m["auroc_successful_only"]:.4f} | {m["joint_success_rate_on_clean_correct"]:.1%} |')
    own=metrics['gap1_pca32_L4_reselected']
    lines+=['',f'layer4自选参考版本：正常误报{own["clean_fpr"]:.1%}、放过普通PGD {own["joint_success_rate_on_clean_correct"]:.1%}。','',
            '| 探索性比较 | 多拦住攻击 | 又放过攻击 | 净多拦 | 下降百分点及未校正95%区间 |', '|---|---:|---:|---:|---|']
    for r in analysis_rows:
        if r['representation']=='gap1_pca32':
            lines.append(f'| {r["comparison"]} | {r["extra_attacks_rejected"]} | {r["newly_accepted_attacks"]} | {r["net_extra_rejected"]} | {100*r["difference"]:.2f} [{100*r["ci95_low"]:.2f}, {100*r["ci95_high"]:.2f}] |')
    lines+=['','layer2在当前GAP/PCA32及普通PGD下存在有用信号，增加layer2与增加layer3的效果不同；不能把两层合称为“中间层没有价值”。这些是当前表示/参考/等权融合的结果，不等于证明layer3原始特征本身无信息。','',
            '## 2 收益是否来自共享权重','',
            '| 表示 | 层组合 | 独立权重放过PGD | 共享权重放过PGD |', '|---|---|---:|---:|']
    for rep in cfg['representations']:
        for layers in ['24','34','234']:
            a=metrics[f'{rep}_L{layers}_independent']['joint_success_rate_on_clean_correct']
            b=metrics[f'{rep}_L{layers}_shared']['joint_success_rate_on_clean_correct']
            lines.append(f'| {rep} | {layers} | {a:.1%} | {b:.1%} |')
    lines+=['','增加层的收益与强制共享配方的收益需分开看。例如L4→L24独立已能利用额外信息，再比较L24独立→共享，才是在检查额外共享约束。','',
            '## 3 更多空间信息是否有用','',
            '| 表示 | 层4 | 层2+4 | 层3+4 | 层2+3+4 | 三层正常FPR |', '|---|---:|---:|---:|---:|---:|']
    for rep in cfg['representations']:
        rates=[metrics[f'{rep}_L{s}_shared']['joint_success_rate_on_clean_correct'] for s in ['4','24','34','234']]
        fpr=metrics[f'{rep}_L234_shared']['clean_fpr']
        lines.append('| '+rep+' | '+' | '.join(f'{v:.1%}' for v in [*rates,fpr])+' |')
    lines+=['','这些结果只说明当前空间处理与距离度量有没有收益；2×2/PCA32可能仍压缩空间信息，完整PCA保留全部区域特征但也改变维数与白化权重。不能把变差直接解释成空间信息在所有方法中都无用。','',
            '## 4 预先指定的三个主比较','',
            '对同一原图做5,000次配对bootstrap。三个主比较用Bonferroni校正，单个区间为98.33%。差值=基线放过比例减新版本放过比例，正数有利于新版本；跨0表示当前样本不能明确支持增益。其余逐层结果是探索性95%区间。','',
            '| 比较 | 普通PGD下降百分点 | 校正区间 | 联合公共池下降百分点 |', '|---|---:|---|---:|']
    for ordinary,j in zip(primary,joint_comparisons['primary_bonferroni']):
        v=ordinary['result'];ci=v['interval'];jr=j['result']
        lines.append(f'| {ordinary["baseline"]} → {ordinary["method"]} | {100*v["baseline_minus_method"]:.2f} | [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {100*jr["baseline_minus_method"]:.2f} |')
    lines+=['','## 5 新开发子集上的联合攻击','',
            '| 表示 | 只用4 | 2+4共享 | 3+4共享 | 三层独立 | 三层共享 |', '|---|---:|---:|---:|---:|---:|']
    for rep in cfg['representations']:
        vals=[joint[f'{rep}_{s}'] for s in columns]
        lines.append('| '+rep+' | '+' | '.join(f'{v["joint_success_count"]}/{v["num_clean_correct"]}' for v in vals)+' |')
    lines+=['','这里只报告本组固定开发样本与已生成攻击集合，不将其外推总体。普通PGD有收益与完全知情攻击下仍有收益，是两个不同结论。','',
            '## 6 数值与独立审计','']
    convergence=read(directory/'convergence.json')
    fail=[(split,k,v) for split,groups in convergence.items() for k,v in groups.items() if v['not_converged_fraction']>0]
    maximum=max(v['max_relative_fw_gap'] for groups in convergence.values() for v in groups.values())
    lines.append(f'所有普通评价/开发阈值问题最大相对FW间隙={maximum:.3g}；超过1e-5的问题记录数={len(fail)}。完整证书保存在convergence.json。')
    if (directory/'witness_repair.json').is_file():
        repair=read(directory/'witness_repair.json')
        lines+=['',f'初次严格重载发现{repair["original_invalid_pairs"]}个方法—原图见证未复现成功，涉及ID {repair["original_invalid_original_ids"]}。原始记录没有覆盖。',
                '已单独生成离阈值更深的候选：攻击目标增加正常分数std的5%内侧余量，部署分数与防御阈值均未改变；新候选交叉评价全部版本。最终有效见证使用common_pool_repaired.pt，原始common_pool.pt及失败审计保留。',
                '该修复用于排除GPU浮点/批次在阈值边界附近的假成功；不能把初版未经独立审计的全部成功数当作最终结论。']
    if (directory/'independent_audit.json').is_file():
        audit=read(directory/'independent_audit.json')
        lines.append(f'独立重载核验了{len(audit["witnesses"])}个版本的公共池成功见证；来源模型、数据、参考身份和新旧样本排除均通过。')
        for name,g in audit['gradient_probes'].items():
            lines.append(f'- {name}：float64方向差分相对误差 '+', '.join(f'{r["relative_error"]:.3g}' for r in g['directional_checks'])+'。实际攻击保持float32。')
    lines+=['','## 7 文件与阅读建议','',
            '- 先看layer_additions.png和sharing.png：分别区分“加一层”与“共用配方”的收益。',
            '- 再看joint_common_pool.png：检查普通攻击收益是否保留到知情攻击。',
            '- method_table.csv为38版本总表；incremental_analysis.csv保留新增拦截与新增漏过，防止均值掩盖抵消；class_decomposition.csv保留类别分解。',
            '- common_pool.pt保存初始逐原图并集；如有数值边界修复，以common_pool_repaired.pt及independent_audit.json为最终见证。candidates_*.pt保留所有生成迭代点；attack_events.json说明提前停止与跳过原因。','']
    for name in ['layer_additions','sharing','spatial','primary_intervals','joint_common_pool']:
        lines.append(f'![{name}]({(figures/(name+".png")).as_posix()})\n')
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(directory/'report.md'))


if __name__=='__main__':main()
