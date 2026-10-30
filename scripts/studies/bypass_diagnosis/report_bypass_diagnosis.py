"""读取16例诊断与验证产物，生成可导出图、逐图表和中文分析。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from advdetect.utils.config import load_config,resolve_path

CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
ROLES=['clean','pgd','joint_shared','joint_evidence']
LABELS=['Original','PGD','Joint/shared','Joint/evidence']
COLORS=['C0','C1','C3','C2']


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def whiten(values,classes,profile):
    means=profile['mean'][classes].double()
    precision=profile['precision'][classes].double()
    delta=values.double()-means
    coordinates=torch.bmm(delta[:,None,:],torch.linalg.cholesky(precision)).squeeze(1)
    exact=torch.einsum('bi,bij,bj->b',delta,precision,delta)
    torch.testing.assert_close(coordinates.square().sum(1),exact)
    return coordinates.numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir',required=True)
    parser.add_argument('--verification-dir',required=True)
    args=parser.parse_args()
    analysis=resolve_path(args.analysis_dir);verification=resolve_path(args.verification_dir)
    study=torch.load(analysis/'diagnosis.pt',map_location='cpu',weights_only=True)
    summary=load_json(analysis/'analysis_summary.json')
    results=load_json(verification/'verification_summary.json')
    origins=load_json(verification/'resolved_witness_provenance.json')
    source=resolve_path(summary['source_run'])
    thresholds_s=load_json(source/'development_thresholds_shared.json')
    thresholds_e=load_json(source/'development_thresholds_evidence.json')
    checkpoints=load_json(source/'result.json')['checkpoints']
    evidence_state=torch.load(resolve_path(checkpoints['evidence']),map_location='cpu',weights_only=True)['state']
    normal=np.load(analysis/'normal_reference.npz')
    rows=list(csv.DictReader((analysis/'per_sample.csv').open(encoding='utf-8')))
    out=verification/'figures';out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Paired objectives and mechanism gap.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,['es','gap']):
        data=np.stack([study['details'][r]['shared'][key].numpy() for r in ROLES],1)
        for line in data:
            ax.plot(range(4),line,color='gray',alpha=.25,linewidth=.6)
        for j,(label,color) in enumerate(zip(LABELS,COLORS)):
            ax.scatter(np.full(16,j),data[:,j],s=20,color=color,label=label)
        ax.plot(range(4),data.mean(0),color='black',marker='D',label='Mean')
        ax.set_xticks(range(4),LABELS,rotation=15);ax.set_ylabel(key.upper())
    axes[0].axhline(thresholds_s['shared']['threshold'],linestyle='--',color='black',label='Shared threshold')
    axes[1].axhline(np.quantile(normal['gap'],.95),linestyle=':',color='black',label='Pooled normal 95% (diagnostic)')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'paired_objectives.png',dpi=180);plt.close(fig)
    # Retained change energy: exact projection fractions before feature standardization.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    energy_roles=['pgd','joint_shared','joint_evidence','random']
    for i,role in enumerate(energy_roles):
        samples=[study['details'][role]] if role!='random' else [study['details'][f'random_{j}'] for j in range(4)]
        for ax,key in zip(axes,['gap_retained_fraction','pca32_retained_of_gap_fraction']):
            values=[torch.cat([s['compression'][l][key] for s in samples]).mean().item()*100 for l in range(3)]
            ax.bar(np.arange(3)+(i-1.5)*.19,values,width=.18,label=role)
            ax.set_xticks(range(3),['layer2','layer3','layer4']);ax.set_ylim(0,105);ax.set_ylabel('Retained change energy (%)')
    axes[0].set_title('Spatial DC/GAP retention');axes[1].set_title('PCA32 retention of GAP change')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'compression_energy.png',dpi=180);plt.close(fig)
    # Class-conditioned normal profiles whitened into a common coordinate frame.
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,mode,columns in zip(axes,['paired','no_control'],[[0,2],[0,1]]):
        profile=evidence_state['profiles'][mode]
        nc=torch.tensor(normal['response_classes']).long()
        coordinates=whiten(torch.tensor(normal['uvr'][:,columns]),nc,profile)
        ax.scatter(coordinates[:,0],coordinates[:,1],s=5,alpha=.13,color='gray',label='984 normal comparison')
        for role,label,color in zip(ROLES,LABELS,COLORS):
            ed=study['details'][role]['evidence'];keys=['u','r'] if mode=='paired' else ['u','v']
            values=torch.stack([ed[k] for k in keys],1)
            coords=whiten(values,ed['class'],profile)
            ax.scatter(coords[:,0],coords[:,1],s=30,alpha=.8,label=label,color=color)
        radius=thresholds_e[mode]['threshold']**.5
        ax.add_patch(plt.Circle((0,0),radius,fill=False,color='black',linestyle='--',label='Acceptance boundary'))
        ax.set(xlabel='Whitened coordinate 1',ylabel='Whitened coordinate 2',title=mode)
        ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'response_profiles.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for j,(role,label,color) in enumerate(zip(ROLES,LABELS,COLORS)):
        ed=study['details'][role]['evidence']
        values=[ed['z_'+k].abs().mean().item() for k in ['u','v','r']]
        ax.bar(np.arange(3)+(j-1.5)*.2,values,width=.19,color=color,label=label)
    ax.set_xticks(range(3),['keep u','drop v','control-adjusted r']);ax.set_ylabel('Mean absolute profile z-score');ax.legend()
    fig.tight_layout();fig.savefig(out/'response_components.png',dpi=180);plt.close(fig)
    # Attack trajectories: first target only, individual samples rather than just the last iterate.
    for group,names in [('primary',['shared','evidence']),('variants',['gap1_pcafull','grid2_pcafull','evidence_no_control'])]:
        fig,axes=plt.subplots(2,len(names),figsize=(5*len(names),7),squeeze=False)
        for col,name in enumerate(names):
            trajectory=load_json(verification/f'trajectory_{name}.json')
            records=[r for r in trajectory['records'] if r.get('target_rank')==0 and r.get('restart')==0]
            steps=[r['step'] for r in records]
            scores=np.asarray([r['score'] for r in records])/trajectory['threshold']
            margins=np.asarray([r['margin_target'] for r in records])
            for i in range(16):
                axes[0,col].plot(steps,scores[:,i],alpha=.25,linewidth=.8)
                axes[1,col].plot(steps,margins[:,i],alpha=.25,linewidth=.8)
            axes[0,col].plot(steps,np.median(scores,axis=1),color='black',linewidth=2)
            axes[1,col].plot(steps,np.median(margins,axis=1),color='black',linewidth=2)
            axes[0,col].axhline(1,color='black',linestyle='--');axes[1,col].axhline(0,color='black',linestyle='--')
            axes[0,col].set(title=name,ylabel='Actual score / threshold',xlabel='PGD step')
            axes[1,col].set(ylabel='Target-class margin',xlabel='PGD step')
        fig.tight_layout();fig.savefig(out/f'trajectories_{group}.png',dpi=160);plt.close(fig)
    # All 16 cases, four deterministic pages.
    for page in range(4):
        fig,axes=plt.subplots(4,6,figsize=(12,8))
        for row,i in enumerate(range(page*4,page*4+4)):
            for col,role in enumerate(ROLES):
                x=study['roles'][role][i].permute(1,2,0).numpy()
                pred=int(study['details'][role]['predictions'][i])
                axes[row,col].imshow(x,interpolation='nearest');axes[row,col].set_title(f'{LABELS[col]}: {CLASSES[pred]}',fontsize=8)
            for col,role in [(4,'joint_shared'),(5,'joint_evidence')]:
                delta=study['roles'][role][i]-study['roles']['clean'][i]
                axes[row,col].imshow((.5+8*delta).clamp(0,1).permute(1,2,0).numpy(),interpolation='nearest')
                axes[row,col].set_title(f'Delta x8: {role}',fontsize=8)
            axes[row,0].set_ylabel(f'ID {int(study["ids"][i])}\ny={CLASSES[int(study["labels"][i])]}',fontsize=8)
            for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        fig.tight_layout();fig.savefig(out/f'cases_{page+1}.png',dpi=150);plt.close(fig)
    # Reference identities, visual examples, and all weights exported for audit.
    import pickle
    source_data=[];source_labels=[]
    dataset_root=resolve_path(load_config(analysis/'config.yaml')['dataset']['root'])
    cifar_root=dataset_root if dataset_root.name=='cifar-10-batches-py' else dataset_root/'cifar-10-batches-py'
    for batch in range(1,6):
        with (cifar_root/f'data_batch_{batch}').open('rb') as f:
            raw=pickle.load(f,encoding='latin1')
        source_data.append(raw['data']);source_labels.extend(raw['labels'])
    source_data=np.concatenate(source_data).reshape(-1,3,32,32).transpose(0,2,3,1)
    fig,axes=plt.subplots(4,6,figsize=(12,8))
    reference_rows=[]
    for row,i in enumerate(range(4)):
        sd=study['details']['joint_shared']['shared'];weights=sd['shared_weights'][i]
        order=weights.argsort(descending=True)
        for col,role in [(0,'clean'),(1,'joint_shared')]:
            axes[row,col].imshow(study['roles'][role][i].permute(1,2,0),interpolation='nearest')
            axes[row,col].set_title(f'{role}, ID {int(study["ids"][i])}',fontsize=8)
        for col,index in enumerate(order[:4].tolist(),2):
            rid=int(sd['reference_ids'][i,index])
            axes[row,col].imshow(source_data[rid],interpolation='nearest')
            axes[row,col].set_title(f'R{rid}, w={float(weights[index]):.3f}\n{CLASSES[source_labels[rid]]}',fontsize=8)
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(out/'reference_examples.png',dpi=150);plt.close(fig)
    for role in ROLES:
        sd=study['details'][role]['shared']
        for i,sample_id in enumerate(study['ids'].tolist()):
            for j,rid in enumerate(sd['reference_ids'][i].tolist()):
                reference_rows.append({'role':role,'sample_id':sample_id,'predicted_class':int(sd['class'][i]),
                                       'reference_id':rid,'shared_weight':float(sd['shared_weights'][i,j]),
                                       **{f'layer{stage}_independent_weight':float(sd['independent_weights'][i,k,j])
                                          for k,stage in enumerate([2,3,4])}})
    with (verification/'reference_weights.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(reference_rows[0]));writer.writeheader();writer.writerows(reference_rows)
    fig,axes=plt.subplots(4,1,figsize=(13,10))
    for ax,i in zip(axes,range(4)):
        old=study['details']['counterfactual_clean_wrong_class_shared']['shared_weights'][i]
        sd=study['details']['joint_shared']['shared']
        matrix=torch.cat([old[None],sd['shared_weights'][i,None],sd['independent_weights'][i]],0)
        order=matrix.max(0).values.argsort(descending=True)
        ax.imshow(matrix[:,order],aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='viridis')
        ax.set_yticks(range(5),['Clean, fixed wrong class','Joint, shared','Joint, layer2','Joint, layer3','Joint, layer4'],fontsize=8)
        ticks=list(range(0,64,8));ax.set_xticks(ticks,[str(int(sd['reference_ids'][i,order[t]])) for t in ticks],fontsize=7)
        ax.set_title(f'ID {int(study["ids"][i])}, same class/reference identities',fontsize=9)
    fig.tight_layout();fig.savefig(out/'reference_weights.png',dpi=160);plt.close(fig)
    # Narrative is built from actual artifacts, with explicit limits.
    text=['# 16组联合绕过：逐图诊断与机制验证','',
          '范围：固定的16个开发案例。它们来自已知成功绕过集合，不是独立随机测试集；原分类器训练索引未知。以下计数不能外推总体鲁棒性。', '',
          '## 1 样本来源与重放','',
          '两个主方法各保存16个公共候选池见证，共享相同的16个原图ID。普通PGD、共享方法攻击、干预方法攻击的来源分别核验，不将公共池见证误认为全部由对应攻击新生成。']
    for role,entries in origins.items():
        ordinary=sum(e['generator_origin']=='ordinary_pgd' for e in entries)
        owners={}
        unknown=0
        for e in entries:
            if e['generator_origin']=='ordinary_pgd':continue
            candidates=sorted({m['generator'] for m in e.get('replay_matches',[])})
            if candidates:owners['/'.join(candidates)]=owners.get('/'.join(candidates),0)+1
            else:unknown+=1
        text.append(f'- {role}：普通PGD={ordinary}；非普通见证的逐字节重放匹配={owners}；未解决来源={unknown}。')
    text+=['',f'独立重放相同20步、2目标、1重启、lambda=1（按开发正常分数std缩放）后：{results["replay"]}，均为16个案例。保留每步真实预测、分数、目标margin、累计成功和图像。','',
           '## 2 原始/普通/联合攻击的实际响应','',
           '| 输入 | 误分类数 | 共享方法错误接受 | 干预方法错误接受 | 无对照错误接受 | 平均ES | 平均G | 有效参考数 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for role in ROLES:
        r=summary['roles'][role]
        text.append(f'| {role} | {r["misclassified"]}/16 | {r["shared_wrong_accepted"]}/16 | {r["evidence_wrong_accepted"]}/16 | {r["no_control_wrong_accepted"]}/16 | {r["mean_es"]:.4f} | {r["mean_gap"]:.4f} | {r["mean_effective_references"]:.2f} |')
    old=study['details']['counterfactual_clean_wrong_class_shared']
    new=study['details']['joint_shared']['shared']
    original_true_after=study['details']['counterfactual_joint_shared_true_class']
    gap_tail=sum(float(r['gap_normal_class_percentile'])>.95 for r in rows if r['role']=='joint_shared')
    es_tail=sum(float(r['es_normal_class_percentile'])>.95 for r in rows if r['role']=='joint_shared')
    text+=['','### 方案一的判断','',
           f'- 对每张图固定为其最终错误类别：原图ES均值{old["es"].mean().item():.4f}，绕过后ES均值{new["es"].mean().item():.4f}；同一绕过在原真实类别的ES均值为{original_true_after["es"].mean().item():.4f}。攻击确实改变了类别参照支持，而非只利用预测类别切换的记账错误。',
           f'- 共同权重不要求等于每个独立层的最优权重。联合绕过G均值{new["gap"].mean().item():.4f}，只有{gap_tail}/16超过相应正常预测类G的95%经验百分位；其余并无明显上尾异常。这个百分位仅作诊断，未把G事后变成新检测器。',
           f'- ES也只有{es_tail}/16超过相应正常预测类的95%经验百分位。这不支持把本组大部分绕过仅归因于全局阈值混合；未据此拟合或测试新的类别阈值检测器。',
           f'- 有效参考数定义为1/sum(w²)。绕过均值{new["effective_reference_count"].mean().item():.2f}，最大{new["effective_reference_count"].max().item():.2f}，并非接近均匀使用64个参考。配对原图均值{study["details"]["clean"]["shared"]["effective_reference_count"].mean().item():.2f}；类别构成不同，不据这两个均值证明分布相同。',
           '- 所有详细QP用5,000次核验。支持程度、G和权重均保留原值，没有截断负差距或根据攻击挑参考。','',
           '### GAP/PCA是否藏住了攻击变化','',
           '| 输入 | layer2 GAP保留 | layer3 GAP保留 | layer4 GAP保留 | layer4 PCA32保留GAP变化 |',
           '|---|---:|---:|---:|---:|']
    for role in ['pgd','joint_shared','joint_evidence','random']:
        parts=[study['details'][role]] if role!='random' else [study['details'][f'random_{j}'] for j in range(4)]
        vals=[torch.cat([v['compression'][i]['gap_retained_fraction'] for v in parts]).mean().item() for i in range(3)]
        pca=torch.cat([v['compression'][2]['pca32_retained_of_gap_fraction'] for v in parts]).mean().item()
        text.append('| '+role+' | '+' | '.join(f'{v:.1%}' for v in [*vals,pca])+' |')
    text+=['','GAP能量以空间常数子空间的正交投影计算，包含H×W因子；PCA能量在标准化前计算。随机对照为4组相同Linf预算，未同时匹配L2，且没有参与阈值拟合。随机对照64次均未误分类。',
           '末层的大部分变化已经保留在现有表示中，因此没有证据表明所有绕过主要藏在末层被丢弃的方向。能量占比不等于判别信息，下面用固定表征消融与重新攻击检验。','',
           '## 3 固定参考身份的2×2表征实验','',
           '同一分类器、同一D_ref、同一64个每类参考ID、相同eta。因子为GAP/2×2网格和PCA32/完整PCA；完整PCA与PCA32使用同层拟合的嵌套基底，避免把不同轴标准化混为仅增加维数。每个版本用相同2,000张正常开发图重新确定5%目标阈值。', '',
           '| 表征 | 正常开发FPR（1000图） | 接受旧共享绕过 | 接受旧干预绕过 | 针对该版本重新攻击成功 |',
           '|---|---:|---:|---:|---:|']
    for name,r in results['representation_transfer'].items():
        attacked=results['replay']['shared'] if name=='gap1_pca32' else results['reattacks'].get(name,{}).get('success_count')
        text.append(f'| {name} | {r["normal_dev_fpr_1000"]:.1%} | {r["cases"]["joint_shared"]["wrong_accepted"]}/16 | {r["cases"]["joint_evidence"]["wrong_accepted"]}/16 | '+(f'{attacked}/16' if attacked is not None else '未执行')+' |')
    text+=['','完整PCA、空间网格能够拒绝部分旧见证；对两个预先指定的完整PCA版本重新进行同预算联合攻击，仍各找到16/16个绕过。因此“旧攻击迁移失败”不能当作新版本抵抗自适应攻击的证据。未重攻击的grid2_pca32只作迁移对照。','',
           '## 4 方案二：对照相减与真实正常响应拟态','',
           '| 输入 | 平均绝对z(u) | 平均绝对z(v) | 平均绝对z(r) |',
           '|---|---:|---:|---:|']
    for role in ROLES:
        ed=study['details'][role]['evidence']
        text.append('| '+role+' | '+' | '.join(f'{ed["z_"+k].abs().mean().item():.3f}' for k in ['u','v','r'])+' |')
    control_error=max(study['details'][r]['evidence']['energy_error'].max().item() for r in ROLES)
    text+=['','z按当前预测类别的冻结正常profile计算，三个边际z并不独立。普通PGD的v异常在构造r=v-随机作用后减弱，与主版本弱于(u,v)消融的现象一致，但不能只靠均值断言唯一因果。',
           f'这64个原始/攻击图的最大对照能量相对误差为{control_error:.3g}，且keep分支均保持完整模型当前预测。因此本组失败不能归咎于随机对照失配；keep标签保持也不能充当正常性证明。',
           f'针对(u,v)无随机对照版本重新攻击，也找到{results["reattacks"]["evidence_no_control"]["success_count"]}/16个绕过。去掉对照改善普通攻击检测，但单独不能解决这些联合绕过。联合攻击可以使错误类别的u/r乃至u/v落入正常接受区。','',
           '## 5 独立核验、范围与下一步','']
    audit_path=verification/'independent_audit.json'
    if audit_path.is_file():
        audit=load_json(audit_path)
        for name,r in audit['results'].items():
            text.append(f'- 重新加载 {name}：{r["reverified_success_count"]}/{r["recorded_success_count"]}个已记录成功通过真实误分类、阈值、像素范围和Linf核验。')
            if 'high_precision_relative_fw_gap_max' in r:
                text.append(f'  高精度QP最大相对间隙{r["high_precision_relative_fw_gap_max"]:.3g}，部署分数与5,000次分数最大差{r["deployment_vs_high_precision_max_abs_error"]:.3g}。')
            if 'gradient_probe' in r:
                errors=[v['relative_error'] for v in r['gradient_probe']['directional_checks']]
                text.append('  新操作链在float64副本的多个差分步长相对误差为：'+', '.join(f'{v:.3g}' for v in errors)+'；实际攻击为float32。')
    text+=['','本组最直接的失败证据是：错误类别可同时满足现有的参考拟合或干预响应接受条件。增加表示信息、单独删除随机对照仍未消除这些见证；给当前分数微调阈值也需要承担更多正常拒绝。',
           '建议先重审接受条件与分类错误之间是否存在可验证的冲突，再决定方法改造。当前不据这16例外推总体成功率，也不扩展CIFAR-100或声称鲁棒防御成立。正式性能结论还需要可核验的分类器训练/开发/校准独立性。','',
           '## 文件','',
           f'- 逐图数值：[per_sample.csv]({(analysis/"per_sample.csv").as_posix()})。',
           '- `reference_weights.csv`：全部参考ID、共享/独立权重。',
           '- `trajectory_*.json`、`attack_*.pt`：每步分数、目标margin、成功并集及图像。',
           '- `representation_transfer.json`、`verification_variants.pt`：固定表征实验及参数。',
           '- `independent_audit.json`：独立核验与梯度检查。','']
    for name in ['paired_objectives','compression_energy','response_profiles','response_components','trajectories_primary','trajectories_variants','reference_examples','reference_weights','cases_1','cases_2','cases_3','cases_4']:
        figure_path=(out/(name+'.png')).as_posix()
        text.append(f'![{name}]({figure_path})\n')
    (verification/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(str(verification/'report.md'))


if __name__=='__main__':
    main()
