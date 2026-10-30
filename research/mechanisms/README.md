# 两个方案的独立机制验证

本入口只调用现有的分类器加载、数据索引、标准攻击和指标工具，不修改 `src/advdetect/`、原始配置、原检测器注册、已有权重或 LR Innovation。研究实现分别位于 `research/r002_shared_reference/` 和 `research/r003_evidence_intervention/`。

## 范围和数据

首轮为 **CIFAR-10/CIFAR ResNet18、探索性 P0/P1**。原分类器来源权重缺少训练索引，不能证明开发图没有参与其训练，因此结果不能称为严格独立数据上的最终检测效果。检测器自身的数据用途严格分开。

固定 seed=42：官方训练集拆成 model=40,000、dev=5,000、保留 calibration=5,000。model 内 gate=32,000 与 profile=8,000 不重叠；reference=8,000 与 fit_diagnostic=8,000 不重叠，跨方案允许使用相同正常训练图。dev 再拆 threshold=2,000、evaluate=1,000、select=2,000，均为分层固定索引。`final_calibration_reserved` 仅记录索引，不构造可运行数据集；不载入官方测试样本用于训练或评价。

P1 的 threshold 是**开发阈值**，在所有正常图上拟合，包括自然错分图；与未来最终校准阈值不同。所有模式用相同正常误报目标，但各自独立设置分数阈值。现有 LR 历史表的数据协议不同，不直接混入本轮公平比较表。

## 命令

```powershell
conda activate cv_env
python -m unittest discover -s tests -v

# 分别拟合，默认两个方法一起；不自动开始最终评价。
python scripts/studies/mechanisms/run_mechanisms.py fit --method both

# 用上一步返回的两个新 checkpoint 跑开发集实验。
python scripts/studies/mechanisms/run_mechanisms.py p1 --shared-checkpoint checkpoints/mechanisms/RUN_ID/shared.pt --evidence-checkpoint checkpoints/mechanisms/RUN_ID/evidence.pt

# 从拟合到 P1 一次运行（不会复用/覆盖既有分类器和 LR 权重）。
python scripts/studies/mechanisms/run_mechanisms.py run

# 单独处理某个方案。
python scripts/studies/mechanisms/run_mechanisms.py fit --method shared
python scripts/studies/mechanisms/run_mechanisms.py p1 --method shared --shared-checkpoint checkpoints/mechanisms/RUN_ID/shared.pt

# 离线生成表格和图，不重新训练/攻击。
python scripts/studies/mechanisms/report_mechanisms.py --run-dir outputs/mechanisms/RUN_ID

# 重新加载检测器，对保存的两个主方法绕过见证作独立核验。
python scripts/studies/mechanisms/verify_mechanism_witnesses.py --run-dir outputs/mechanisms/RUN_ID

# 独立/单层基线较病态时，用相同PGD缓存作高精度复核，再刷新报告。
python scripts/studies/mechanisms/refine_mechanism_baselines.py --run-dir outputs/mechanisms/RUN_ID --steps 5000
python scripts/studies/mechanisms/report_mechanisms.py --run-dir outputs/mechanisms/RUN_ID
```

所有路径由项目根解析，也支持绝对配置路径。配置为 `configs/experiments/studies/mechanisms/cifar10_mechanisms_p1.yaml`，可用 `--set key=value` 调整。运行输出位于 `outputs/mechanisms/`；检测器位于 `checkpoints/mechanisms/`。checkpoint 绑定模型哈希、数据指纹、索引、方法参数与实现文件哈希，配置改变不能悄悄复用。

## 两个实现

**方案一**：完整 `backbone.layer2/3/4` 的激活后输出；GAP（可配置空间网格）→各层在 reference 上独立 PCA→逐维标准化→每类真实样本 k-medoids 覆盖。保存跨层共用的原始参考 ID。单纯形投影使用排序算法，固定步数展开凸二次规划；Gram 缓存；记录 Frank-Wolfe 间隙。相同 eta 加到共享与独立各层的目标，并按层权重加权。主分数 shared；核心消融 independent、shuffled、random_reference、last_layer，外加 MSP/margin。G=shared-independent 为机制诊断，数值负值保留并报告，不静默截断。

首轮数值检查发现普通投影梯度200次未达到间隙容差；当前使用 **FISTA加速投影、固定1,000次**，并把各消融的独立QP沿batch维合并执行，减少循环调度成本。仍是相同凸目标与相同单纯形投影，整个固定展开保留梯度。初版200次结果只保留为数值诊断，不作为当前主实现结果。更长迭代只依赖正常开发图的数值收敛选择，没有按攻击结果挑求解器。

主方法和打乱/随机参考在正常1,000图上达到容差；独立最深层在1,000次时仍有25.2%未达标。`refine_mechanism_baselines.py` 因此对全部开发阈值、正常评价和已保存PGD作5,000次独立QP复核，保存独立的`baseline_refinement.json`与分数，不覆盖原运行历史。最终报告优先使用复核基线，主方法仍是原1,000次。三层在所有这些数据上均达到1e-5，AUROC/FPR/JASR0未变化。

**方案二**：默认完整 `backbone.layer3` 输出。总体逐通道均值只从 gate 数据拟合；sigmoid 连续门控，q=0.5、alpha=0.5、T=1；Adam lr=0.01、30 epochs、batch=128、类别均衡采样；冻结主模型及 BN。只在 dev_select 正常图上按“keep retention 距95%缺口最小，再正常训练目标最小”选 epoch。每类 kappa 为 gate 数据正 margin 中位数的0.25倍。默认 lambda_drop=1、lambda_budget=10、lambda_nonclass=0。

profile 只用 gate 外的正确正常同类图拟合。主特征为(u,r)，核心消融 keep、drop、no_control=(u,v)、random_pair。random_pair 把固定通道置乱门控作为主门控，再用另外两组固定置乱作为其匹配对照；与主方法具有相同分支预算，重新拟合自己的正常 profile 和阈值。每类每层预存3组置乱：主方法取前两组为对照；随机主方法取第一组为主门控，后两组为对照。低样本类回退到对角方差，缺少任何正确 profile 样本会报错而不是自动拒绝该类。

每个分支从未经修改的同一个 stage 输出开始。随机对照强度随输入变化，并保留梯度；记录强度截断造成的能量匹配误差和无效对照。所有消融一起评价比只运行主方法更贵，性能表须区分这一点。

## 验证与攻击

P0 人工检查包括 SciPy SLSQP 独立 QP 核验、梯度检查、共享/独立约束顺序、共同置换不变性、完整 stage 续跑、全开 keep、alpha=0、分支无污染、BN不变、能量缩放梯度和零能量处理。真实网络另保存方向有限差分结果：初版float32差分存在明显数值误差，当前在独立float64副本上验证同一操作链，部署/攻击仍为float32。1,000 张正常开发图比较50/100/200/500/1000/2000迭代的收敛间隙与原始G，部署使用固定1,000次；未收敛需公开报告，不能当成防御收益。

P1 使用同一批1,000开发图及一次生成的标准库 PGD（8/255、20步、step=2/255）评估所有方案和消融，按原图配对 bootstrap；保存 margin 分箱诊断。普通攻击结果只回答非联合条件下的机制信号。

联合 PGD 接口 `research/mechanisms/adaptive.py:joint_pgd` 支持目标枚举、lambda扫描、多重启、固定目标类下的完整可微分数；每个迭代点用真实预测类与实际部署分数核验，保留所有已发现成功及见证，包含原图。`observer` 逐点对其他方法评分，构造公共候选流成功并集。

P1 默认只在16图上做20步、2目标、1重启、lambda=1的实现探针；两主方法各生成候选，所有消融在同一候选流上评分。**这不是充分自适应攻击评估**，没有执行完整目标枚举、步数/重启收敛、BPDA/SPSA、多攻击族或第二骨干。`adaptive_full` 只预留后续100步、9目标、5重启、5权重配置，P1不会自动执行。

本轮不自动扫描大网格、不进行额外攻击训练、不访问最终校准/测试、不运行 CIFAR-100。方案一16/64维、空间网格、不同参考规模，以及方案二多层、多q/alpha、非同类惩罚可配置，后续按机制结果决定是否开展。

首轮只验证了`lambda_nonclass=0`；非零分支当前使用固定相邻负类近似，并非全非同类平均惩罚，应在扩展后再验证。`fit_diagnostic`的8,000个索引仅作保留，本轮正常数值诊断使用开发集1,000图，未额外遍历该保留子集。
