# 层贡献、共享约束与空间信息的独立消融

配置：`configs/experiments/studies/layer_contribution/cifar10_layer_contribution.yaml`。本模块不修改原始分类器、LR、两个主方法、旧配置或旧产物。

## 要回答的问题

1. 只用layer4、增加layer2、增加layer3、增加两层，是否改变攻击检测表现？
2. 同样的表示/参考/层权重下，独立权重与共享权重是否有差别？
3. GAP与保留2×2位置、PCA32与完整PCA分别产生什么影响？

“空间平均消掉大量变化”只描述变化能量，不能直接判断剩余特征是否能区分正常/攻击。本实验按实际误报与错误接受比较，而非按变化能量大小下结论。

## 设计

- 四种表示：GAP/PCA32、2×2/PCA32、GAP/完整PCA、2×2/完整PCA。沿用上轮同一D_ref拟合的变换与每类64个参考ID，不重新看攻击选特征。
- 六个层集：2、3、4、2+4、3+4、2+3+4；多层同时评价独立与共享权重。单层二者相同，不重复。
- 对GAP/PCA32和2×2/PCA32额外增加layer4自选64个medoid的基线，选择只用layer4的正常参考特征，预算与算法相同。
- 共38个版本，对应26个不重复QP。各层误差除以同层维数、层之间等权平均，eta=0.001。独立版本是同样正则目标的逐层最小值加权，保持正则总强度一致。
- 全部QP固定5,000次FISTA，保存相对Frank-Wolfe间隙。把共享的单层问题复用并沿batch维合并，避免同一独立层重复求解。
- 相同2,000正常开发图设5%目标阈值，相同1,000正常开发图与同一标准PGD缓存评价。正常图不按正确分类过滤。
- 三个主比较在配置中预先指定，按原图配对bootstrap，Bonferroni控制三个比较的总体alpha=0.05。其他留一层、单层自选参考等比较为探索性区间，不能从多次筛选中挑最好列当最终结论。

## 联合攻击与公共池

从原1,000开发图中排除此前16例，按固定seed分层抽取64图。选择仅依赖标签与seed，不依赖哪种版本更容易被攻击。原分类正确集合固定，用它作所有版本JASR0分母。

每个版本在同一逐原图候选池上评价。候选含原图、普通PGD、每个生成器产生的所有迭代点；不是只保存最后一步或只对某个检测器选有利样本。一次真实误分类、分数不超阈值、像素/范数合法就保留成功，之后失败不撤销。

联合攻击最多40步、3个近竞争目标、1重启、lambda=[1,0.1,10]，分数按对应正常开发std缩放；第一条用普通PGD暖启动，之后可复现随机初始化。输入到PCA/标准化/完整5,000次QP的梯度不断开。

单个批次所有图均已被目标版本真实绕过时提前停止；某版本已被当前公共池全部绕过时，不再重复为它生成攻击。这只在该组成功率已达上限时节约计算，所有实际产生候选仍交叉评价全部版本。`attack_events.json`记录执行/跳过原因、原先未破数量和实际梯度步数，不能用这些执行次数直接比较各版本攻击难度。

40步、3目标不是穷尽攻击。若仍未找到某个绕过，只能报告当前攻击集合未找到，不能据此称安全。若某版本64/64已有有效见证，也不能将这64图的比例直接外推整个数据分布。

## 运行

```powershell
conda activate cv_env
python -m unittest discover -s tests -v
python scripts/studies/layer_contribution/run_layer_contribution.py
# 用上一步打印的目录替换 RUN_ID。
python scripts/studies/layer_contribution/audit_layer_contribution.py --run-dir outputs/layer_contribution/RUN_ID
python scripts/studies/layer_contribution/report_layer_contribution.py --run-dir outputs/layer_contribution/RUN_ID
```

若严格重载审计在阈值附近发现见证不一致，保留失败记录，先诊断再修复，不能忽略：

```powershell
python scripts/studies/layer_contribution/audit_layer_contribution.py --run-dir outputs/layer_contribution/RUN_ID --diagnostic-only
python scripts/studies/layer_contribution/repair_layer_witnesses.py --run-dir outputs/layer_contribution/RUN_ID
python scripts/studies/layer_contribution/audit_layer_contribution.py --run-dir outputs/layer_contribution/RUN_ID
python scripts/studies/layer_contribution/report_layer_contribution.py --run-dir outputs/layer_contribution/RUN_ID
```

修复只寻找明确落在接受区内部的攻击，优化目标要求低于原阈值至少正常std的5%；原防御阈值和分数不变。保存`common_pool_repaired.pt`、`witness_repair.json`，不覆盖初始公共池。审计默认优先核验修复后的见证，所有新增候选仍交叉评价整个方法族。

原方法的检查点加载仍采用已有严格模型/数据/源码绑定；新family状态单独保存。审计额外检查借用表征的模型/数据、固定与自选参考身份、64图排除旧16、全量普通PGD预算、38版本成功见证和新增梯度路径。

输出包含总表、逐图分数、同一原图的“多拦住/又放过”分解、类别分解、主/探索对比区间、所有迭代候选及最终成功并集。原分类器训练索引未知，本轮仍是开发集探索，不使用最终校准或官方测试样本。
