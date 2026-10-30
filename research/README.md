# 研究方向索引

每个目录对应一个研究问题，不对应一次改参数。目录建议 `r001_简短英文名称`。

| 编号 | 问题 | 状态 | 结果位置 |
|---|---|---|---|
| r000_template | 新方向的记录与代码模板 | 模板，尚未实验 | 无 |
| r001_lr_innovation | LR残差条件建模 | 独立探索实现 | `docs/methods/lr_innovation/` |
| r002_shared_reference | 跨层共同参照约束 | CIFAR-10机制验证 | `docs/methods/shared_reference/`、`docs/studies/mechanisms/` |
| r003_evidence_intervention | 类别证据对照干预 | CIFAR-10机制验证 | `docs/methods/evidence_intervention/`、`docs/studies/mechanisms/` |
| r004_bypass_diagnosis | 16例联合绕过的逐图诊断 | 开发集探索 | `docs/studies/bypass_diagnosis/` |
| r005_layer_contribution | 中间层、共享权重与空间消融 | 38版本开发集验证 | `docs/studies/layer_contribution/` |
| r006_witness_geometry | 位移与错误类参考重构 | 既有见证几何诊断 | `docs/studies/witness_geometry/` |

通过导入 `advdetect` 复用数据、模型、攻击和评价；不复制公共代码。
参数变化写在配置中，机制变化才拆方向。保留失败结果与必要对照。
稳定后将算法移到 `src/advdetect/detectors/proposed/`，在工厂中增加显式入口。

`mechanisms/`是两方法及后续研究共用的数据划分、骨干视图和联合攻击协议。其Python位置及rXXX算法位置保持稳定，因为已有检查点/审计记录绑定这些文件的哈希。方法参数放在各自`configs/detectors/proposed/<方法>/`；组合实验入口位于`scripts/studies/mechanisms/`。研究数值与绘图依赖见`requirements/research.txt`。
