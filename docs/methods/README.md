# 方法文档索引

| 方法 | 文档 | 参数 | 实现 |
|---|---|---|---|
| LR基线 | [论文与适配边界](lr/lr_reproduction.md)、[实验记录](lr/实验记录.md) | `configs/detectors/baselines/lr/` | `src/advdetect/detectors/baselines/lr.py` |
| MSP基线 | 最大softmax置信度，稳定入口`run.py` | `configs/detectors/baselines/msp/` | `src/advdetect/detectors/baselines/msp.py` |
| JAD迁移 | [实验记录](jad/实验记录.md) | `configs/detectors/proposed/jad/` | `src/advdetect/detectors/proposed/jad.py` |
| LR Innovation | [实验记录](lr_innovation/实验记录.md) | `configs/detectors/proposed/lr_innovation/` | `research/r001_lr_innovation/` |
| 跨层共同参照 | [设计](shared_reference/方案一_跨层共同参照约束.md) | `configs/detectors/proposed/shared_reference/` | `research/r002_shared_reference/` |
| 类别证据对照干预 | [设计](evidence_intervention/方案二_类别证据对照干预.md) | `configs/detectors/proposed/evidence_intervention/` | `research/r003_evidence_intervention/` |

两方法联合P0/P1、后续消融和诊断见[研究实验索引](../studies/README.md)。方法文档不与跨方法研究报告混放。
