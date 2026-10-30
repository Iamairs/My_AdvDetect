# 配置组织

| 配置 | 分组原则 |
|---|---|
| `experiments/baselines/lr/`、`experiments/baselines/msp/` | 对比方法的完整实验组合 |
| `experiments/proposed/jad/`、`experiments/proposed/lr_innovation/` | 各方法的完整实验组合 |
| `experiments/studies/<研究问题>/` | 两方法P0/P1、绕过诊断、层贡献、见证几何 |
| `detectors/baselines/<方法>/`、`detectors/proposed/<方法>/` | 方法自身参数 |
| `attacks/pgd/`、`attacks/autoattack/` | 公共攻击参数 |
| `models/cifar10/`、`models/cifar100/` | 公共分类器定义和权重位置 |
| `datasets/` | 公共数据集与切分规则 |
| `sweeps/lr/` | LR实验参数扫描 |

`shared_reference`和`evidence_intervention`的参数已从两方法组合配置拆出，组合实验通过继承使用；展开后的参数与迁移前一致。

`extends`相对当前YAML解析；数据、权重、输入输出路径相对项目根目录解析。移动YAML时必须一起检查继承路径。`--set section.key=value`继续支持覆盖参数。

`legacy_paths.yaml`只保存旧地址到新地址的映射，**没有复制配置内容**。旧地址对应文件不存在时，`load_config`会发出迁移提示并读取新地址；未知地址仍报错。旧运行目录里已保存的完整配置可直接读取。

数据、权重和运行产物不属于配置，也不放入这些目录。
