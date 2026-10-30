# AdvDetect Research Workspace

这是用于毕业论文对抗检测研究的可迁移实验工程。公共代码位于 `src/advdetect/`，未定型方法放在 `research/`，实验参数通过 `configs/` 中的 YAML 管理。

## 环境

推荐使用已经配置好的 `cv_env` 环境。研究实验使用 Python 3.11、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、torchattacks 3.5.1，GPU 为 NVIDIA RTX 3050。基础依赖由 `pyproject.toml` 管理；研究数值检查和绘图可安装 `requirements/research.txt`。

在 Windows PowerShell 中先激活该环境，再运行项目命令：

```powershell
conda activate cv_env
python -m unittest discover -s tests -v
python run.py check-resources --config configs/experiments/baselines/msp/cifar10_msp_smoke.yaml
```

## 已导入资源

`data/cifar-10-batches-py/` 保存 CIFAR-10 二进制数据，`checkpoints/classifiers/cifar10_resnet18.pt` 是从 `CCDE-ID_AdvDetect` 导入并经哈希校验的 CIFAR ResNet18 权重。资源来源和校验值记录在 `docs/resource_manifest.json`。

## 实验入口

```powershell
# MSP 基线 + PGD 冒烟流程
python run.py run --config configs/experiments/baselines/msp/cifar10_msp_smoke.yaml

# LR 基线 + PGD 冒烟流程
python run.py run --config configs/experiments/baselines/lr/cifar10_lr_smoke.yaml

# LR 基线 + 标准 AutoAttack 冒烟流程
python run.py run --config configs/experiments/baselines/lr/cifar10_lr_autoattack.yaml --set limits.fit=128 --set limits.calibration=128 --set attack.verbose=false

# 汇总已经完成的评价结果
python run.py summarize --output-root outputs

# CIFAR-100/timm ResNet50：先微调目标分类器，再分别评估标准 PGD 与 AutoAttack
python scripts/training/lr/train_cifar100_resnet50.py
python run.py run --config configs/experiments/baselines/lr/cifar100_lr_pgd.yaml
python run.py run --config configs/experiments/baselines/lr/cifar100_lr_autoattack.yaml
```

运行目录会保存实际配置、资源绑定、逐样本分数和指标。数据、权重、缓存和运行输出不应提交到版本库。冒烟配置只用于检查流程和接口，不能作为论文最终结果。

LR 的论文、官方实现和本项目适配边界见 `docs/methods/lr/lr_reproduction.md`。

## 按方法与研究问题找文件

方法自己的脚本、配置、文档和测试分别位于对应的子目录；跨方法消融与诊断属于 `studies`。公共数据、模型、攻击和评价仍由 `src/advdetect/` 提供。

```text
scripts/
  common/                  资源导入、通用分数分析、批量实验
  methods/lr_innovation/   独立方法入口
  studies/                 mechanisms、bypass_diagnosis、layer_contribution、witness_geometry
  training/                lr、jad 的分类器训练入口
configs/
  experiments/             baselines/<方法>、proposed/<方法>、studies/<研究问题>
  detectors/               baselines/<方法>、proposed/<方法>
  attacks/                 pgd、autoattack
  models/                  cifar10、cifar100
  datasets/                公共数据集定义
  sweeps/lr/               LR 参数扫描
docs/
  methods/<方法>/          方法说明和该方法实验记录
  studies/<研究问题>/      联合实验、消融、诊断与逐图分析
  project/                 项目维护记录与迁移表
tests/
  core/                    数据、配置、攻击、评价和目录兼容检查
  methods/<方法>/          方法专用测试
  studies/<研究问题>/      研究阶段的数值和机制测试
research/                  每个假设一个 rXXX 模块，算法实现位置保持稳定
```

- [方法文档索引](docs/methods/README.md)、[研究实验索引](docs/studies/README.md)、[研究代码索引](research/README.md)。
- [脚本入口与示例](scripts/README.md)、[配置组织与继承](configs/README.md)、[测试说明](tests/README.md)。
- [实验记录索引](实验记录.md)、[目录迁移说明](docs/project/目录重构.md)。

例如，近期三层消融使用：

```powershell
python scripts/studies/layer_contribution/run_layer_contribution.py --config configs/experiments/studies/layer_contribution/cifar10_layer_contribution.yaml
python scripts/studies/witness_geometry/analyze_witness_geometry.py --config configs/experiments/studies/witness_geometry/cifar10_witness_geometry.yaml
```

历史 YAML 地址通过 `configs/legacy_paths.yaml` 集中兼容，继续使用会提示新地址。旧脚本已迁入对应子目录，请使用索引中的新命令。已有数据、权重、缓存与运行输出保留原位置。
