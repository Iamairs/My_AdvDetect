# AdvDetect Research Workspace

这是用于毕业论文对抗检测研究的可迁移实验工程。公共代码位于 `src/advdetect/`，未定型方法放在 `research/`，实验参数通过 `configs/` 中的 YAML 管理。

## 环境

推荐使用已经配置好的 `cv_env` 环境。当前已验证的运行环境为：Python 3.12、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、torchattacks 3.5.1、timm 1.0.30，GPU 为 NVIDIA RTX 3050。

在 Windows PowerShell 中先激活该环境，再运行项目命令：

```powershell
conda activate cv_env
python -m unittest discover -s tests -v
python run.py check-resources --config configs/experiments/cifar10_msp_smoke.yaml
```

## 已导入资源

`data/cifar-10-batches-py/` 保存 CIFAR-10 二进制数据，`checkpoints/classifiers/cifar10_resnet18.pt` 是从 `CCDE-ID_AdvDetect` 导入并经哈希校验的 CIFAR ResNet18 权重。资源来源和校验值记录在 `docs/resource_manifest.json`。

## 实验入口

```powershell
# MSP 基线 + PGD 冒烟流程
python run.py run --config configs/experiments/cifar10_msp_smoke.yaml

# LR 基线 + PGD 冒烟流程
python run.py run --config configs/experiments/cifar10_lr_smoke.yaml

# LR 基线 + 标准 AutoAttack 冒烟流程
python run.py run --config configs/experiments/cifar10_lr_autoattack.yaml --set limits.fit=128 --set limits.calibration=128 --set attack.verbose=false

# 汇总已经完成的评价结果
python run.py summarize --output-root outputs

# CIFAR-100/timm ResNet50：先微调目标分类器，再分别评估标准 PGD 与 AutoAttack
python scripts/train_cifar100_resnet50.py
python run.py run --config configs/experiments/cifar100_lr_pgd.yaml
python run.py run --config configs/experiments/cifar100_lr_autoattack.yaml
```

运行目录会保存实际配置、资源绑定、逐样本分数和指标。数据、权重、缓存和运行输出不应提交到版本库。冒烟配置只用于检查流程和接口，不能作为论文最终结果。

LR 的论文、官方实现和本项目适配边界见 `docs/methods/lr_reproduction.md`。
