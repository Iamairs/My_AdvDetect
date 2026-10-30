# Layer Regression 复现说明

## 来源

- 论文：`Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers`，TMLR 2025。
- 官方实现：<https://github.com/furkanmumcu/Layer-Regression>。
- 标准攻击参考：<https://github.com/Iamairs/CCDE-ID_AdvDetect>，当前只接入 `torchattacks` 的标准 PGD 和 AutoAttack。

## 方法对应关系

| 论文/官方实现 | 当前工程 |
|---|---|
| 只用干净训练样本 | `LayerRegressionDetector.fit` 使用 `fit` split，不生成对抗样本 |
| 只保留分类正确样本 | `detector.correct_only: true`，按 batch 过滤 |
| 3 个中间层 | `LRAutomator.generate_config(num_layers=3)`，候选范围为全部加权叶模块的 1/5 到 4/5 |
| 每层取中间 60% | 每层 flatten 后使用 `[20%, 80%)` 切片 |
| 两层隐藏层 MLP，宽度 2048 | `LayerRegressionMLP(input_dim, 2048, 2048, 512)` |
| Adam，学习率 `3e-4` | `detector.lr: 0.0003` |
| 目标为倒数第二层特征 | CIFAR ResNet18 的 `pool(layer4)`，维度 512 |
| MSE 作为异常分数 | `score = mean((m(v) - target)^2)`，分数越大越异常 |

## CIFAR-10/ResNet18 适配边界

论文主实验使用 ImageNet/CIFAR-100 和多种 `timm` 模型；本工程使用已导入的 CIFAR-10、CIFAR ResNet18 和 32x32 输入。因此当前结果是**论文方法的可追溯适配与流程复现**，不是论文表格的逐项重现。层选择由官方自动选择器在当前 ResNet18 模块树上执行，检测器输出维度由 512 适配。

论文评价只把分类器正确的 clean 样本和成功误分类的 adversarial 样本纳入主协议。当前默认 `evaluation.calibrate_on_correct_only: true`，阈值在校准集的 clean-correct 样本上拟合；评价同时保留 all-generated 指标和 paper-protocol 指标。

## CIFAR-100/timm ResNet50 适配

`configs/experiments/baselines/lr/cifar100_lr_pgd.yaml` 与 `cifar100_lr_autoattack.yaml` 是 ResNet50/CIFAR-100 实验配置。目标分类器采用 `timm` 的 `resnet50`、100 类头和 ImageNet 预训练参数微调；CIFAR 图像缩放至 224×224，并使用 ImageNet 均值、标准差。分类器训练集、LR fit、selection、calibration 分别对应 CIFAR-100 训练集中的 40,000、8,000、2,000 张，最终评价使用完整 10,000 张测试集。PGD 和 AutoAttack 共用该划分、分类器、LR 检查点及阈值。

论文附录 F 指定 `conv2` 的第 5、8、13 层及三组切片。本环境的 `timm 1.0.30` ResNet50 实际有 16 个名含 `conv2` 的模块，而论文写 15 个；按注册顺序取 5、8、13 得到 `layer2.1.conv2`、`layer3.0.conv2`、`layer3.5.conv2`，和论文层序号对应。分类器预训练来源、微调轮数和优化器细节在论文中没有完整说明；本项目将明确记录为 ImageNet 预训练初始化、20 轮 CIFAR-100 微调（AdamW，学习率 `1e-4`），属于透明的复现适配设置，不能据此宣称论文中未披露的训练细节已被逐字复现。

运行前先训练目标分类器：

```powershell
python scripts/training/lr/train_cifar100_resnet50.py
python run.py run --config configs/experiments/baselines/lr/cifar100_lr_pgd.yaml
python run.py run --config configs/experiments/baselines/lr/cifar100_lr_autoattack.yaml
```

PGD 使用 Linf `8/255`、20 步；AutoAttack 使用 standard 四组件。两个配置都使用 batch size 16。AutoAttack 会在完整 10,000 张图像上运行，预计明显慢于 PGD。

## 攻击

```powershell
conda activate cv_env
python run.py run --config configs/experiments/baselines/lr/cifar10_lr_pgd.yaml
python run.py run --config configs/experiments/baselines/lr/cifar10_lr_autoattack.yaml
```

两种攻击都使用 `[0, 1]` 输入尺度、`L_inf` 预算 `8/255`。PGD 的默认步数为 20；AutoAttack 使用标准四组件集合。攻击生成在 `torch.enable_grad()` 中显式执行，检测器普通评分仍可保留输入梯度，避免自适应攻击被无意截断。
