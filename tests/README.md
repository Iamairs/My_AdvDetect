# 测试组织

- `core/`：公共数据切分、配置继承/旧路径兼容、攻击包装、分数与评价协议。
- `methods/lr/`：LR官方特征对齐、检查点往返、输入梯度和ResNet50层映射。
- `methods/lr_innovation/`：独立创新方法的分数、白化及状态恢复。
- `studies/mechanisms/`：两方法和共用协议的数值/梯度检查。
- `studies/bypass_diagnosis/`：诊断尺度、图像身份和空间/PCA分解。
- `studies/layer_contribution/`：共享/独立QP一致性、公共候选池及联合攻击。
- `studies/witness_geometry/`：显式重构、按类参考距离、正常百分位。

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

只运行某类测试，例如：

```powershell
python -m unittest discover -s tests/studies/layer_contribution -v
```

各子目录包含`__init__.py`以支持递归发现。目录迁移没有改变原30项测试的断言；新增3项配置/入口回归检查。核心测试中的CIFAR切分与LR检查需要本机已有数据和依赖。
