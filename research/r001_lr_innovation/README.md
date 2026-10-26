# 条件标准化 LR 创新量

这是独立于原始 LR 的探索实现。它不修改 `src/advdetect/detectors/baselines/lr.py`，也不注册到原有 detector factory。

当前实现包括：

- 5 折 out-of-fold clean residual 统计；
- 原始 MSE、去均值 MSE、对角白化、类别与 margin 条件白化；
- top-2 分类决策方向的双侧和单侧分数；
- 标准 PGD 与 AutoAttack 的独立评价入口。

先执行 `fit` 生成独立 checkpoint，再执行 `evaluate`。本目录只保存研究代码和说明，不保存数据、权重或攻击缓存。
