# 现有16组绕过的诊断与验证

这是独立研究分析，不修改 `r002/r003`、既有 `src/advdetect/`、分类器、LR或此前实验。输入为既有 P1 的 `adaptive_probe_witnesses.pt` 和 `common_pgd.pt`；严格核对样本ID与模型/数据/源码绑定。

## 运行

在项目根目录、`cv_env`环境执行：

```powershell
python -m unittest discover -s tests -v
python scripts/studies/bypass_diagnosis/analyze_mechanism_bypasses.py analyze
# 将上一步打印的目录填入 --analysis-dir。
python scripts/studies/bypass_diagnosis/analyze_mechanism_bypasses.py verify --analysis-dir outputs/bypass_diagnosis/ANALYSIS_RUN
python scripts/studies/bypass_diagnosis/audit_bypass_verification.py --analysis-dir outputs/bypass_diagnosis/ANALYSIS_RUN --verification-dir outputs/bypass_diagnosis/VERIFY_RUN
python scripts/studies/bypass_diagnosis/report_bypass_diagnosis.py --analysis-dir outputs/bypass_diagnosis/ANALYSIS_RUN --verification-dir outputs/bypass_diagnosis/VERIFY_RUN
```

配置为 `configs/experiments/studies/bypass_diagnosis/cifar10_bypass_diagnosis.yaml`，路径相对项目根解析。可以通过`--set diagnosis.source_run=...`选择兼容源运行。分析/验证分阶段保存，后续审计和绘图只读取产物，不重新运行攻击。

## 固定设计

1. 对齐同一16个原图，读取普通PGD、共享方法公共池见证、干预方法公共池见证。分别保留来源，不把全部公共池见证称为对应攻击新生成。
2. 共享分数、独立分数、G、单层残差、真实参考ID和全部权重用5,000次QP诊断；另保存实际1,000次部署分数。
3. 以同一错误类别比较原图与绕过，避免将不同类别的参考列或权重直接比较。有效参考数量使用`1/sum(w²)`，不宣传L1稀疏性。
4. GAP变化能量按空间常数子空间投影计算，包含H×W因子；PCA能量在标准化前计算。四组随机扰动匹配Linf，不同时匹配L2，因此只作方向与预算对照。
5. 统计对照使用原1,000正常开发图中排除16例后的984张。干预z和Mahalanobis仍使用原8,000张profile拟合的冻结统计，不用984张重新训练检测器。
6. 重放两个原方法的20步、2目标、1重启联合PGD，记录全部迭代图像、分数、目标margin和累计真实成功；按逐字节哈希确认旧见证来源。
7. 预先固定2×2表征实验：GAP/2×2网格 × PCA32/完整PCA。完整PCA保留全部维数并继续白化；它与PCA32使用嵌套基底，没有把更换坐标系混入维数实验。同一每类64个参考ID、同一正常参考图和eta，每个版本重新在相同2,000正常开发图上设5%目标阈值。
8. 预先指定重新攻击GAP完整PCA、2×2完整PCA与无对照(u,v)干预版本，不根据迁移结果挑有利版本。`grid2_pca32`仅是迁移对照，未为它单独开发攻击。
9. 独立进程重新加载变体，核验有效见证、QP高精度一致性及输入梯度。梯度差分在float64副本上检查，实际攻击为float32。

## 当前结果

分析目录为 `outputs/bypass_diagnosis/20261029_143300_bypass-analyze_49c8f2/`；验证目录为 `outputs/bypass_diagnosis/20261029_143813_bypass-verify_782d18/`。

报告含全部16例图、响应与轨迹图、参考图和权重。核心结论与复跑路径见 `docs/studies/bypass_diagnosis/16组联合绕过_诊断与实验验证.md`。

这些是固定的已知绕过案例，不是总体鲁棒性评价；原分类器训练索引未知。本轮没有访问最终校准/官方测试样本，也没有重训分类器或扩展CIFAR-100。
