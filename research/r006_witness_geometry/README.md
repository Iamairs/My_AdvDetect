# 成功见证的特征与参考重构几何

只读分析R005已严格审计的修复公共池；不改原模型、参考、阈值或候选，不生成新攻击。配置均相对项目根目录解析。

```powershell
conda activate cv_env
python scripts/studies/witness_geometry/analyze_witness_geometry.py --config configs/experiments/studies/witness_geometry/cifar10_witness_geometry.yaml
python scripts/studies/witness_geometry/audit_witness_geometry.py --run-dir outputs/witness_geometry/20261029_222341_witness-geometry_ffbbfb
python scripts/studies/witness_geometry/report_witness_geometry.py --run-dir outputs/witness_geometry/20261029_222341_witness-geometry_ffbbfb
```

第一条会新建实验目录；后两条展示本次既有结果，复跑时将目录换成新运行目录。

## 距离定义

z为原检测器的池化、PCA、标准化特征；每版本使用实际部署的共享或独立QP最优权重。保存原图到攻击图的L2位移、攻击图到错误类组合的L2残差、各层每维MSE及权重正则。对共享版本不能把某层残差误称为该层独立优化的最小残差。

正常尺度来自原2,000张dev_threshold图，不与64张原图或参考图交集。按类计算：到同类64张参考的最近距离、正常不同图对距离、同版本同层的重构误差分布。每类5,000图对允许重复采样，不能当作独立图片数。

反事实保持原图不动，只使用最终错误类别参考库重新优化。低于阈值仅表示错误类别的检测接受条件已满足；原图分类仍正确。还保存原图到攻击图所用同一组合的距离，核验三角不等式，并独立重建分数分解与目标类经验百分位。

## 范围和解释

38版本各64个最低分成功候选，是相关的事后分析；不是最小特征位移的见证，不证明攻击路线或必需距离。小位移规则固定为低于真实类正常近邻距离q05，高残差规则固定为高于错误类正常重构残差q95，标志允许重叠。连续量全部保留。

报告的“仅收紧阈值”曲线复用正常阈值数据和固定见证，未重新攻击、未改变部署阈值，仅用于描述分数分布重叠。原分类器训练索引未知，不能作为最终独立评估。

主结果见 `outputs/witness_geometry/20261029_222341_witness-geometry_ffbbfb/report.md`，文字解读见 `docs/studies/witness_geometry/成功见证_特征位移与错误类重构.md`。
