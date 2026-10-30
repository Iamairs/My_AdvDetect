# 脚本入口

所有Python入口从文件位置识别项目根目录，不依赖启动时工作目录。参数、输入像素尺度和原实验协议保持一致。下表路径均从项目根目录起算。

| 用途 | 入口目录 | 主要脚本 |
|---|---|---|
| 资源导入、离线分数、批量实验 | `scripts/common/` | `import_resources.py`、`analyze_scores.py`、`run_sweep.py` |
| LR Innovation | `scripts/methods/lr_innovation/` | `run_lr_innovation.py` |
| 两方法共享P0/P1协议 | `scripts/studies/mechanisms/` | `run_mechanisms.py`、`verify_mechanism_witnesses.py`、`refine_mechanism_baselines.py`、`report_mechanisms.py` |
| 16例绕过诊断 | `scripts/studies/bypass_diagnosis/` | `analyze_mechanism_bypasses.py`、`audit_bypass_verification.py`、`report_bypass_diagnosis.py` |
| 层/共享/空间消融 | `scripts/studies/layer_contribution/` | `run_layer_contribution.py`、`audit_layer_contribution.py`、`repair_layer_witnesses.py`、`report_layer_contribution.py` |
| 位移与错误类重构 | `scripts/studies/witness_geometry/` | `analyze_witness_geometry.py`、`audit_witness_geometry.py`、`report_witness_geometry.py` |
| LR分类器训练 | `scripts/training/lr/` | `train_cifar100_resnet50.py` |
| JAD分类器训练 | `scripts/training/jad/` | `train_resnet50_jad.py` |

稳定LR、MSP、JAD检测实验使用根目录的 `run.py`，不用为每个配置复制入口。

```powershell
python run.py run --config configs/experiments/baselines/lr/cifar10_lr_pgd.yaml
python scripts/methods/lr_innovation/run_lr_innovation.py --help
python scripts/studies/mechanisms/run_mechanisms.py --help
python scripts/common/run_sweep.py --sweep configs/sweeps/lr/lr_exploration.yaml
```

跨脚本导入使用 `scripts.studies.<研究问题>.<模块>`。算法实现仍在 `src/advdetect/` 或 `research/rXXX_*`，入口只负责组织流程。

本机JAD监控脚本归入 `scripts/training/jad/local/` 并继续忽略，不随源码上传；不因目录验证启动训练或监控。
