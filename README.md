# Neuromaps 梯度图空间相关分析

本项目基于Neuromaps二次开发，将 MNI152 空间的梯度统计 t 图与 78 张参考脑图谱统一到 fsLR 32k 表面，按同一套脑区模板提取 ROI 值，再计算空间相关。脚本通过空间零模型估计置换 p 值，并对有效结果进行 Benjamini–Hochberg FDR 校正。

## 环境

安装 Python 依赖和 Connectome Workbench。Workbench 安装后，将脚本开头的 `WORKBENCH` 改为本机包含 `wb_command`（Windows 下为 `wb_command.exe`）的目录。

```powershell
python -m pip install neuromaps statsmodels brainsmash
```

如果 `workspace` 位于已有的 neuromaps 源码目录中，可以从 `workspace` 运行 `python -m pip install -e ..`，让脚本使用上一级目录的 neuromaps 源码。

## 输入文件

从仓库根目录运行。按下表准备本地文件，或修改脚本里的 `GRADIENT`、`ATLAS` 和 `MANIFEST` 路径。

| 文件 | 用途 |
| --- | --- |
| `data/G1_VCI_minus_NC_t.nii` | MNI152 空间的输入梯度 t 图 |
| `data/surface_atlases/fsLR/atlas_hemi-L.label.gii` | fsLR 32k 左半球脑区标签 |
| `data/surface_atlases/fsLR/atlas_hemi-R.label.gii` | fsLR 32k 右半球脑区标签 |
| `selected_78_maps.csv` | 参考图谱索引，包含 `map_id`、`space`、`den` 和 `paths` |

按索引的 `paths` 准备参考图谱。索引中的相对路径以脚本所在目录为基准，现有索引指向 `neuromaps-data/annotations/`。表面图谱的左右半球路径按左、右顺序填写，用 `|` 分隔。影像输入、下载数据及运行结果由 `.gitignore` 排除，需要在本地单独准备。

## 参数与运行

| 参数 | 当前默认值 | 可选值或用途 |
| --- | --- | --- |
| `CORR` | `spearmanr` | `spearmanr` 或 `pearsonr` |
| `NULL_METHOD` | `spin` | `spin` 为 Alexander–Bloch 旋转置换，`burt2020` 为 BrainSMASH |
| `N_PERM` | `10` | 空间置换次数。默认值适合快速试运行，正式分析时按研究方案调整 |

使用 `burt2020` 时，脚本会为左右半球分别计算测地距离，并运行变异函数拟合评估。左右半球的非零 ROI 标签按左、右顺序拼接后须从 1 连续编号。可在脚本中调整 `BRAINSMASH_PV`、`BRAINSMASH_NH` 和 `BRAINSMASH_DELTAS`。

```powershell
python official_teaching_meuromaps.py
```

## 输出

结果写入 `outputs/official_teaching/<零模型>_<相关方法>/`。默认目录为 `outputs/official_teaching/spin_spearmanr/`。

- `results.csv` 按 `q_fdr` 升序排列，记录图谱 ID、相关系数 `r`、空间置换 p 值 `p_spatial`、校正后的 `q_fdr` 和 `significant`（FDR 阈值为 0.05）。
- `skipped.csv` 记录在转换、ROI 提取或相关分析阶段被跳过的图谱及原因。
