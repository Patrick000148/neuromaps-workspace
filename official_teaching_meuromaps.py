from pathlib import Path
import csv
import os
import numpy as np

# 兼容旧版BrainSMASH与NumPy 2.x
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from neuromaps import transforms, nulls, stats
from neuromaps.datasets import fetch_atlas
from neuromaps.images import load_data
from neuromaps.parcellate import Parcellater
from neuromaps.points import get_surface_distance
from statsmodels.stats.multitest import multipletests
from brainsmash.mapgen.eval import base_fit

# =========================================
# 1. 用户数据配置
# =========================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
# Workbench路径
WORKBENCH = Path(r"D:\research\tool\workbench-windows64-v2.2.1\workbench\bin_windows64")
# 输入梯度t图：MNI152空间
GRADIENT = DATA / "G1_VCI_minus_NC_t.nii"
# 输入脑区模板：fsLR32k左右半球
ATLAS = (
    DATA / "surface_atlases/fsLR/atlas_hemi-L.label.gii",
    DATA / "surface_atlases/fsLR/atlas_hemi-R.label.gii"
)
# 已筛选好的78张参考图谱
MANIFEST = ROOT / "selected_78_maps.csv"

# =========================================
# 2. 用户参数配置
# =========================================
# 相关方法：pearsonr / spearmanr
CORR = "spearmanr"
# 零模型：spin / burt2020
NULL_METHOD = "spin"
# spin / burt2020的保存路径
OUTPUT = ROOT / "outputs" / "official_teaching" / f"{NULL_METHOD}_{CORR}"
# 空间置换次数
N_PERM = 10
# BrainSMASH手动调参
# =========================================
# PV：变异函数拟合距离百分位
BRAINSMASH_PV = 25
# NH：距离分箱数量
BRAINSMASH_NH = 25
# DELTAS：候选空间平滑邻域比例
BRAINSMASH_DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# =========================================
# 3. 功能函数定义
# =========================================
# 读取参考图谱函数
# =========================================
def load_annotations():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"读取参考图谱：{len(rows)}张")
    return rows
# =========================================
# 空间转换函数：统一到fsLR32k
# =========================================
def to_fslr(item):
    # 支持相对路径和绝对路径
    paths = [Path(p.strip()) for p in item["paths"].split("|") if p.strip()]
    paths = [str(p if p.is_absolute() else ROOT / p) for p in paths]
    space = item["space"]
    # MNI152体积图 → fsLR32k
    if space == "MNI152":
        return transforms.mni152_to_fslr(paths[0], "32k")
    # 表面图谱：左半球、右半球
    surfaces = tuple(paths)
    # fsLR空间
    if space == "fsLR":
        if item["den"] == "32k":
            return surfaces
        return transforms.fslr_to_fslr(surfaces, "32k")
    # fsaverage → fsLR32k
    if space == "fsaverage":
        return transforms.fsaverage_to_fslr(surfaces, "32k")
    # CIVET → fsLR32k
    if space.lower() == "civet":
        return transforms.civet_to_fslr(surfaces, "32k")
    raise ValueError(f"不支持的图谱空间：{space}")
# =========================================
#  BrainSMASH空间距离矩阵
# =========================================
def get_distance_matrices(n_roi):
    # 获取官方fsLR32k表面
    surfaces = fetch_atlas("fsLR", "32k")["midthickness"]
    # 读取脑区标签
    labels = []
    for parc in ATLAS:
        values = np.unique(np.asarray(load_data(str(parc))).ravel())
        labels.append(values[values != 0].astype(int))
    # 检查脑区标签顺序
    all_labels = np.concatenate(labels)
    if not np.array_equal(all_labels, np.arange(1, n_roi + 1)):
        raise ValueError("左右半球标签与ROI顺序不匹配")
    # 分别计算左右半球测地距离
    distances = []
    for hemi, surface, parc, hemi_labels in zip(("L", "R"), surfaces, ATLAS, labels):
        print(f"计算{hemi}半球空间距离")
        dist = np.asarray(get_surface_distance(surface, parcellation=parc, n_proc=1), dtype=float)
        count = len(hemi_labels)
        if dist.shape != (count, count) or not np.isfinite(dist).all():
            raise ValueError(f"{hemi}半球距离矩阵异常：{dist.shape}")
        distances.append(dist)
    return tuple(distances)
# =========================================
# 空间零模型
# =========================================
def generate_nulls(gradient_roi):
    # Spin和BrainSMASH公共参数
    common = dict(atlas="fsLR", density="32k", parcellation=ATLAS, n_perm=N_PERM, seed=42)
    # Spin test
    if NULL_METHOD == "spin":
        print("使用Spin空间零模型")
        null_maps = nulls.alexander_bloch(gradient_roi, **common)
    # BrainSMASH
    elif NULL_METHOD == "burt2020":
        print("使用BrainSMASH空间零模型")
        print("PV：", BRAINSMASH_PV)
        print("NH：", BRAINSMASH_NH)
        print("DELTAS：", BRAINSMASH_DELTAS)
        # 每次运行重新计算距离矩阵
        distances = get_distance_matrices(len(gradient_roi))
        # BrainSMASH官方拟合评估
        n_left = distances[0].shape[0]
        base_fit(gradient_roi[:n_left], distances[0], nsurr=100, pv=BRAINSMASH_PV, nh=BRAINSMASH_NH,
                 deltas=np.asarray(BRAINSMASH_DELTAS), kernel="exp")
        base_fit(gradient_roi[n_left:], distances[1], nsurr=100, pv=BRAINSMASH_PV, nh=BRAINSMASH_NH,
                 deltas=np.asarray(BRAINSMASH_DELTAS), kernel="exp")
        # neuromaps官方BrainSMASH接口
        null_maps = nulls.burt2020(gradient_roi, distmat=distances, pv=BRAINSMASH_PV, nh=BRAINSMASH_NH, deltas=np.asarray(BRAINSMASH_DELTAS, dtype=float), **common)
    else:
        raise ValueError("NULL_METHOD只能是spin或burt2020")
    null_maps = np.asarray(null_maps, dtype=float)
    return null_maps
# =========================================
# 保存CSV
# =========================================
def save_csv(filename, rows, columns):

    with (OUTPUT / filename).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
# =========================================
# 4.主程序入口
# =========================================
def main():
    # 配置Workbench
    os.environ["PATH"] = str(WORKBENCH) + os.pathsep + os.environ.get("PATH", "")
    # 创建结果目录
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"相关方法：{CORR} | 零模型：{NULL_METHOD} | 置换次数：{N_PERM}")
    # -------------------------------------
    # 第一步：加载脑区模板
    # -------------------------------------
    print("\n[1] 加载脑区模板")
    parcellater = Parcellater(ATLAS, "fsLR", resampling_target=None).fit()
    # -------------------------------------
    # 第二步：处理梯度t图
    # -------------------------------------
    print("\n[2] 处理梯度t图")
    # 6. 提取ROI
    # =========================================
    gradient_surface = transforms.mni152_to_fslr(str(GRADIENT), "32k")
    gradient_roi = np.asarray(parcellater.transform(gradient_surface, "fsLR"), dtype=float).ravel()
    # 自动识别ROI数量
    n_roi = len(gradient_roi)
    print("ROI数量：", n_roi)
    # -------------------------------------
    # 第三步：读取图谱并提取ROI
    # -------------------------------------
    print("\n[3] 批量转换图谱与提取ROI")
    annotations = load_annotations()
    results = []
    skipped = []
    for i, item in enumerate(annotations, 1):
        name = item["map_id"]
        print(f"[{i}/{len(annotations)}] {name}")
        try:
            # 统一转换到fsLR32k
            surface = to_fslr(item)
            # 提取ROI
            roi = np.asarray(parcellater.transform(surface, "fsLR"), dtype=float).ravel()
            results.append({"map_id": name, "roi": roi})
        except Exception as error:
            print("  跳过：", error)
            skipped.append({"map_id": name, "reason": str(error)})
    if not results:
        raise RuntimeError("没有有效参考图谱")
    # -------------------------------------
    # 第四步：生成空间零模型
    # -------------------------------------
    print("\n[4] 生成空间零模型")
    null_maps = generate_nulls(gradient_roi)
    # -------------------------------------
    # 第五步：空间相关分析
    # -------------------------------------
    print("\n[5] 空间相关分析")
    valid = []
    for row in results:
        try:
            # 计算相关系数及空间置换p值
            r, p = stats.compare_images(gradient_roi, row["roi"], metric=CORR, nulls=null_maps, ignore_zero=False, nan_policy="omit")
            valid.append({"map_id": row["map_id"], "r": float(r), "p_spatial": float(p)})
        except Exception as error:
            print("  相关分析跳过：", row["map_id"], error)
            skipped.append({"map_id": row["map_id"], "reason": str(error)})
    if not valid:
        raise RuntimeError("没有有效相关结果")
    # -------------------------------------
    # 第六步：FDR-BH矫正
    # -------------------------------------
    print("\n[6] FDR-BH矫正")
    reject, q_values, _, _ = multipletests([row["p_spatial"] for row in valid], alpha=0.05, method="fdr_bh")
    for row, q, sig in zip(valid, q_values, reject):
        row["q_fdr"] = float(q)
        row["significant"] = bool(sig)
    # -------------------------------------
    # 第七步：保存结果
    # -------------------------------------
    print("\n[7] 保存CSV")
    valid.sort(key=lambda row: row["q_fdr"])
    save_csv("results.csv", valid, ["map_id", "r", "p_spatial", "q_fdr", "significant"])
    save_csv("skipped.csv", skipped, ["map_id", "reason"])
if __name__ == "__main__":
    main()