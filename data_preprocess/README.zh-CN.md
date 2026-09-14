# L3 数据预处理：PKL 到 OpenMVS `scene.mvs`

本目录集中保存原先分散在多个挂载目录中的数据预处理代码。入口
[`run_preprocess_1_to_8.sh`](run_preprocess_1_to_8.sh) 针对一个 101 帧分段执行八步流程，
最终生成 OpenMVS 可读取的 `scene.mvs`、图像、忽略 mask 和邻居文件。

这里不保存原始 PKL、图像、模型权重、COLMAP/OpenMVS 可执行文件或运行结果。
这些大文件通过环境变量传入。默认输出目录 `data_preprocess/runs/` 已被 Git 忽略。

## 八个步骤

| 步骤 | 输入 | 代码 | 输出 |
|---:|---|---|---|
| 1 | clip 清单、PKL 路径清单、原始 PKL | `01_scan_original_clip_windows_v4.py`、`01_collision_detection_v4.py` | `selected_101f_segments.tsv` 和选中的分段 JSON |
| 2 | 选中 PKL 与原始帧范围 | `02_convert_m18_pkl.py`、`M18proc/__init__.py` | `02_converted_absolute/` |
| 3 | 绝对帧号数据 | `03_normalize_segment_0based.py` | 帧号为 `000000–000100` 的 `03_converted/` |
| 4 | `converted` 图像、内外参和位姿 | `04_prepare_openmvs_input.py` | 轻量符号链接视图和 `input_manifest.json` |
| 5 | 图像与 manifest | `05_prepare_image_lists_and_pairs.py` | 七个相机清单、±20 帧 COLMAP 候选图对 |
| 6 | 707 张图像 | `06_prepare_full_sky_masks.py`、`06_add_camera10_rig_mask.py` | OpenMVS 天空/车体忽略 mask |
| 7 | 图像、内参、固定 pose、候选图对 | `07_*` 工具和 COLMAP | 固定姿态、不做 BA 的三角化稀疏模型 |
| 8 | COLMAP 稀疏模型 | `InterfaceCOLMAP`、`08_build_diverse_neighbors.py` | `scene.mvs` 和每张参考图 20 个源视图的邻居文件 |

步骤 1 的 `frame_start` 和 `frame_end` 都包含端点。例如 `5–105` 是 101 帧。
步骤 2 调用转换器时会改成半开区间 `[5,106)`；步骤 3 再减去 offset 5，最终帧号为
`0–100`。步骤 3 不修改相机矩阵、位姿数值或点云坐标。

## 一键运行

先检查当前机器的依赖路径和 Python 包：

```bash
bash data_preprocess/run_preprocess_1_to_8.sh --check
```

处理已经核对过的 `clip_M18-2_07_20251202110510_DF` 左侧第 5–105 帧：

```bash
bash data_preprocess/run_preprocess_1_to_8.sh \
  --clip-name clip_M18-2_07_20251202110510_DF \
  --side left \
  --frame-start 5 \
  --frame-end 105 \
  --output-root /mnt/nuplan/l3data-reconstruction-bingxing/preprocess_runs
```

如果已经有步骤 1 生成的 TSV，可以复用它：

```bash
bash data_preprocess/run_preprocess_1_to_8.sh \
  --segments-tsv /path/to/selected_101f_segments.tsv \
  --clip-name clip_M18-2_07_20251202110510_DF \
  --side left --frame-start 5 --frame-end 105 \
  --output-root /path/to/output
```

中断后，只有写入 `.state/step_N.complete` 的完整步骤才会复用：

```bash
bash data_preprocess/run_preprocess_1_to_8.sh \
  --clip-name clip_M18-2_07_20251202110510_DF \
  --side left --frame-start 5 --frame-end 105 \
  --output-root /path/to/output --resume
```

可用 `--from-step N` 和 `--to-step N` 限定运行范围。若某一步中途失败且未产生完成标记，
脚本会拒绝覆盖该步的半成品目录；删除该步目录后再用 `--resume` 继续。

## 本机配置

默认路径记录在 [`config.example.env`](config.example.env)。如果机器路径不同，可以复制一份，
修改后在运行前加载：

```bash
cp data_preprocess/config.example.env /tmp/dvp_preprocess.env
# 编辑 /tmp/dvp_preprocess.env
set -a
source /tmp/dvp_preprocess.env
set +a
bash data_preprocess/run_preprocess_1_to_8.sh --check
```

主要外部依赖为：

- 能导入 `numpy`、`opencv-python`、`open3d` 的 Python；
- 生成天空 mask 时还需要 `torch`、`Pillow`、`transformers` 和 SegFormer 权重；
- 支持 CUDA SIFT 的 COLMAP；
- OpenMVS `InterfaceCOLMAP` 及其动态库。

如果暂时不生成天空 mask，可加 `--no-sky-mask`。此时仍会生成全零 OpenMVS mask，
并默认叠加 camera 10 的固定车体区域。

## 输出

每个分段的目录结构如下：

```text
<output-root>/<clip>_f<start>_<end>_<side>/
├── 01_scan/
├── 02_converted_absolute/
├── 03_converted/
├── 04_openmvs_input/
│   └── input_manifest.json
├── 05_pairs/
│   └── colmap_match_pairs.txt
├── 06_masks/openmvs_masks/
├── 07_sparse/triangulated_binary/
├── 08_openmvs_input/global/
│   ├── scene.mvs
│   └── neighbors_diverse_pm20_top20.txt
├── audit/
└── logs/
```

步骤 8 完成后可以将以下四个路径交给 OpenMVS 稠密重建：

```text
scene       = 08_openmvs_input/global/scene.mvs
images      = 04_openmvs_input/images
masks       = 06_masks/openmvs_masks
neighbors   = 08_openmvs_input/global/neighbors_diverse_pm20_top20.txt
```

该入口在步骤 8 停止，不运行 OpenMVS PatchMatch/融合，也不会生成结果 PLY。

## 从准备好的多视图数据批量运行到点云

[`run_batch_mvsnet_to_openmvs_ply.sh`](run_batch_mvsnet_to_openmvs_ply.sh) 不处理 PKL，
也不运行 COLMAP。它只读取 `run_preprocess_1_to_8.sh` 已经生成的多视图样本目录，执行
MoGeV3、DVP-MVS 和 OpenMVS depth filter/fusion：

```text
准备好的 sample-root
  ├── 03_converted
  ├── 06_masks
  ├── 07_sparse
  └── 08_openmvs_input/scene.mvs
  -> MoGeV3 米制深度先验
  -> DVP-MVS PatchMatch 深度
  -> OpenMVS DMAP
  -> OpenMVS 深度图去除 speckle + dense-fuse
  -> openmvs_filtered_fused.ply
```

## 从 MVSNet 风格数据到 DVP-MVS 点云

这里需要区分两个目录。`03_converted/` 是归一化后的传感器数据，还不是 APD 直接读取的
MVSNet 格式；batch 阶段 1 会把它和 mask、邻居关系转换为 `09_dvp/scene/`。后者才是 DVP-MVS
实际读取的 MVSNet 风格场景：

```text
09_dvp/scene/
├── images/                         # 00000000.jpg 等连续八位 ID
├── cams/                           # 00000000_cam.txt：W2C、K、深度范围
├── pair.txt                        # 每张参考图及其 20 张源图
├── metric_prior/
│   ├── 00000000.dmb                # MoGeV3 米制深度先验
│   └── 00000000_normal.dmb         # 世界坐标法线先验
└── blocks/mask_0.jpg               # 有效区域为 255
```

标准 MVSNet 的 `images/cams/pair.txt` 还不够用于当前定制版 APD；还需要
`metric_prior` 和 `blocks`。它们由
[`prepare_clean.py`](../moge3_dvp_sample_00001_clean7/prepare_clean.py) 和
[`finalize_clean.py`](../moge3_dvp_sample_00001_clean7/finalize_clean.py) 生成。

### Batch 阶段 1：生成 DVP-MVS 输入

| 读入数据 | 变换 | DVP 输出 |
|---|---|---|
| `03_converted/images` | 保持宽高比缩放并放入 `960×640` 画布 | `scene/images` |
| `03_converted/intrinsics` | 缩放 `fx/fy/cx/cy`，主点再加 padding | `scene/cams` 中的 `K` |
| `03_converted/ego_pose` | `W2C = inverse(C2W)` | `scene/cams` 中的外参 |
| RGB + MoGeV3 | 预测稠密深度 | 相对/单目深度形状 |
| `03_converted/lidar_depth` | `median(LiDAR/MoGe)` 求每张图米制尺度 | `metric_prior/*.dmb` |
| OpenMVS 天空和车体 mask | 去掉天空、车体、padding 和模型无效像素 | `blocks/mask_*.jpg` |
| OpenMVS/COLMAP 邻居 TSV | 映射成连续 DVP ID，每张图选 20 个源视图 | `pair.txt` |

相机 `00–04` 的 `1920×1280` 图像变成 `960×640`。相机 `09/10` 的
`1920×1080` 图像先变成 `960×540`，再在上下各补 50 像素。MoGeV3 在未补边的有效内容
上推理，稀疏 LiDAR 用下面的中位数比例把预测深度变成米：

```text
metric_depth = moge_depth × median(lidar_depth / moge_depth)
```

默认批处理使用 `--unbounded-lidar-calibration`，即尺度标定采用所有大于 2 m 的有效
LiDAR 点，不再排除 80 m 以外的点。最终先验和 APD 搜索范围仍按 `0.1–100 m` 过滤。
随后按相机对尺度做中位数/MAD 稳健裁剪，删除小于 20 像素的孤立深度块，并由深度
反投影计算世界坐标法线。

### Batch 阶段 2：APD 深度估计和 DVP 融合

APD 从 `scene/pair.txt` 开始逐张读取参考图、20 张源图、相机、mask、米制深度和法线
先验。`960×640` 输入使用 `480×320` 和 `960×640` 两层金字塔进行 PatchMatch。
每个视角输出：

```text
09_dvp/scene/APD/<八位图像ID>/depths.dmb
09_dvp/scene/APD/<八位图像ID>/APD_normals.dmb
09_dvp/scene/APD/<八位图像ID>/weak.bin
09_dvp/scene/APD/<八位图像ID>/selected_views.bin
```

APD 随后执行 DVP 自己的多视角几何一致性融合，生成：

```text
09_dvp/scene/APD/APD.ply
```

该点云是 **DVP-MVS 融合结果**，没有经过 OpenMVS depth filter。

### Batch 阶段 3–4：OpenMVS 深度过滤和融合

[`11_convert_dvp_to_openmvs_dmap.py`](steps/11_convert_dvp_to_openmvs_dmap.py)
把每张 `depths.dmb` 转成 OpenMVS `depthNNNN.dmap`。转换时会：

1. 去掉相机 `09/10` 的上下 padding；
2. 应用相同的天空/车体 mask 和 `0.1–100 m` 范围；
3. 从当前 `scene.mvs` 和 COLMAP 模型读取准确的 OpenMVS image ID；
4. 写入对应的 20 个邻居 ID、缩放后的内参、W2C 旋转和相机中心；
5. 从 DVP 深度重新计算 OpenMVS 使用的相机坐标法线。

OpenMVS 随后读取这些 DVP 深度，使用 `--postprocess-dmaps 1` 删除 speckle，再以
`--fusion-filter 2` 执行 dense-fuse。融合要求至少两个视图支持，使用 1% 深度差阈值和
2.5 像素重投影阈值。脚本会先生成并验证全部 707 个 DVP DMAP，OpenMVS 直接复用这些
已有深度；`--geometric-iters 0` 不再追加 OpenMVS 几何 PatchMatch 迭代。因此这里使用的
是 **DVP 估计的深度 + OpenMVS 过滤和融合策略**。最终生成：

```text
10_openmvs_fusion/openmvs_filtered_fused.ply
```

两个 PLY 的区别如下：

| 点云 | 深度来源 | 深度过滤与融合程序 |
|---|---|---|
| `09_dvp/scene/APD/APD.ply` | DVP-MVS `depths.dmb` | DVP-MVS `RunFusion` |
| `10_openmvs_fusion/openmvs_filtered_fused.ply` | 同一批 DVP-MVS 深度转成 DMAP | OpenMVS speckle filter + dense-fuse |

## 批次运行方法

复制批次清单后，每行填写一个已经完成步骤 1–8 的样本目录，以及可选的结果目录：

```bash
cp data_preprocess/batch.example.tsv /tmp/dvp_batch.tsv
```

清单是制表符分隔的 TSV。`result_root` 留空时，结果直接写在 `sample_root` 下：

```text
sample_root  result_root
```

检查依赖并运行：

```bash
bash data_preprocess/run_batch_mvsnet_to_openmvs_ply.sh --check

bash data_preprocess/run_batch_mvsnet_to_openmvs_ply.sh \
  --batch-file /tmp/dvp_batch.tsv \
  --resume
```

也可以直接处理一个准备好的样本：

```bash
bash data_preprocess/run_batch_mvsnet_to_openmvs_ply.sh \
  --sample-root /mnt/nuplan/l3data-reconstruction-bingxing/preprocess_runs/clip_M18-2_07_20251202110510_DF_f5_105_left \
  --result-root /mnt/nuplan/l3data-reconstruction-bingxing/dvp_openmvs_results/clip_M18-2_07_20251202110510_DF_f5_105_left \
  --resume
```

批处理默认使用 7 张 GPU，MoGe/LiDAR 尺度标定不设 80 m 上限，DVP 和 OpenMVS
深度范围都是 `0.1–100 m`。可以分别用 `--gpu-ids`、`--bounded-lidar`、
`--depth-min` 和 `--depth-max` 修改。脚本内部只有四个阶段；可以用
`--from-stage 2 --resume` 跳过已经完成的 MoGe/DVP 场景准备。

Batch 阶段 3 根据当前 `scene.mvs`、COLMAP image ID 和邻居文件生成 DMAP，避免借用其他样本
的 OpenMVS 深度模板。阶段 4 使用 `--postprocess-dmaps 1` 去除深度 speckle，再以
`--fusion-filter 2`、两视角最低支持执行 dense-fuse；输入深度是 DVP-MVS 的输出。

结果目录中的两个点云位于：

```text
<result-root>/09_dvp/scene/APD/APD.ply
<result-root>/10_openmvs_fusion/openmvs_filtered_fused.ply
```

中间深度和点云属于运行结果；仓库内的 `data_preprocess/runs/` 已被 Git 忽略。

## 代码来源与改动

步骤 1–4 来自已经核对过的原始脚本；步骤 5–8来自旧版
`colmap+openmvs/run_l3_full.sh` 及其 `pipeline_tools`。收拢时做了以下必要调整：

- 所有代码依赖改成仓库内相对路径；
- 扫描器增加 `--clip-name`，允许只扫描目标 clip；
- 当前 V5 PKL 根目录改成可配置参数；
- 把旧 Bash 中的内联 Python 拆成可检查的独立工具；
- 八步入口只生成 OpenMVS 输入，不执行稠密重建。
