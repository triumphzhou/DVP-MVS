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

## 批量运行到 OpenMVS 融合点云

[`run_batch_pkl_to_openmvs_ply.sh`](run_batch_pkl_to_openmvs_ply.sh) 将上述八步继续串到
MoGeV3、DVP-MVS 和 OpenMVS depth filter/fusion。最终的数据流为：

```text
PKL
  -> converted
  -> COLMAP 固定姿态稀疏模型和 OpenMVS scene.mvs
  -> MoGeV3 米制深度先验
  -> DVP-MVS PatchMatch 深度
  -> OpenMVS DMAP
  -> OpenMVS 深度图去除 speckle + dense-fuse
  -> openmvs_filtered_fused.ply
```

复制批次清单后，每行填写一个 101 帧分段：

```bash
cp data_preprocess/batch.example.tsv /tmp/dvp_batch.tsv
```

清单是制表符分隔的 TSV：

```text
clip_name  side  frame_start  frame_end  sample_name
```

检查依赖并运行：

```bash
bash data_preprocess/run_batch_pkl_to_openmvs_ply.sh --check

bash data_preprocess/run_batch_pkl_to_openmvs_ply.sh \
  --batch-file /tmp/dvp_batch.tsv \
  --output-root /mnt/nuplan/l3data-reconstruction-bingxing/preprocess_runs \
  --resume
```

也可以直接处理一个分段：

```bash
bash data_preprocess/run_batch_pkl_to_openmvs_ply.sh \
  --clip-name clip_M18-2_07_20251202110510_DF \
  --side left --frame-start 5 --frame-end 105 \
  --output-root /mnt/nuplan/l3data-reconstruction-bingxing/preprocess_runs \
  --resume
```

批处理默认使用 7 张 GPU，MoGe/LiDAR 尺度标定不设 80 m 上限，DVP 和 OpenMVS
深度范围都是 `0.1–100 m`。可以分别用 `--gpu-ids`、`--bounded-lidar`、
`--depth-min` 和 `--depth-max` 修改。`--from-stage 9 --resume` 可以复用已有的
步骤 1–8，从 MoGeV3 开始继续。

步骤 11 根据当前 `scene.mvs`、COLMAP image ID 和邻居文件生成 DMAP，避免借用其他样本
的 OpenMVS 深度模板。步骤 12 使用 `--postprocess-dmaps 1` 去除深度 speckle，再以
`--fusion-filter 2`、两视角最低支持执行 dense-fuse；输入深度是 DVP-MVS 的输出。

每个样本的两个点云位于：

```text
<output-root>/<sample>/09_dvp/scene/APD/APD.ply
<output-root>/<sample>/10_openmvs_fusion/openmvs_filtered_fused.ply
```

中间深度和点云属于运行结果，默认输出根目录位于仓库外；仓库内默认的
`data_preprocess/runs/` 也已被 Git 忽略。

## 代码来源与改动

步骤 1–4 来自已经核对过的原始脚本；步骤 5–8来自旧版
`colmap+openmvs/run_l3_full.sh` 及其 `pipeline_tools`。收拢时做了以下必要调整：

- 所有代码依赖改成仓库内相对路径；
- 扫描器增加 `--clip-name`，允许只扫描目标 clip；
- 当前 V5 PKL 根目录改成可配置参数；
- 把旧 Bash 中的内联 Python 拆成可检查的独立工具；
- 八步入口只生成 OpenMVS 输入，不执行稠密重建。
