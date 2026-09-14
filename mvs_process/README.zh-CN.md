# MVS processing：准备好的多视图数据到点云

本文档从数据预处理完成的多视图样本开始，说明 MoGeV3 深度先验、DVP-MVS PatchMatch，
以及 OpenMVS depth filter/dense-fuse 到最终 PLY 的完整流程。批处理已关闭 DVP 原生融合。

上游 PKL 到 `03_converted`、COLMAP 固定姿态模型和 `scene.mvs` 的过程见
[`data_preprocess/README.zh-CN.md`](../data_preprocess/README.zh-CN.md)。本流程不读取原始 PKL，
也不运行 COLMAP。

本目录集中保存这一阶段的通用代码：

```text
mvs_process/
├── run_batch_mvsnet_to_openmvs_ply.sh
├── prepare_dvp_scene.py
├── finalize_dvp_scene.py
├── convert_dvp_to_openmvs_dmap.py
├── openmvs_dvp_depth_filter.cfg
├── batch.example.tsv
├── config.example.env
└── README.zh-CN.md
```

关闭原生融合依赖当前源码中的 `DVP_SKIP_FUSION` 开关，逐轮 mask 依赖
`DVP_DENSE_MASK_V1` 实现。首次使用或更新代码后需要重新编译 `build/APD`；批处理预检会
拒绝不支持这两项功能的旧二进制，防止误跑 DVP 融合或漏掉动态 mask。

## 从准备好的多视图数据批量运行到点云

[`run_batch_mvsnet_to_openmvs_ply.sh`](run_batch_mvsnet_to_openmvs_ply.sh) 不处理 PKL，
也不运行 COLMAP。它只读取 `run_preprocess_1_to_8.sh` 已经生成的多视图样本目录，执行
MoGeV3、DVP-MVS 和 OpenMVS depth filter/fusion：

```text
准备好的 sample-root
  ├── 03_converted
  ├── 06_masks
  ├── 07_sparse
  └── 08_openmvs_input/global/scene.mvs
  -> MoGeV3 米制深度先验
  -> DVP-MVS PatchMatch 深度
  -> OpenMVS DMAP
  -> OpenMVS 深度图去除 speckle + dense-fuse
  -> openmvs_filtered_fused.ply
```

## 从 MVSNet 风格数据到 OpenMVS 融合点云

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

标准 MVSNet 的 `images/cams/pair.txt` 没有统一的动态 mask 字段。当前数据转换器额外生成
`03_converted/dynamic_mask/<帧_相机>.png`，其中非零像素表示动态目标。预处理步骤 6 将它
与天空、camera 10 车体 mask 合并成 OpenMVS 的 `255=忽略` mask。当前定制版 APD 还需要
`metric_prior` 和 `blocks`。它们由
[`prepare_dvp_scene.py`](prepare_dvp_scene.py) 和
[`finalize_dvp_scene.py`](finalize_dvp_scene.py) 生成。

### Batch 阶段 1：生成 DVP-MVS 输入

| 读入数据 | 变换 | DVP 输出 |
|---|---|---|
| `03_converted/images` | 保持宽高比缩放并放入 `960×640` 画布 | `scene/images` |
| `03_converted/intrinsics` | 缩放 `fx/fy/cx/cy`，主点再加 padding | `scene/cams` 中的 `K` |
| `03_converted/ego_pose` | `W2C = inverse(C2W)` | `scene/cams` 中的外参 |
| RGB + MoGeV3 | 预测稠密深度 | 相对/单目深度形状 |
| `03_converted/lidar_depth` | `median(LiDAR/MoGe)` 求每张图米制尺度 | `metric_prior/*.dmb` |
| OpenMVS 天空、动态目标和车体组合 mask | 去掉这些区域、padding 和模型无效像素 | `blocks/mask_*.jpg` |
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

### Batch 阶段 2：APD 深度估计

APD 从 `scene/pair.txt` 开始逐张读取参考图、20 张源图、相机、mask、米制深度和法线
先验。`960×640` 输入使用 `480×320` 和 `960×640` 两层金字塔进行 PatchMatch。组合 mask
在 DVP 输入图像中置黑，并在每次 PatchMatch 迭代写出结果前将对应深度、法线、状态和
已选视图清零，因此动态区域不会进入下一轮几何一致性，也不会进入最终 DMAP。
每个视角输出：

```text
09_dvp/scene/APD/<八位图像ID>/depths.dmb
09_dvp/scene/APD/<八位图像ID>/APD_normals.dmb
09_dvp/scene/APD/<八位图像ID>/weak.bin
09_dvp/scene/APD/<八位图像ID>/selected_views.bin
```

批处理运行 APD 时设置 `DVP_SKIP_FUSION=1`。APD 只保存逐视角深度、法线和辅助数据，
不再调用 DVP 的 `RunFusion`，也不生成 `09_dvp/scene/APD/APD.ply`。

### Batch 阶段 3–4：OpenMVS 深度过滤和融合

[`convert_dvp_to_openmvs_dmap.py`](convert_dvp_to_openmvs_dmap.py)
把每张 `depths.dmb` 转成 OpenMVS `depthNNNN.dmap`。转换时会：

1. 去掉相机 `09/10` 的上下 padding；
2. 应用相同的天空/动态目标/车体组合 mask 和 `0.1–100 m` 范围；
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

批处理只输出一个融合点云：

| 点云 | 深度来源 | 深度过滤与融合程序 |
|---|---|---|
| `10_openmvs_fusion/openmvs_filtered_fused.ply` | DVP-MVS `depths.dmb` 转成 DMAP | OpenMVS speckle filter + dense-fuse |

## 批次运行方法

机器路径不同时，先加载本目录的环境配置：

```bash
cp mvs_process/config.example.env /tmp/mvs_process.env
# 编辑 /tmp/mvs_process.env
set -a
source /tmp/mvs_process.env
set +a
```

复制批次清单后，每行填写一个已经完成步骤 1–8 的样本目录，以及可选的结果目录：

```bash
cp mvs_process/batch.example.tsv /tmp/dvp_batch.tsv
```

清单是制表符分隔的 TSV。`result_root` 留空时，结果直接写在 `sample_root` 下：

```text
sample_root  result_root
```

检查依赖并运行：

```bash
bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh --check

bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
  --batch-file /tmp/dvp_batch.tsv \
  --resume
```

也可以直接处理一个准备好的样本：

```bash
bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
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

最终点云位于：

```text
<result-root>/10_openmvs_fusion/openmvs_filtered_fused.ply
```

中间深度和点云属于运行结果；仓库内的 `data_preprocess/runs/` 已被 Git 忽略。
