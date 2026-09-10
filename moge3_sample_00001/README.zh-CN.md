# MoGe-3 真实片段运行结果

测试日期：2026-09-06。

实际输入：/mnt/nuplan/l3data-reconstruction-bingxing/samples/sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/converted/images

用户路径中的 l3-data-reconstruction-bingxing 不存在，实际为 l3data-reconstruction-bingxing。
输入共 707 张、101 帧、7 相机；本次抽测第 0、50、100 帧共 21 张，不是全量运行。
保留 1920×1280 或 1920×1080 原尺寸，默认 resolution_level=9，refine_steps=3。
使用之前安装的 MoGe/.venv，权重 weights/moge-3-vitl/model.pt；未补装包、未修改模型。

官方 CLI 退出码 0，进度条总耗时约 2 分 13 秒（含推理和导出，不含启动加载）。
21/21 张导出回读检查通过：有效区域深度为正且有限，点坐标有限，点云非空，GLB 可加载。
有效区域比例 77.63%–94.51%。
每张点云 1,795,798–2,203,626 点。
无真值精度评估，不能据此断言距离尺度准确或所有场景无 bug。
每个点云是单张图片估计，没有跨相机/跨帧位姿配准融合。

结果目录：/mnt/zhoukaixuan_workspace/code/DVP-MVS/moge3_sample_00001/results
每张图片目录包含 image.jpg、depth.exr、depth_vis.png、points.exr、mask.png、normal.png、fov.json、pointcloud.ply、mesh.glb。
结果约 4.0 GiB。总览：overview.jpg；数值检查：validation.json；日志：run.log。
