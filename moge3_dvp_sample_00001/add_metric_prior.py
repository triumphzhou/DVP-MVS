from pathlib import Path
p=Path(__file__).resolve().parents[1]/'APD.cpp';b=p.read_bytes();nl=b'\r\n' if b'\r\n' in b else b'\n'
old=b'\tparams_host.use_mono_prior = exists(problem.dense_folder / "dep"'
assert old in b
start=b.index(old);end=b.index(b'\t//if (false){',start)
original=b[start:end]
text='''\tconst path metric_depth_path = problem.dense_folder / "metric_prior" / (ToFormatIndex(problem.ref_image_id) + ".dmb");
\tconst bool has_metric_prior = exists(metric_depth_path);
\tparams_host.use_mono_prior = has_metric_prior || (exists(problem.dense_folder / "dep" / (ToFormatIndex(problem.ref_image_id) + ".dmb")) && exists(problem.dense_folder / "sfm" / (ToFormatIndex(problem.ref_image_id) + ".txt")));
\tif (params_host.state == FIRST_INIT && has_metric_prior) {
\t\tcv::Mat prior_depth, prior_normal;
\t\tconst path normal_path = problem.dense_folder / "metric_prior" / (ToFormatIndex(problem.ref_image_id) + "_normal.dmb");
\t\tif (!ReadBinMat(metric_depth_path, prior_depth) || prior_depth.type() != CV_32FC1 ||
\t\t    !ReadBinMat(normal_path, prior_normal) || prior_normal.type() != CV_32FC3 || prior_depth.size() != prior_normal.size())
\t\t\tthrow std::runtime_error("Invalid metric depth/world-normal prior");
\t\tRescaleMatToTargetSize<float>(prior_depth, prior_depth, cv::Size(width, height));
\t\tRescaleMatToTargetSize<cv::Vec3f>(prior_normal, prior_normal, cv::Size(width, height));
\t\tint valid_count = 0;
\t\tfor (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
\t\t\tconst float d = prior_depth.at<float>(y, x);
\t\t\tconst cv::Vec3f n = prior_normal.at<cv::Vec3f>(y, x);
\t\t\tconst float norm = cv::norm(n);
\t\t\tif (std::isfinite(d) && d >= params_host.depth_min && d <= params_host.depth_max && std::isfinite(norm) && norm > 0.5f) {
\t\t\t\tplane_hypotheses_host[y * width + x] = make_float4(n[0]/norm, n[1]/norm, n[2]/norm, d);
\t\t\t\t++valid_count;
\t\t\t}
\t\t}
\t\tstd::cout << "Metric MoGe prior loaded: " << valid_count << " / " << width*height << " pixels" << std::endl;
\t} else if (params_host.state == FIRST_INIT && !params_host.use_mono_prior) {
\t\tstd::cout << "No complete prior; using random PatchMatch fallback" << std::endl;
\t}
\tif (params_host.state == FIRST_INIT && params_host.use_mono_prior && !has_metric_prior) {
'''.encode().replace(b'\n',nl)
p.write_bytes(b[:start]+text+b[end:])
