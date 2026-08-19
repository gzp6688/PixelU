import torch_fidelity # type: ignore

metrics_dict = torch_fidelity.calculate_metrics(
    input1="result/PixelU-H-16/heun-epoch-600-steps50-cfg2.2-interval0.1-1.0-image50000-res256",
    input2=None,
    fid_statistics_file="fid_stats/jit_in256_stats.npz",
    cuda=True,
    isc=True,
    fid=True,
    kid=False,
    prc=False,
    verbose=False,
)
fid = metrics_dict['frechet_inception_distance']
inception_score = metrics_dict['inception_score_mean']
print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))