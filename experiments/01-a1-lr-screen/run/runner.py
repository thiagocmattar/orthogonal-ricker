from paper_exp.runner import ParallelLaunchAuthorization, run_launch


CONFIGS = (
    "experiments/01-a1-lr-screen/run/001-a1-lr-5e-4.yaml",
    "experiments/01-a1-lr-screen/run/002-a1-lr-1e-3.yaml",
    "experiments/01-a1-lr-screen/run/003-a1-lr-2e-3.yaml",
)

PARALLEL_AUTHORIZATION = ParallelLaunchAuthorization(
    worker_count=2,
    required_gpu_name="NVIDIA A40",
    config_ids=(
        "001-a1-lr-5e-4",
        "002-a1-lr-1e-3",
        "003-a1-lr-2e-3",
    ),
)


if __name__ == "__main__":
    run_launch(
        __file__,
        CONFIGS,
        parallel_authorization=PARALLEL_AUTHORIZATION,
    )
