from paper_exp.runner import ParallelLaunchAuthorization, run_launch


CONFIGS = (
    "experiments/01-a1-lr-screen/run/001-a1-lr-5e-4.yaml",
    "experiments/01-a1-lr-screen/run/002-a1-lr-1e-3.yaml",
    "experiments/01-a1-lr-screen/run/003-a1-lr-2e-3.yaml",
    "experiments/01-a1-lr-screen/run/004-a1-lr-4e-3.yaml",
    "experiments/01-a1-lr-screen/run/005-a1-lr-8e-3.yaml",
    "experiments/01-a1-lr-screen/run/006-a1-lr-1p6e-2.yaml",
    "experiments/01-a1-lr-screen/run/007-a1-lr-3p2e-2.yaml",
    "experiments/01-a1-lr-screen/run/008-a1-lr-6p4e-2.yaml",
)

REQUIRED_COMPLETED_CONFIG_IDS = (
    "001-a1-lr-5e-4",
    "002-a1-lr-1e-3",
    "003-a1-lr-2e-3",
    "004-a1-lr-4e-3",
    "005-a1-lr-8e-3",
)

PARALLEL_AUTHORIZATION = ParallelLaunchAuthorization(
    worker_count=3,
    required_gpu_name="NVIDIA A40",
    config_ids=(
        "001-a1-lr-5e-4",
        "002-a1-lr-1e-3",
        "003-a1-lr-2e-3",
        "004-a1-lr-4e-3",
        "005-a1-lr-8e-3",
        "006-a1-lr-1p6e-2",
        "007-a1-lr-3p2e-2",
        "008-a1-lr-6p4e-2",
    ),
)


if __name__ == "__main__":
    run_launch(
        __file__,
        CONFIGS,
        parallel_authorization=PARALLEL_AUTHORIZATION,
        required_completed_config_ids=REQUIRED_COMPLETED_CONFIG_IDS,
    )
