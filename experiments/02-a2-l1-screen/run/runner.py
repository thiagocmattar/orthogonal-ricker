from paper_exp.runner import ParallelLaunchAuthorization, run_launch


CONFIGS = (
    "experiments/02-a2-l1-screen/run/012-a2-relu-control.yaml",
    "experiments/02-a2-l1-screen/run/013-a2-l1-1e-1.yaml",
    "experiments/02-a2-l1-screen/run/014-a2-l1-5e-1.yaml",
    "experiments/02-a2-l1-screen/run/015-a2-l1-1.yaml",
    "experiments/02-a2-l1-screen/run/016-a2-l1-2.yaml",
    "experiments/02-a2-l1-screen/run/017-a2-l1-5.yaml",
)

PARALLEL_AUTHORIZATION = ParallelLaunchAuthorization(
    worker_count=2,
    required_gpu_name="NVIDIA A40",
    config_ids=(
        "012-a2-relu-control",
        "013-a2-l1-1e-1",
        "014-a2-l1-5e-1",
        "015-a2-l1-1",
        "016-a2-l1-2",
        "017-a2-l1-5",
    ),
)


if __name__ == "__main__":
    run_launch(
        __file__,
        CONFIGS,
        parallel_authorization=PARALLEL_AUTHORIZATION,
    )
