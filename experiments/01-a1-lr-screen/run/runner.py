from paper_exp.runner import run_launch


CONFIGS = (
    "experiments/01-a1-lr-screen/run/001-a1-lr-5e-4.yaml",
    "experiments/01-a1-lr-screen/run/002-a1-lr-1e-3.yaml",
    "experiments/01-a1-lr-screen/run/003-a1-lr-2e-3.yaml",
    "experiments/01-a1-lr-screen/run/004-a1-lr-4e-3.yaml",
    "experiments/01-a1-lr-screen/run/005-a1-lr-8e-3.yaml",
)

REQUIRED_COMPLETED_CONFIG_IDS = (
    "001-a1-lr-5e-4",
    "002-a1-lr-1e-3",
    "003-a1-lr-2e-3",
    "004-a1-lr-4e-3",
)


if __name__ == "__main__":
    run_launch(
        __file__,
        CONFIGS,
        required_completed_config_ids=REQUIRED_COMPLETED_CONFIG_IDS,
    )
