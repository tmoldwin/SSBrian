import argparse

from brian_experiment.feedforward.configs import config_manager as cfg_manager
from brian_experiment.feedforward.sweeps.parameter_sweep import ParameterSweep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a feedforward sweep defined via JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Name or path to a sweep JSON under configs/sweeps.",
    )
    return parser.parse_args()


def build_sweep(payload: cfg_manager.LoadedConfig) -> ParameterSweep:
    params = payload.params
    base_params = params.get("base_params", {})
    varied_params = params.get("varied_params", {})

    sweep_kwargs = {
        "sweep_name": params.get("sweep_name", payload.name),
        "base_params": base_params,
        "varied_params": varied_params,
        "sweep_type": params.get("sweep_type", "grid"),
        "random_samples": params.get("random_samples"),
        "random_seed": params.get("random_seed", 42),
        "generate_individual_plots": params.get("generate_individual_plots", False),
        "plots_config": params.get("plots_config"),
        "n_runs_per_param": params.get("n_runs_per_param", 1),
    }

    return ParameterSweep(**sweep_kwargs)


def main():
    args = parse_args()
    payload = cfg_manager.load_sweep_config(args.config)
    sweep = build_sweep(payload)
    sweep.run()


if __name__ == "__main__":
    main()

