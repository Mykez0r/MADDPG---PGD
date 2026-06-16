#!/usr/bin/env python3
import json
import os

from standalone_experiment_runner import NetworkEngine, StandaloneExperimentRunner


def main():
    config_path = os.environ.get("PHASE2_CONFIG", "/workspace/experiment_config.json")
    results_dir = os.environ.get("PHASE2_RESULTS_DIR", "/workspace/data/results/main_run")
    variants_raw = os.environ.get("PHASE2_VARIANTS", "LC-Simple,CC-Simple")
    variants = {v.strip() for v in variants_raw.split(",") if v.strip()}
    n_eps = int(os.environ.get("PHASE2_DIAG_EPS", "3"))
    n_failures = int(os.environ.get("PHASE2_DIAG_FAILURES", "0"))

    runner = StandaloneExperimentRunner(config_path=config_path, gpu_id=0, results_dir=results_dir)
    training = runner._load("phase1_training_results.json")
    t_per_ep = runner.config["training"]["timesteps_per_episode"]

    out = {}
    for vcfg in runner.config["variants"]:
        name = vcfg["name"]
        if name not in variants or name not in training:
            continue
        maddpg, _engine, env = runner._make_variant(vcfg)
        runner._load_variant_checkpoint(maddpg, name)
        out[name] = runner._run_eval_episodes(
            maddpg,
            env,
            n_eps=n_eps,
            t_per_ep=t_per_ep,
            n_link_failures=n_failures,
        )

    topology_type = runner.config.get("topology", {}).get("type", "service_provider")
    engine = NetworkEngine(topology_type=topology_type, n_nodes=65)
    out["OSPF_FULL"] = runner._run_ospf_episodes(
        engine,
        n_eps=n_eps,
        t_per_ep=t_per_ep,
        n_link_failures=n_failures,
        mode="full",
    )

    print("DIAG_START")
    print(json.dumps(out, indent=2, sort_keys=True))
    print("DIAG_END")


if __name__ == "__main__":
    main()
