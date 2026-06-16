#!/usr/bin/env python3
"""
OSPF load scan — run OSPF baseline at each offered_load in the config load sweep
and report PDR, pkt_loss, utilisation, and overload fraction.

Usage:
    python3 tools/ospf_load_scan.py [--config experiment_config.json] [--episodes 10]

Outputs a TSV table to stdout and saves JSON results alongside the script.
"""
import argparse
import json
import os
import sys

import numpy as np

# Allow running from repo root or tools/
_here = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_here, '..', 'src')
sys.path.insert(0, _src)
sys.path.insert(0, os.path.join(_src, 'maddpg_clean'))

from network_environment import NetworkEngine  # noqa: E402


def run_ospf_at_load(engine: NetworkEngine, n_eps: int, t_per_ep: int,
                     load: float, mode: str = 'full') -> dict:
    ep_losses, ep_utils, ep_pdr = [], [], []
    ep_delay_p95, ep_util_p95, ep_overload = [], [], []

    for _ in range(n_eps):
        engine.reset_with_load(offered_load_factor=load)
        ep_sent = ep_dropped = 0
        for _ in range(t_per_ep):
            if mode == 'queue_level':
                info = engine.ospf_queue_level_step()
            else:
                info = engine.ospf_step()
            ep_sent    += info['packets_sent']
            ep_dropped += info['packets_dropped']

        ep_losses.append(ep_dropped / max(1, ep_sent) * 100)
        ep_utils.append(float(np.nanmean(engine.get_link_utilization_distribution())))
        s = engine.get_episode_stats()
        ep_pdr.append(float(s.get('end_to_end_pdr', 0.0)))
        ep_delay_p95.append(float(s.get('delay_p95', 0.0)))
        ep_util_p95.append(float(s.get('util_p95', 0.0)))
        ep_overload.append(float(s.get('overload_step_fraction', 0.0)))

    return {
        'load':             load,
        'pdr_mean':         float(np.mean(ep_pdr)),
        'pdr_std':          float(np.std(ep_pdr)),
        'pkt_loss_mean':    float(np.mean(ep_losses)),
        'pkt_loss_std':     float(np.std(ep_losses)),
        'util_mean':        float(np.mean(ep_utils)),
        'util_p95_mean':    float(np.mean(ep_util_p95)),
        'delay_p95_mean':   float(np.mean(ep_delay_p95)),
        'overload_frac':    float(np.mean(ep_overload)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',   default='experiment_config.json')
    ap.add_argument('--episodes', type=int, default=10,
                    help='episodes per load point (default 10)')
    ap.add_argument('--timesteps', type=int, default=None,
                    help='timesteps per episode (default: from config)')
    ap.add_argument('--mode', choices=['full', 'queue_level'], default='full')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    topo     = cfg['topology']
    loads    = [float(x) for x in cfg['load_sweep']['offered_loads']]
    t_per_ep = args.timesteps or int(cfg['training']['timesteps_per_episode'])
    n_eps    = args.episodes

    print(f"Topology : {topo['type']}  nodes={topo['nodes']}")
    print(f"Mode     : OSPF-{args.mode}")
    print(f"Episodes : {n_eps} × {t_per_ep} timesteps per load point")
    print(f"Loads    : {loads}")
    print()
    print(f"{'load':>6}  {'PDR%':>7}  {'±':>5}  {'loss%':>7}  {'±':>5}  "
          f"{'util_mean':>10}  {'util_p95':>9}  {'delay_p95':>10}  {'overload_frac':>13}")
    print('-' * 92)

    engine = NetworkEngine(
        topology_type=topo['type'],
        n_nodes=topo['nodes'],
        seed=42,
    )

    results = []
    for load in loads:
        r = run_ospf_at_load(engine, n_eps=n_eps, t_per_ep=t_per_ep,
                             load=load, mode=args.mode)
        results.append(r)
        print(f"{r['load']:>6.2f}  {r['pdr_mean']:>7.2f}  {r['pdr_std']:>5.2f}  "
              f"{r['pkt_loss_mean']:>7.2f}  {r['pkt_loss_std']:>5.2f}  "
              f"{r['util_mean']:>10.4f}  {r['util_p95_mean']:>9.4f}  "
              f"{r['delay_p95_mean']:>10.2f}  {r['overload_frac']:>13.4f}")

    print()
    # Save JSON next to config
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                            'ospf_load_scan.json')
    with open(out_path, 'w') as f:
        json.dump({'mode': f'ospf_{args.mode}', 'episodes': n_eps,
                   'timesteps': t_per_ep, 'results': results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
