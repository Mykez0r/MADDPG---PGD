#!/usr/bin/env python3
"""
Sweep flow_count to find the congested-but-learnable operating point.
Runs OSPF baseline at each flow count and reports util_p95, pkt_loss, PDR.
Target: util_p95 ~ 0.6-0.8, pkt_loss 5-20%, pdr > 70%.

Usage:
    python3 tools/flow_count_scan.py [--config experiment_config.json]
"""
import argparse
import json
import sys
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from maddpg_clean.network_environment import NetworkEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiment_config.json')
    ap.add_argument('--n-eps', type=int, default=8)
    ap.add_argument('--timesteps', type=int, default=128)
    ap.add_argument('--flow-duration', type=int, default=30)
    ap.add_argument('--load-factor', type=float, default=1.0)
    ap.add_argument('--flow-counts', type=int, nargs='+',
                    default=[10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    topo    = cfg['topology']
    traffic = cfg['traffic'].copy()

    print(f"Topology : {topo['type']}  nodes={topo['nodes']}")
    print(f"flow_duration_steps={args.flow_duration}, load_factor={args.load_factor}, "
          f"{args.n_eps} eps x {args.timesteps} steps")
    print()
    print(f"{'flow_count':>12}  {'util_mean':>10}  {'util_p95':>9}  "
          f"{'pkt_loss%':>10}  {'pdr%':>7}  {'overload':>9}  {'drop_overflow%':>15}")
    print('-' * 85)

    results = []
    for fc in args.flow_counts:
        traffic['flow_count']          = fc
        traffic['flow_duration_steps'] = args.flow_duration

        engine = NetworkEngine(
            topology_type  = topo['type'],
            n_nodes        = topo['nodes'],
            seed           = 42,
            traffic_config = traffic,
        )

        util_means, util_p95s, losses, pdrs, overloads, drop_overflows = [], [], [], [], [], []

        for _ in range(args.n_eps):
            engine.reset_with_load(offered_load_factor=args.load_factor)
            sent = dropped = 0
            for _ in range(args.timesteps):
                info = engine.ospf_step()
                sent    += info['packets_sent']
                dropped += info['packets_dropped']

            losses.append(dropped / max(1, sent) * 100)
            util_means.append(float(np.nanmean(engine.get_link_utilization_distribution())))
            s = engine.get_episode_stats()
            pdrs.append(float(s.get('end_to_end_pdr', 0.0)))
            util_p95s.append(float(s.get('util_p95', 0.0)))
            overloads.append(float(s.get('overload_step_fraction', 0.0)))
            drop_overflows.append(float(s.get('drop_overflow_rate', s.get('mean_drop_overflow_rate', 0.0))))

        row = {
            'flow_count':    fc,
            'util_mean':     float(np.mean(util_means)),
            'util_p95':      float(np.mean(util_p95s)),
            'pkt_loss_pct':  float(np.mean(losses)),
            'pdr_pct':       float(np.mean(pdrs)),
            'overload_frac': float(np.mean(overloads)),
            'drop_overflow': float(np.mean(drop_overflows)),
        }
        results.append(row)
        print(f"{fc:>12}  {row['util_mean']:>10.4f}  {row['util_p95']:>9.4f}  "
              f"{row['pkt_loss_pct']:>10.3f}  {row['pdr_pct']:>7.2f}  "
              f"{row['overload_frac']:>9.4f}  {row['drop_overflow']:>15.4f}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(args.config)), 'flow_count_scan.json')
    with open(out, 'w') as f:
        json.dump({'params': vars(args), 'results': results}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
