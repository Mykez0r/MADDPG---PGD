"""
Partial/'final FGSM robustness analyzer for Paper 2.

Reads a phase3_fgsm_results.json (written incrementally, one variant at a time)
and reports the exploitability picture for every variant completed so far:
clean vs attacked PDR per attack type and epsilon, the drop, the SLO verdict,
and --- the load-bearing comparison --- the gradient attack versus the
budget-matched RANDOM control at the same epsilon, which separates a real
adversarial signal from mere perturbation jostling.

Runs on stdlib only (no numpy), so it works anywhere.

Usage:
    python tools/analyze_fgsm.py [path/to/phase3_fgsm_results.json]
Default path: host_data/results/fgsm_rf/phase3_fgsm_results.json
"""
import json
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "host_data/results/fgsm_rf/phase3_fgsm_results.json"
PDR = "mean_end_to_end_pdr"
NON_CASE = {"_pruned", "surface", "gnn_embedding_attack", "attack_summary",
            "_runtime", "error"}
ALL_VARIANTS = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
                "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]


def case_rows(vr):
    """Yield (attack_type, epsilon, clean_pdr, attacked_pdr, slo_ok) per case."""
    for key, c in vr.items():
        if key in NON_CASE or not isinstance(c, dict):
            continue
        if "attacked" not in c or "clean" not in c:
            continue
        rc = c.get("run_config", {})
        atype = rc.get("attack_type", key)
        eps = rc.get("epsilon")
        frac = rc.get("attack_fraction")
        clean = c["clean"].get(PDR)
        att = c["attacked"].get(PDR)
        slo = c.get("slo", {}).get("success")
        yield atype, eps, frac, clean, att, slo


def main():
    if not os.path.exists(PATH):
        print(f"No results file yet at {PATH}")
        return
    d = json.load(open(PATH))
    done = [v for v in ALL_VARIANTS if v in d]
    pending = [v for v in ALL_VARIANTS if v not in d]
    print(f"FGSM analysis — {PATH}")
    print(f"variants complete: {len(done)}/7  {done}")
    if pending:
        print(f"pending: {pending}")
    print("=" * 72)

    # accumulate the cross-variant exploitability summary
    verdicts = {}

    for v in done:
        vr = d[v]
        rows = list(case_rows(vr))
        if not rows:
            print(f"\n### {v} — no completed cases (pruned early?)")
            continue
        clean_mean = sum(r[3] for r in rows) / len(rows)
        print(f"\n### {v}   (clean PDR ~ {clean_mean:.1f}%)")

        # group by attack type
        by_type = {}
        for atype, eps, frac, clean, att, slo in rows:
            by_type.setdefault(atype, []).append((eps, frac, clean, att, slo))

        # gradient (packet_loss, full-fraction) vs random control, matched eps
        grad = {eps: (clean, att) for eps, frac, clean, att, slo
                in by_type.get("packet_loss", []) if not frac}
        rand = {eps: (clean, att) for eps, frac, clean, att, slo
                in by_type.get("random", [])}

        for atype in ("packet_loss", "reward_minimize", "confusion", "random"):
            if atype not in by_type:
                continue
            print(f"  {atype}:")
            for eps, frac, clean, att, slo in sorted(
                    by_type[atype], key=lambda r: (r[0] or 0, r[1] or 0)):
                drop = clean - att
                tag = "" if frac is None else f" frac={frac}"
                flag = "  <-- SLO BREAK" if slo is False else ""
                extra = ""
                if atype == "packet_loss" and not frac and eps in rand:
                    rdrop = rand[eps][0] - rand[eps][1]
                    net = drop - rdrop
                    extra = f"   (random ctrl drop {rdrop:+.1f}; net attack {net:+.1f})"
                print(f"    eps={eps:<5}{tag}  clean {clean:5.1f} -> {att:5.1f}"
                      f"   drop {drop:+5.1f}pp{extra}{flag}")

        # verdict: max net gradient drop above random control
        max_net = None
        for eps in grad:
            gdrop = grad[eps][0] - grad[eps][1]
            rdrop = (rand[eps][0] - rand[eps][1]) if eps in rand else 0.0
            net = gdrop - rdrop
            max_net = net if max_net is None else max(max_net, net)
        if max_net is not None:
            verdict = ("EXPLOITABLE" if max_net >= 5 else
                       "borderline" if max_net >= 2 else "ROBUST")
            verdicts[v] = (max_net, verdict)
            print(f"  => max net gradient drop above random control: "
                  f"{max_net:+.1f}pp  [{verdict}]")

    if verdicts:
        print("\n" + "=" * 72)
        print("EXPLOITABILITY SUMMARY (net packet_loss drop above random control):")
        for v in done:
            if v in verdicts:
                net, verdict = verdicts[v]
                print(f"  {v:18s} {net:+6.1f}pp   {verdict}")


if __name__ == "__main__":
    main()
