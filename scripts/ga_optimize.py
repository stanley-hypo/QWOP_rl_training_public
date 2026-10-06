"""QWOP trajectory optimizer: evolve action sequences directly.

Instead of training a policy, evolve the raw per-frame action sequence
against the official physics engine. Seeded from elite replays produced
by RL training runs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replay_io import ReplayRuleset, read_replay, write_replay
from running_env import VectorRunningGameEnv

KEY_ACTIONS = list(range(1, 16))

NEWLINE = chr(10)


def runs_of(seq):
    runs = []
    for a in seq:
        if runs and runs[-1][0] == a:
            runs[-1][1] += 1
        else:
            runs.append([a, 1])
    return runs


def unruns(runs):
    out = []
    for a, n in runs:
        out.extend([a] * n)
    return out


def load_seeds(paths):
    seqs = []
    seen = set()
    for p in paths:
        _, actions = read_replay(Path(p))
        key = tuple(actions)
        if key not in seen:
            seen.add(key)
            seqs.append(list(actions))
    return seqs


def mutate_runs(runs, rng, ops):
    runs = [r[:] for r in runs]
    for _ in range(ops):
        if len(runs) < 4:
            runs.append([0, 30])
        op = rng.choice(("len", "act", "dup", "del", "swap", "window"))
        i = rng.randrange(len(runs))
        if op == "len":
            runs[i][1] = max(1, runs[i][1] + rng.randint(-40, 40))
        elif op == "act":
            runs[i][0] = rng.choice(KEY_ACTIONS)
        elif op == "dup":
            runs.insert(i, runs[i][:])
        elif op == "del":
            if len(runs) > 4:
                del runs[i]
        elif op == "swap":
            j = min(i + 1, len(runs) - 1)
            runs[i], runs[j] = runs[j], runs[i]
        elif op == "window":
            if len(runs) - i >= 2:
                k = rng.randint(2, min(6, len(runs) - i))
                seg = [r[:] for r in runs[i:i + k]]
                runs[i:i] = seg
    return runs


def cataclysm(runs, rng):
    """Large-scale structural mutation: escape the local gait basin."""
    runs = [r[:] for r in runs]
    n = len(runs)
    if n < 4:
        return runs
    op = rng.choice(("rotate", "double_stride", "halve", "shuffle", "reverse_seg"))
    if op == "rotate":
        k = rng.randrange(1, n)
        runs = runs[k:] + runs[:k]
    elif op == "double_stride":
        total = sum(r[1] for r in runs if len(r) > 1)
        if total <= 12000:
            i, j = sorted(rng.sample(range(n), 2))
            runs[j:j] = [r[:] for r in runs[i:j]]
    elif op == "halve":
        runs = [r for idx, r in enumerate(runs) if idx % 2 == 0]
    elif op == "shuffle":
        rng.shuffle(runs)
    elif op == "reverse_seg":
        i, j = sorted(rng.sample(range(n), 2))
        runs[i:j] = [r[:] for r in runs[i:j][::-1]]
    if len(runs) < 4:
        runs.append([0, 30])
    return runs


def crossover(ra, rb, rng):
    if len(ra) < 8 or len(rb) < 4:
        return [r[:] for r in ra]
    c1 = rng.randrange(1, len(ra) - 2)
    c2 = rng.randrange(0, len(rb))
    k = rng.randint(1, max(1, len(rb) - c2))
    return (
        [r[:] for r in ra[:c1]]
        + [r[:] for r in rb[c2:c2 + k]]
        + [r[:] for r in ra[c1:]]
    )


def pad_actions(seqs, num_envs):
    m = max(len(s) for s in seqs)
    out = np.zeros((num_envs, m), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        if len(s) > 0:
            out[i, len(s):] = s[-1]
    return out


class Evaluator:
    """Vectorized evaluation of a population of action sequences."""

    def __init__(self, num_envs, num_threads, max_frames):
        self.env = VectorRunningGameEnv(num_envs=num_envs, num_threads=num_threads)
        self.max_frames = max_frames

    def evaluate(self, seqs):
        p = len(seqs)
        seqs = [s[: self.max_frames] for s in seqs]
        acts = pad_actions(seqs, p)
        done = np.zeros(p, dtype=bool)
        finals = [None] * p
        last_score = [0.0] * p
        self.env.reset()
        max_t = min(self.max_frames, acts.shape[1])
        for t in range(max_t):
            frame = np.where(done, 0, acts[:, t])
            result = self.env.step_arrays(frame)
            has_final = result["has_final"]
            for i in range(p):
                if done[i]:
                    continue
                if has_final[i]:
                    done[i] = True
                    score = float(result["final_score"][i])
                    distance = float(result["final_distance"][i])
                    time_s = float(result["final_time"][i])
                    success = bool(result["final_success"][i])
                    if not success and score >= 99.9:
                        success = True
                    finals[i] = {
                        "score": score,
                        "distance": distance,
                        "time": time_s,
                        "success": success,
                        "frames": t + 1,
                    }
                else:
                    last_score[i] = float(result["score"][i])
            if done.all():
                break
        results = []
        for i in range(p):
            f = finals[i]
            if f is None:
                score = last_score[i]
                results.append({
                    "fitness": score * 10.0,
                    "score": score,
                    "time": 0.0,
                    "success": False,
                    "frames": int(acts.shape[1]),
                })
            elif f["success"]:
                results.append({
                    "fitness": 10000.0 - f["time"] * 100.0,
                    "score": f["score"],
                    "time": f["time"],
                    "success": True,
                    "frames": f["frames"],
                })
            else:
                results.append({
                    "fitness": f["score"] * 10.0,
                    "score": f["score"],
                    "time": f["time"],
                    "success": False,
                    "frames": f["frames"],
                })
        return results


def tournament(ranked, rng, k=3):
    best = None
    for _ in range(k):
        ind = ranked[rng.randrange(len(ranked))]
        if best is None or ind["fit"] > best["fit"]:
            best = ind
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--generations", type=int, default=400)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=280.0)
    ap.add_argument("--out", default="replays/ga_best.bin")
    ap.add_argument("--progress", default="replays/ga_progress.jsonl")
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--num-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cataclysm-every", type=int, default=15)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seed_seqs = load_seeds(args.seeds)
    print("loaded %d unique seeds" % len(seed_seqs), flush=True)

    evaluator = Evaluator(args.pop, args.num_threads, args.max_frames)

    population = []
    for s in seed_seqs:
        population.append({"runs": runs_of(s), "fit": None, "res": None})
    while len(population) < args.pop:
        base = rng.choice(population)["runs"]
        population.append({
            "runs": mutate_runs(base, rng, rng.randint(1, 4)),
            "fit": None,
            "res": None,
        })

    best_overall = {"fitness": -1e18, "seq": None, "meta": None}
    gen = 0
    start = time.time()

    while gen < args.generations:
        if (time.time() - start) / 60.0 > args.max_minutes:
            print("time budget reached", flush=True)
            break
        seqs = [unruns(ind["runs"]) for ind in population]
        results = evaluator.evaluate(seqs)
        for ind, res in zip(population, results):
            ind["fit"] = res["fitness"]
            ind["res"] = res
        population.sort(key=lambda ind: ind["fit"], reverse=True)

        best = population[0]
        if best["fit"] > best_overall["fitness"]:
            seq = unruns(best["runs"])
            end_frames = None
            if best.get("res"):
                end_frames = best["res"].get("frames")
            if end_frames and 0 < end_frames < len(seq):
                seq = seq[:end_frames]
            best_overall = {
                "fitness": best["fit"],
                "seq": seq,
                "meta": dict(best["res"]),
            }
            write_replay(
                Path(args.out),
                best_overall["seq"],
                ReplayRuleset.CLASSIC_100M,
            )
            rec = {"gen": gen, "fitness": best["fit"]}
            rec.update(best_overall["meta"])
            print("GEN %d new best: %s" % (gen, json.dumps(rec)), flush=True)
            with open(args.progress, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + NEWLINE)

        fits = [ind["fit"] for ind in population]
        succ = sum(1 for ind in population if ind.get("res", {}).get("success"))
        print(
            "GEN %d best=%.1f med=%.1f worst=%.1f success=%d/%d elapsed=%.1fm"
            % (
                gen,
                fits[0],
                fits[len(fits) // 2],
                fits[-1],
                succ,
                len(fits),
                (time.time() - start) / 60.0,
            ),
            flush=True,
        )

        gen += 1
        elites = [ind["runs"] for ind in population[: args.elite]]
        next_pop = [
            {"runs": [r[:] for r in elites[i]], "fit": None, "res": None}
            for i in range(len(elites))
        ]
        for _ in range(max(2, args.pop // 8)):
            next_pop.append({
                "runs": mutate_runs(
                    population[0]["runs"], rng, rng.randint(1, 3)
                ),
                "fit": None,
                "res": None,
            })
        if args.cataclysm_every and gen % args.cataclysm_every == 0:
            print("CATACLYSM at gen %d" % gen, flush=True)
            for _ in range(max(4, args.pop // 8)):
                base = rng.choice(elites)
                next_pop.append({
                    "runs": cataclysm(base, rng),
                    "fit": None,
                    "res": None,
                })
        while len(next_pop) < args.pop:
            a = tournament(population, rng)
            b = tournament(population, rng)
            child = crossover(a["runs"], b["runs"], rng)
            child = mutate_runs(child, rng, rng.randint(1, 4))
            next_pop.append({"runs": child, "fit": None, "res": None})
        population = next_pop[: args.pop]

    print("DONE best fitness=%.1f" % best_overall["fitness"], flush=True)
    if best_overall["meta"]:
        print("BEST: %s" % json.dumps(best_overall["meta"]), flush=True)


if __name__ == "__main__":
    main()
