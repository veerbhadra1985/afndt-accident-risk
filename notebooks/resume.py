"""Continue an interrupted run_all.py: runs the sections that are still missing and skips finished work.
Usage: python resume.py --csv FILE --out RESULTS_DIR --sections robust,faith,forward,sens
Safe to run again after a disconnect: each finished piece is recorded in results.json."""
import argparse, json, os, time, numpy as np, torch
from fedhsa.config import Config
from fedhsa.data import load_records, build_contexts, seasonal_split, forward_split
from fedhsa.training import Structure, train_federated, predict
from fedhsa.models import FedHSA
from fedhsa.stats import auc
from fedhsa import explain as EX
from run_all import run_models, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sections", default="robust,faith,forward,sens")
a = ap.parse_args(); cfg = Config(csv=a.csv, out=a.out); secs = a.sections.split(",")
path = os.path.join(a.out, "results.json"); res = json.load(open(path))
assert "main" in res and "ablation" in res, "results.json must already contain the main and ablation results"

def save():
    tmp = path + ".tmp"; json.dump(res, open(tmp, "w"), indent=1, default=float); os.replace(tmp, path)
done = set(res.setdefault("_resume_done", []))
def mark(tag):
    res["_resume_done"].append(tag); done.add(tag); save(); print("finished:", tag, flush=True)

# sensitivity results saved by the interrupted run (whole seeds only) are kept and not repeated
if "sensitivity" in res and not any(t.startswith("sens|") for t in done):
    for p in ("alpha", "beta", "rounds"):
        for v, lst in res["sensitivity"].get(p, {}).items():
            for s in cfg.sens_seeds[:len(lst)]:
                res["_resume_done"].append(f"sens|{p}|{v}|{s}"); done.add(f"sens|{p}|{v}|{s}")

torch.set_num_threads(max(1, os.cpu_count() or 1)); t0 = time.time()
df, _ = load_records(cfg.csv); ctx = build_contexts(df); S = Structure(ctx); print("data loaded", flush=True)

if "robust" in secs:
    names = ["FedHSA", "HGNN", "GAT", "GCN", "Random forest", "Logistic regression", "XGBoost"]
    R = res.setdefault("robustness", {}); fed_cache = {}
    for key, aug in (("baselines_no_aug", 0.0), ("baselines_with_aug", cfg.sigma_r)):
        R.setdefault(key, {})
        for seed in cfg.seeds:
            tag = f"robust|{key}|{seed}"
            if tag in done: continue
            D = seasonal_split(ctx, seed, cfg.val_frac, cfg.label_top)
            if aug == 0:
                out = run_models(D, S, cfg, seed, which=set(names))
            else:
                out = run_models(D, S, cfg, seed, aug=aug, which=set(names) - {"FedHSA"})
                if seed not in fed_cache:
                    fed_cache[seed] = run_models(D, S, cfg, seed, which={"FedHSA"})["FedHSA"]
                out["FedHSA"] = fed_cache[seed]
            out.pop("_fedhsa_hist", None)
            if aug == 0: fed_cache[seed] = out["FedHSA"]
            rng = np.random.default_rng(seed); new = {}
            for n in names:
                _, fed, mdl = out[n]; rows = {}
                for sg in [0.0] + cfg.noise_sigmas:
                    aucs = []
                    for _ in range(1 if sg == 0 else cfg.noise_draws):
                        Xn = D["X"].copy()
                        if sg: Xn[:, [4, 5]] += rng.normal(0, sg, (len(Xn), 2)).astype(np.float32)
                        p = mdl.predict_proba(Xn)[:, 1] if hasattr(mdl, "predict_proba") else predict(mdl, D, S, X=Xn, federated=fed)
                        aucs.append(auc(D["y"][D["test"]], p[D["test"]]))
                    rows[str(sg)] = float(np.mean(aucs))
                new[n] = rows
            for n, rows in new.items(): R[key].setdefault(n, []).append(rows)
            mark(tag)

if "faith" in secs and "faith" not in done:
    s0 = cfg.seeds[0]; D = seasonal_split(ctx, s0, cfg.val_frac, cfg.label_top)
    m, _ = train_federated(FedHSA(cfg), D, S, cfg, seed=s0); yt = D["y"][D["test"]]
    full = auc(yt, predict(m, D, S, federated=True)[D["test"]]); print("faith: model trained", flush=True)
    ours = EX.intrinsic(m, D, S); shp = EX.kernel_shap(m, D, S, cfg.shap_nodes, cfg.shap_nsamples)
    ig = EX.integrated_gradients(m, D, S, cfg.shap_nodes); print("faith: SHAP and IG done", flush=True)
    rankings = {"Spike rate x attention": ours, "KernelSHAP": shp, "Integrated gradients": ig}
    F = dict(importance={k: v.tolist() for k, v in rankings.items()}, full_auc=full, methods={})
    for k, imp in rankings.items():
        idx = EX.top_k(imp); D2 = EX.remove_features(D, idx)
        m2, _ = train_federated(FedHSA(cfg), D2, S, cfg, seed=s0)
        F["methods"][k] = dict(top_features=EX.feature_names(idx), tau_vs_shap=EX.rank_agreement(imp, shp),
                               auc_after_removal=auc(yt, predict(m2, D2, S, federated=True)[D["test"]]))
    rng = np.random.default_rng(0); rand = []
    for _ in range(3):
        idx = [int(i) for i in rng.choice(6, 2, replace=False)]; D2 = EX.remove_features(D, idx)
        m2, _ = train_federated(FedHSA(cfg), D2, S, cfg, seed=s0); rand.append(auc(yt, predict(m2, D2, S, federated=True)[D["test"]]))
    F["methods"]["Random ranking"] = dict(top_features=None, tau_vs_shap=None, auc_after_removal=float(np.mean(rand)))
    res["faithfulness"] = F; mark("faith")

if "forward" in secs:
    cy = build_contexts(df, by_year=True); Sy = Structure(cy); res.setdefault("forward", {})
    for seed in cfg.seeds:
        tag = f"forward|{seed}"
        if tag in done: continue
        Dy = forward_split(cy, seed, cfg.val_frac, cfg.label_top)
        out = run_models(Dy, Sy, cfg, seed, which={"FedHSA", "HGNN", "XGBoost", "Logistic regression"}); out.pop("_fedhsa_hist", None)
        r = {n: evaluate(Dy, p) for n, (p, _, _) in out.items()}; r["_info"] = Dy["info"]
        res["forward"][str(seed)] = r; mark(tag)

if "sens" in secs:
    Sn = res.setdefault("sensitivity", dict(alpha={}, beta={}, rounds={}))
    for seed in cfg.sens_seeds:
        D = seasonal_split(ctx, seed, cfg.val_frac, cfg.label_top); yv = D["y"][D["val"]]
        for p, grid in (("alpha", cfg.alpha_grid), ("beta", cfg.beta_grid), ("rounds", cfg.rounds_grid)):
            for v in grid:
                tag = f"sens|{p}|{v}|{seed}"
                if tag in done: continue
                if p == "rounds":
                    m, h = train_federated(FedHSA(cfg), D, S, cfg, rounds=v, seed=seed)
                    val = dict(val_auc=auc(yv, predict(m, D, S, federated=True)[D["val"]]), final_train_loss=h[-1]["train_loss"])
                else:
                    m, _ = train_federated(FedHSA(cfg, **{p: v}), D, S, cfg, seed=seed)
                    val = auc(yv, predict(m, D, S, federated=True)[D["val"]])
                Sn[p].setdefault(str(v), []).append(val); mark(tag)

res["total_minutes"] = res.get("total_minutes", 0) + (time.time() - t0) / 60; save()
print("ALL REQUESTED SECTIONS FINISHED")
