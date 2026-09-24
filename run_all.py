"""Runs every experiment reported in the manuscript and writes results/results.json.

Usage:  python run_all.py --csv /path/to/US_Accidents_March23.csv [--out results] [--quick]
"""
import argparse, json, os, platform, time, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from fedhsa.config import Config, REGIONS, TIME_BINS, FEATURES
from fedhsa.data import load_records, build_contexts, seasonal_split, forward_split
from fedhsa.training import Structure, train_central, train_federated, predict, fit_sklearn
from fedhsa.models import FedHSA, MLP, HGNN, HGNNPlus, GCN, GAT, GCNII
from fedhsa.stats import metrics, best_threshold, delong, mcnemar, auc
from fedhsa import explain as EX

def save(res, cfg):
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "results.json"), "w") as f: json.dump(res, f, indent=1, default=float)

def evaluate(D, p):
    thr = best_threshold(D["y"][D["val"]], p[D["val"]])
    return metrics(D["y"][D["test"]], p[D["test"]], thr)

def run_models(D, S, cfg, seed, aug=0.0, which=None):
    """Returns {name: (probabilities for all nodes, is_federated, model_or_clf)}."""
    torch.manual_seed(seed); np.random.seed(seed); out = {}
    def want(n): return which is None or n in which
    if want("Logistic regression"):
        c = fit_sklearn(LogisticRegression(max_iter=2000, class_weight="balanced"), D, aug, seed); out["Logistic regression"] = (c.predict_proba(D["X"])[:, 1], False, c)
    if want("Random forest"):
        c = fit_sklearn(RandomForestClassifier(100, class_weight="balanced", random_state=seed, n_jobs=-1), D, aug, seed); out["Random forest"] = (c.predict_proba(D["X"])[:, 1], False, c)
    if want("XGBoost"):
        spw = float((D["y"][D["train"]] == 0).sum() / max((D["y"][D["train"]] == 1).sum(), 1))
        c = fit_sklearn(XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                      scale_pos_weight=spw, random_state=seed, n_jobs=-1, eval_metric="auc"), D, aug, seed)
        out["XGBoost"] = (c.predict_proba(D["X"])[:, 1], False, c)
    for name, ctor in [("MLP", MLP), ("GCN", GCN), ("GAT", GAT), ("GCNII", GCNII), ("HGNN", HGNN), ("HGNN+", HGNNPlus)]:
        if want(name):
            m, _ = train_central(ctor(), D, S, cfg, aug_sigma=aug, seed=seed); out[name] = (predict(m, D, S), False, m)
    for name, mode in [("FedAvg (HGNN)", "fedavg"), ("FedProx (HGNN)", "fedprox"), ("SCAFFOLD (HGNN)", "scaffold")]:
        if want(name):
            m, _ = train_federated(HGNN(), D, S, cfg, mode=mode, seed=seed, aug_sigma=aug); out[name] = (predict(m, D, S, federated=True), True, m)
    if want("FedHSA"):
        m, h = train_federated(FedHSA(cfg), D, S, cfg, mode="hier_ramal", seed=seed); out["FedHSA"] = (predict(m, D, S, federated=True), True, m); out["_fedhsa_hist"] = h
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="tiny settings to check the pipeline end to end")
    ap.add_argument("--skip", default="", help="comma-separated sections to skip: forward,sens,robust,faith")
    a = ap.parse_args(); cfg = Config(csv=a.csv, out=a.out); skip = set(a.skip.split(",")) if a.skip else set()
    if a.quick:
        cfg.seeds, cfg.rounds, cfg.central_epochs, cfg.patience = [0], 3, 15, 5
        cfg.alpha_grid, cfg.beta_grid, cfg.rounds_grid, cfg.sens_seeds = [0.3, 0.6], [0.9, 0.95], [2, 3], [0]
        cfg.noise_draws, cfg.shap_nodes, cfg.shap_nsamples = 2, 5, 40
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    res = dict(config=cfg.__dict__, regions=REGIONS, time_bins=TIME_BINS, features=FEATURES,
               environment=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                                processor=platform.processor(), cpu_count=os.cpu_count(), cuda=torch.cuda.is_available()))
    t0 = time.time()
    df, audit = load_records(cfg.csv); res["data_audit"] = audit; print("audit", {k: v for k, v in audit.items() if k not in ("states", "years")})
    ctx = build_contexts(df); res["n_contexts"] = int(len(ctx)); S = Structure(ctx); save(res, cfg)

    # ---------------- main comparison (Tables I-II) ----------------
    res["main"] = {}; res["significance"] = {}; res["convergence"] = []; all_models = {}
    for seed in cfg.seeds:
        D = seasonal_split(ctx, seed, cfg.val_frac, cfg.label_top); res.setdefault("split_info", {})[seed] = D["info"]
        out = run_models(D, S, cfg, seed); res["convergence"].append(out.pop("_fedhsa_hist"))
        res["main"][seed] = {n: evaluate(D, p) for n, (p, _, _) in out.items()}
        pf = out["FedHSA"][0]; thr_f = res["main"][seed]["FedHSA"]["threshold"]; yt = D["y"][D["test"]]
        res["significance"][seed] = {}
        for n, (p, _, _) in out.items():
            if n == "FedHSA": continue
            _, _, z, pd_ = delong(yt, pf[D["test"]], p[D["test"]])
            mc = mcnemar(yt, (pf[D["test"]] >= thr_f).astype(int), (p[D["test"]] >= res["main"][seed][n]["threshold"]).astype(int))
            res["significance"][seed][n] = dict(delong_z=z, delong_p=pd_, mcnemar=mc)
        all_models[seed] = out
        if seed == cfg.seeds[0]:
            D0, fed0 = D, out["FedHSA"][2]
        save(res, cfg); print(f"seed {seed} done", {n: round(v["auc"], 4) for n, v in res["main"][seed].items()})

    # ---------------- efficiency ----------------
    n_par = sum(p.numel() for p in fed0.parameters())
    tt = time.perf_counter(); [predict(fed0, D0, S, federated=True) for _ in range(20)]; lat = (time.perf_counter() - tt) / 20
    res["efficiency"] = dict(parameters=int(n_par), bytes_per_update=int(n_par * 4),
                             hier_messages_per_round=49 + 6, hier_bytes_per_round=int(n_par * 4 * (49 + 6)),
                             flat_messages_per_round=49, flat_bytes_per_round=int(n_par * 4 * 49),
                             cloud_updates_per_round_hier=6, cloud_updates_per_round_flat=49,
                             inference_seconds_all_contexts=lat)
    save(res, cfg)

    # ---------------- ablation (Table V) ----------------
    res["ablation"] = {}
    variants = {"w/o residual term": dict(kw=dict(use_residual=False)), "w/o hypergraph": dict(kw=dict(use_hypergraph=False)),
                "w/o spiking layer": dict(kw=dict(use_spiking=False)), "w/o RAMAL adaptation": dict(kw={}, mode="hier_fixed"),
                "w/o federation": dict(kw={}, central=True)}
    for seed in cfg.seeds:
        D = seasonal_split(ctx, seed, cfg.val_frac, cfg.label_top); res["ablation"][seed] = {}
        for vn, spec in variants.items():
            model = FedHSA(cfg, **spec["kw"])
            if spec.get("central"):
                model, _ = train_central(model, D, S, cfg, ramal=True, seed=seed); p = predict(model, D, S)
            else:
                model, h = train_federated(model, D, S, cfg, mode=spec.get("mode", "hier_ramal"), seed=seed); p = predict(model, D, S, federated=True)
                if vn == "w/o residual term": res["ablation"][seed]["_resid_diag"] = dict(prob_min=float(p.min()), prob_max=float(p.max()), prob_std=float(p.std()), train_loss=[x["train_loss"] for x in h])
            res["ablation"][seed][vn] = evaluate(D, p)
        save(res, cfg); print("ablation seed", seed)

    # ---------------- sensitivity (Table VI) ----------------
    if "sens" not in skip:
        res["sensitivity"] = dict(alpha={}, beta={}, rounds={})
        for seed in cfg.sens_seeds:
            D = seasonal_split(ctx, seed, cfg.val_frac, cfg.label_top); yv = D["y"][D["val"]]
            for al in cfg.alpha_grid:
                m, _ = train_federated(FedHSA(cfg, alpha=al), D, S, cfg, seed=seed)
                res["sensitivity"]["alpha"].setdefault(str(al), []).append(auc(yv, predict(m, D, S, federated=True)[D["val"]]))
            for be in cfg.beta_grid:
                m, _ = train_federated(FedHSA(cfg, beta=be), D, S, cfg, seed=seed)
                res["sensitivity"]["beta"].setdefault(str(be), []).append(auc(yv, predict(m, D, S, federated=True)[D["val"]]))
            for R in cfg.rounds_grid:
                m, h = train_federated(FedHSA(cfg), D, S, cfg, rounds=R, seed=seed)
                res["sensitivity"]["rounds"].setdefault(str(R), []).append(dict(val_auc=auc(yv, predict(m, D, S, federated=True)[D["val"]]), final_train_loss=h[-1]["train_loss"]))
            save(res, cfg)

    # ---------------- robustness (Fig. 4) ----------------
    if "robust" not in skip:
        res["robustness"] = {}
        names = ["FedHSA", "HGNN", "GAT", "GCN", "Random forest", "Logistic regression", "XGBoost"]
        for key, aug in (("baselines_no_aug", 0.0), ("baselines_with_aug", cfg.sigma_r)):
            res["robustness"][key] = {}
            for seed in cfg.seeds:
                D = seasonal_split(ctx, seed, cfg.val_frac, cfg.label_top)
                if aug == 0:
                    out = all_models[seed]
                else:   # retrain baselines with the same Gaussian augmentation; FedHSA keeps its own L_rob
                    out = run_models(D, S, cfg, seed, aug=aug, which=set(names) - {"FedHSA"})
                    out["FedHSA"] = all_models[seed]["FedHSA"]
                rng = np.random.default_rng(seed)
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
                    res["robustness"][key].setdefault(n, []).append(rows)
                save(res, cfg)

    # ---------------- faithfulness (Table IV) ----------------
    if "faith" not in skip:
        D, m = D0, fed0; ours = EX.intrinsic(m, D, S)
        shp = EX.kernel_shap(m, D, S, cfg.shap_nodes, cfg.shap_nsamples); ig = EX.integrated_gradients(m, D, S, cfg.shap_nodes)
        rng = np.random.default_rng(0)
        rankings = {"Spike rate x attention": ours, "KernelSHAP": shp, "Integrated gradients": ig}
        faith = dict(importance={k: v.tolist() for k, v in rankings.items()}, full_auc=res["main"][cfg.seeds[0]]["FedHSA"]["auc"], methods={})
        for k, imp in rankings.items():
            idx = EX.top_k(imp); D2 = EX.remove_features(D, idx)
            m2, _ = train_federated(FedHSA(cfg), D2, S, cfg, seed=cfg.seeds[0])
            faith["methods"][k] = dict(top_features=EX.feature_names(idx), tau_vs_shap=EX.rank_agreement(imp, shp),
                                       auc_after_removal=auc(D["y"][D["test"]], predict(m2, D2, S, federated=True)[D["test"]]))
        rand = []
        for r in range(3):
            idx = list(rng.choice(6, 2, replace=False)); D2 = EX.remove_features(D, idx)
            m2, _ = train_federated(FedHSA(cfg), D2, S, cfg, seed=cfg.seeds[0]); rand.append(auc(D["y"][D["test"]], predict(m2, D2, S, federated=True)[D["test"]]))
        faith["methods"]["Random ranking"] = dict(top_features=None, tau_vs_shap=None, auc_after_removal=float(np.mean(rand)))
        res["faithfulness"] = faith; save(res, cfg)

    # ---------------- forward-chaining protocol ----------------
    if "forward" not in skip:
        cy = build_contexts(df, by_year=True); Sy = Structure(cy); res["forward"] = {}
        for seed in cfg.seeds:
            Dy = forward_split(cy, seed, cfg.val_frac, cfg.label_top)
            out = run_models(Dy, Sy, cfg, seed, which={"FedHSA", "HGNN", "XGBoost", "Logistic regression"}); out.pop("_fedhsa_hist", None)
            res["forward"][seed] = {n: evaluate(Dy, p) for n, (p, _, _) in out.items()}; res["forward"][seed]["_info"] = Dy["info"]
            save(res, cfg)
    res["total_minutes"] = (time.time() - t0) / 60; save(res, cfg); print("finished in %.1f min" % res["total_minutes"])

if __name__ == "__main__":
    main()
