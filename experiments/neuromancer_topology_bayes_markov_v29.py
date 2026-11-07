#!/usr/bin/env python3
"""V29 matched-seed topology comparison with Beta-Binomial and Dirichlet posteriors."""
import argparse, csv, json, pathlib, subprocess
import numpy as np

BASE = dict(p_ee=.004, p_ei=.010, p_ie=.020, p_ii=.010,
            weight_min=.020, weight_max=.020, a_plus=.005,
            a_minus=.00525, tau_pre=20.0, tau_post=20.0,
            max_delay=1, prune_below=.0025, growth_fraction=.01,
            ei_weight_scale=1.0)
COMPACT = dict(BASE, p_ee=.00268920, p_ei=.00715726, p_ie=.0161526,
               p_ii=.0131998, weight_min=.020, weight_max=.020,
               a_plus=.00130513, a_minus=.0182602, tau_pre=10.1057,
               tau_post=10.1057)
FAMILIES = {
    "fixed_baseline": (BASE, "fixed_baseline"),
    "compact_excitatory": (COMPACT, "fixed_baseline"),
    "inhibitory_expanded": (dict(BASE, p_ei=.030, p_ie=.040,
                                  ei_weight_scale=1.5), "fixed_baseline"),
    "structural_growth_pruning": (dict(BASE, growth_fraction=.025,
                                        prune_below=.006), "structural_growth"),
    "inhibitory_regulated_growth": (dict(BASE, growth_fraction=.025,
                                          prune_below=.006, p_ei=.020,
                                          p_ie=.030, ei_weight_scale=1.5),
                                    "inhibitory_regulated"),
}

def run_one(exe, out, name, seed, params, architecture, control="candidate"):
    cfg = dict(params, seed=int(seed), control=control, architecture=architecture)
    cp=out/"configs"/(name+".json"); rp=out/"results"/(name+".json")
    cp.write_text(json.dumps(cfg,indent=2)+"\n")
    subprocess.run([exe,str(cp),str(rp)],check=True)
    result=json.loads(rp.read_text())
    result["candidate"]=name
    result["architecture"]=architecture
    result["control"]=control
    return result

def beta_summary(successes, total, draws, rng):
    a=1+int(successes); b=1+int(total-successes)
    samples=rng.beta(a,b,size=draws)
    return {"alpha":a,"beta":b,"mean":float(a/(a+b)),
            "credible_95":[float(x) for x in np.quantile(samples,[.025,.975])],
            "posterior_samples":samples}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exe",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--seed",type=int,default=29001)
    ap.add_argument("--posterior-draws",type=int,default=100000)
    a=ap.parse_args()
    out=pathlib.Path(a.output).resolve()
    (out/"configs").mkdir(parents=True,exist_ok=True)
    (out/"results").mkdir(exist_ok=True)
    rng=np.random.default_rng(a.seed)
    records=[]
    seed_values=[a.seed+i*101 for i in range(a.seeds)]
    for family,(params,architecture) in FAMILIES.items():
        for j,seed in enumerate(seed_values):
            name=f"{family}_seed_{j+1:02d}"
            r=run_one(a.exe,out,name,seed,params,architecture)
            r["seed_index"]=j
            records.append(r)
            print(f"{name}: acc={r['accuracy']:.3f} c0={r['class0_correct']}/16 "
                  f"c1={r['class1_correct']}/16 I={r['i_spikes']} "
                  f"syn={r['synapses']} E_rate={r['e_rate_hz']:.3f} "
                  f"I_rate={r['i_rate_hz']:.3f}",flush=True)
    # Matched baseline controls on each seed; controls are never ranked as architectures.
    for j,seed in enumerate(seed_values):
        for control in ("untrained","shuffled_labels"):
            name=f"control_{control}_seed_{j+1:02d}"
            r=run_one(a.exe,out,name,seed,BASE,"fixed_baseline",control)
            r["seed_index"]=j
            records.append(r)
            print(f"{name}: acc={r['accuracy']:.3f} I={r['i_spikes']} syn={r['synapses']}",flush=True)
    with (out/"experiments.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in records for k in row
                                             if not isinstance(row[k],(dict,list))}))
        w.writeheader()
        for row in records: w.writerow({k:v for k,v in row.items() if k in w.fieldnames})
    candidate_rows=[r for r in records if r["control"]=="candidate"]
    by_family={}
    for family in FAMILIES:
        rows=[r for r in candidate_rows if r["candidate"].startswith(family+"_")]
        successes=sum(int(r["class0_correct"])+int(r["class1_correct"]) for r in rows)
        total=32*len(rows)
        post=beta_summary(successes,total,a.posterior_draws,rng)
        post["per_seed_accuracy"]=[float(r["accuracy"]) for r in rows]
        post["mean_synapses"]=float(np.mean([r["synapses"] for r in rows]))
        post["sd_synapses"]=float(np.std([r["synapses"] for r in rows],ddof=1)) if len(rows)>1 else 0.0
        post["mean_i_spikes"]=float(np.mean([r["i_spikes"] for r in rows]))
        post["mean_e_rate_hz"]=float(np.mean([r["e_rate_hz"] for r in rows]))
        post["mean_i_rate_hz"]=float(np.mean([r["i_rate_hz"] for r in rows]))
        post["mean_dropped_events"]=float(np.mean([r["dropped_events"] for r in rows]))
        post["finite_all"]=all(r["finite_state"] for r in rows)
        post["all_seeds_classwise_pass"]=all(r["class0_correct"]>=12 and r["class1_correct"]>=12 for r in rows)
        # Pool edge-state transitions with a symmetric Dirichlet(1) prior.
        matrices=np.zeros((3,3),dtype=np.int64); born=dead=pruned=grown=0
        for r in rows:
            for ep in r.get("epoch_history",[]):
                mat=np.asarray(ep["edge_markov"]["transition_counts"],dtype=np.int64)
                matrices+=mat; born+=int(ep["edge_markov"]["born"]); dead+=int(ep["edge_markov"]["dead"])
                pruned+=int(ep["structural_update"]["pruned"]); grown+=int(ep["structural_update"]["grown"])
        row_post=[]
        for i in range(3):
            alpha=matrices[i].astype(float)+1.0
            row_post.append({"alpha":[float(x) for x in alpha],
                             "posterior_mean":[float(x) for x in alpha/alpha.sum()]})
        post["edge_markov"]={"state_labels":["weak","medium","strong"],
                             "transition_counts":matrices.tolist(),
                             "dirichlet_posterior_rows":row_post,
                             "born":born,"dead":dead,"pruned":pruned,"grown":grown}
        by_family[family]=post
    base_draw=by_family["fixed_baseline"]["posterior_samples"]
    comparisons={}
    for family,post in by_family.items():
        draws=post.pop("posterior_samples")
        comparisons[family]={"posterior_probability_accuracy_gt_baseline":
            float(np.mean(draws>base_draw)) if family!="fixed_baseline" else .5,
            "posterior_probability_accuracy_ge_0_75":float(np.mean(draws>=.75))}
    summary={"experiment":"Neuromancer V29 topology architecture comparison",
             "seed_values":seed_values,"families":by_family,
             "bayesian_comparisons":comparisons,
             "controls":{c:[r["accuracy"] for r in records if r["control"]==c]
                         for c in ("untrained","shuffled_labels")},
             "interpretation_note":"Beta posteriors use classwise correct counts over 32 held-out trials per seed. Markov rows use Dirichlet(1) priors over coarse edge-weight states. These are benchmark-specific statistical summaries, not proof of biological mechanisms."}
    (out/"bayesian_markov_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    leaderboard=sorted(by_family,key=lambda k:(by_family[k]["all_seeds_classwise_pass"],
        by_family[k]["mean"],by_family[k]["mean_i_spikes"]>0,by_family[k]["mean_synapses"]*-1),reverse=True)
    (out/"architecture_leaderboard.json").write_text(json.dumps(
        [{"family":k,**{x:y for x,y in by_family[k].items() if x!="edge_markov"}}
         for k in leaderboard],indent=2)+"\n")
    print("\nV29 complete:",out)
    print("Bayesian/Markov report:",out/"bayesian_markov_summary.json")
    for k in leaderboard:
        p=by_family[k]
        print(f"{k}: acc posterior mean={p['mean']:.3f} "
              f"synapses={p['mean_synapses']:.0f} I spikes={p['mean_i_spikes']:.1f} "
              f"all-seed classwise pass={p['all_seeds_classwise_pass']}")

if __name__=="__main__": main()
