#!/usr/bin/env python3
"""Reproducible V28 campaign: matched controls, initial design, then GP expected improvement."""
import argparse, csv, json, math, pathlib, subprocess, sys
import numpy as np

BOUNDS = [
    (0.002, 0.012), (0.005, 0.040), (0.010, 0.060), (0.002, 0.025),
    (0.010, 0.040), (0.001, 0.015), (0.001, 0.020), (5.0, 50.0)
]
NAMES = ["p_ee","p_ei","p_ie","p_ii","weight","a_plus","a_minus","tau"]

def decode(x, seed, control="candidate"):
    return {"p_ee":float(x[0]),"p_ei":float(x[1]),"p_ie":float(x[2]),"p_ii":float(x[3]),
      "weight_min":float(x[4]),"weight_max":float(x[4]),"a_plus":float(x[5]),
      "a_minus":float(x[6]),"tau_pre":float(x[7]),"tau_post":float(x[7]),
      "max_delay":1,"seed":int(seed),"control":control}

def lhs(n, rng):
    d=len(BOUNDS); x=np.empty((n,d))
    for j,(lo,hi) in enumerate(BOUNDS):
        bins=(np.arange(n)+rng.random(n))/n
        rng.shuffle(bins); x[:,j]=lo+bins*(hi-lo)
    return x

def chol_solve(k, y):
    jitter=1e-8
    for _ in range(8):
        try:
            l=np.linalg.cholesky(k+np.eye(len(k))*jitter)
            return np.linalg.solve(l.T,np.linalg.solve(l,y))
        except np.linalg.LinAlgError: jitter*=10
    raise RuntimeError("GP covariance remained singular")

def propose(xs, ys, rng):
    lo=np.array([b[0] for b in BOUNDS]); hi=np.array([b[1] for b in BOUNDS])
    z=(xs-lo)/(hi-lo); best=float(np.max(ys))
    # Matérn-5/2 GP with fixed, normalized length scale and observation jitter.
    diff=z[:,None,:]-z[None,:,:]; r=np.sqrt(np.sum(diff*diff,axis=2)+1e-15)
    k=(1+math.sqrt(5)*r+5*r*r/3)*np.exp(-math.sqrt(5)*r)
    alpha=chol_solve(k+np.eye(len(k))*1e-6,ys-np.mean(ys))
    pool=rng.random((4096,len(BOUNDS)))
    delta=pool[:,None,:]-z[None,:,:]
    rr=np.sqrt(np.sum(delta*delta,axis=2)+1e-15)
    kk=(1+math.sqrt(5)*rr+5*rr*rr/3)*np.exp(-math.sqrt(5)*rr)
    mu=np.mean(ys)+kk@alpha
    v=1-np.sum(kk*chol_solve(k+np.eye(len(k))*1e-6,kk.T).T,axis=1)
    sigma=np.sqrt(np.maximum(v,1e-12))
    improvement=mu-best-0.01
    q=improvement/sigma
    # EI = delta*Phi(q) + sigma*phi(q), using erf without SciPy.
    phi=np.exp(-0.5*q*q)/math.sqrt(2*math.pi)
    Phi=np.array([0.5*(1+math.erf(float(a)/math.sqrt(2))) for a in q])
    ei=improvement*Phi+sigma*phi
    return pool[int(np.argmax(ei))]*(hi-lo)+lo

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exe",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--initial",type=int,default=6); ap.add_argument("--bayes",type=int,default=4)
    ap.add_argument("--seed",type=int,default=28001)
    a=ap.parse_args()
    out=pathlib.Path(a.output).resolve(); (out/"configs").mkdir(parents=True,exist_ok=True)
    (out/"results").mkdir(exist_ok=True)
    rng=np.random.default_rng(a.seed)
    baseline=np.array([.004,.010,.020,.010,.020,.005,.00525,20.0])
    jobs=[("control_baseline",baseline,"candidate"),
          ("control_untrained",baseline,"untrained"),
          ("control_shuffled_labels",baseline,"shuffled_labels")]
    jobs += [(f"initial_{i+1:02d}",x,"candidate") for i,x in enumerate(lhs(a.initial,rng))]
    records=[]; xs=[]; ys=[]
    def run(name,x,control):
        seed=a.seed+len(records)*101
        cfg=decode(x,seed,control)
        cp=out/"configs"/(name+".json"); rp=out/"results"/(name+".json")
        cp.write_text(json.dumps(cfg,indent=2)+"\n")
        subprocess.run([a.exe,str(cp),str(rp)],check=True)
        result=json.loads(rp.read_text())
        acc=float(result["accuracy"])
        # Penalize silent inhibition gently; hard integrity failures are rejected by evaluator.
        utility=acc-(3.0 if int(result["i_spikes"])==0 else 0.0)
        result["candidate"]=name; result["utility"]=utility
        records.append(result)
        if control=="candidate": xs.append(x.copy()); ys.append(utility)
        with (out/"experiments.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for row in records for k in row}))
            w.writeheader(); w.writerows(records)
        print(f"{name}: heldout={acc:.3f} class0={result['class0_correct']}/16 "
              f"class1={result['class1_correct']}/16 I_spikes={result['i_spikes']} "
              f"synapses={result['synapses']} utility={utility:.3f}",flush=True)
    for name,x,control in jobs: run(name,x,control)
    for i in range(a.bayes):
        x=propose(np.asarray(xs),np.asarray(ys),rng) if len(xs)>=3 else lhs(1,rng)[0]
        run(f"bayes_{i+1:02d}",x,"candidate")
    ranked=sorted((r for r in records if r["control"]=="candidate"),
                  key=lambda r:r["utility"],reverse=True)
    # Non-dominated accuracy / synapse-count archive.
    pareto=[]
    for r in ranked:
        if not any(q["accuracy"]>=r["accuracy"] and q["synapses"]<=r["synapses"] and
                   (q["accuracy"]>r["accuracy"] or q["synapses"]<r["synapses"]) for q in ranked):
            pareto.append(r)
    (out/"pareto_frontier.json").write_text(json.dumps(pareto,indent=2)+"\n")
    (out/"best_candidate.json").write_text(json.dumps(ranked[0] if ranked else {},indent=2)+"\n")
    print("\nCampaign complete:",out)
    print("Best candidate:",ranked[0]["candidate"] if ranked else "none")
    print("Pareto candidates:",", ".join(r["candidate"] for r in pareto))
if __name__=="__main__": main()
