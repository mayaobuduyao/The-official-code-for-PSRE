"""Master runner: executes all RQs and the case study, then aggregates."""
import sys, os, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCRIPTS = ["run_rq1.py", "run_rq2.py", "run_rq3.py", "run_rq4.py",
           "run_rq5.py", "run_casestudy.py", "run_gpu_throughput.py"]


def main():
    os.chdir(ROOT)
    t0 = time.time()
    for s in SCRIPTS:
        print(f"\n{'='*60}\n# {s}\n{'='*60}")
        r = subprocess.run([sys.executable, os.path.join("experiments", s)],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print("STDERR:", r.stderr[-2000:])
            print(f"!! {s} FAILED")
            sys.exit(1)
    print(f"\nAll experiments completed in {time.time()-t0:.1f}s")

    # aggregate
    agg = {}
    for name in ["rq1", "rq2", "rq3", "rq4", "rq5", "casestudy",
                 "gpu_throughput"]:
        p = f"results/{name}.json"
        if os.path.exists(p):
            with open(p) as f:
                agg[name] = json.load(f)
    with open("results/all.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("Aggregated -> results/all.json")


if __name__ == "__main__":
    main()