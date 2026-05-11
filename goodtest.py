import os
import time
import numpy as np
import runtime

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
# Set this to 1, 2, or 3 to switch load stages
CURRENT_STAGE = 3

stages = {
    1: {  # Target: 0.x seconds (Current scale)
        "FINE": {"count": 1000, "size": 50, "depth": 3, "branch": 10},
        "MEDIUM": {"count": 200, "size": 300, "depth": 3, "branch": 5},
        "COARSE": {"count": 15, "size": 1500, "depth": 2, "branch": 3}
    },
    2: {  # Target: 3 - 7 seconds
        "FINE": {"count": 8000, "size": 60, "depth": 3, "branch": 20},
        "MEDIUM": {"count": 1200, "size": 350, "depth": 4, "branch": 6},
        "COARSE": {"count": 40, "size": 2500, "depth": 3, "branch": 3}
    },
    3: {  # Target: > 10 seconds
        "FINE": {"count": 20000, "size": 70, "depth": 4, "branch": 15},
        "MEDIUM": {"count": 3000, "size": 450, "depth": 5, "branch": 5},
        "COARSE": {"count": 80, "size": 3500, "depth": 4, "branch": 3}
    }
}

cfg = stages[CURRENT_STAGE]


# ==========================================
# PAYLOAD & WRAPPERS
# ==========================================
def compute_payload(matrix_size):
    """CPU intensive task: O(N^3) complexity."""
    a = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size, matrix_size)
    return np.trace(np.dot(a, b))


def run_master_worker(task_count, matrix_size):
    active_tasks = []
    for _ in range(task_count):
        t = runtime.submit(compute_payload, matrix_size)
        if t: active_tasks.append(t)
    runtime.waitall()


def nested_worker(depth, branch_factor, matrix_size):
    compute_payload(matrix_size)
    if depth > 0:
        child_tasks = []
        for _ in range(branch_factor):
            t = runtime.submit(nested_worker, depth - 1, branch_factor, matrix_size)
            if t: child_tasks.append(t)
        runtime.waitall()


# ==========================================
# BENCHMARK ORCHESTRATOR
# ==========================================
def run_analysis():
    algo = os.getenv("TORCPY_SCHEDULING", "round_robin").upper()
    print(f"\n{'=' * 60}")
    print(f" ANALYSIS STAGE {CURRENT_STAGE} | ALGO: {algo}")
    print(f"{'=' * 60}")

    # Format: (Label, Function, Args)
    tests = [
        ("T1: Master-Worker | Fine  ", run_master_worker, (cfg["FINE"]["count"], cfg["FINE"]["size"])),
        ("T1: Master-Worker | Medium", run_master_worker, (cfg["MEDIUM"]["count"], cfg["MEDIUM"]["size"])),
        ("T1: Master-Worker | Coarse", run_master_worker, (cfg["COARSE"]["count"], cfg["COARSE"]["size"])),
        ("T2: Nested Trees  | Fine  ",
         lambda: runtime.submit(nested_worker, cfg["FINE"]["depth"], cfg["FINE"]["branch"], cfg["FINE"]["size"]), ()),
        ("T2: Nested Trees  | Medium",
         lambda: runtime.submit(nested_worker, cfg["MEDIUM"]["depth"], cfg["MEDIUM"]["branch"], cfg["MEDIUM"]["size"]),
         ()),
        ("T2: Nested Trees  | Coarse",
         lambda: runtime.submit(nested_worker, cfg["COARSE"]["depth"], cfg["COARSE"]["branch"], cfg["COARSE"]["size"]),
         ()),
    ]

    for label, func, args in tests:
        print(f"{label}...", end=" ", flush=True)
        start = time.time()

        if label.startswith("T2"):
            func()  # Launch root
            runtime.waitall()
        else:
            func(*args)

        elapsed = time.time() - start
        print(f"DONE in {elapsed:.4f}s")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    runtime.start(run_analysis, profile="cpu")