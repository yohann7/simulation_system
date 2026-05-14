"""Small-scale optimization test: PopSize=8, MaxIter=5."""
import os
import sys
import time
import numpy as np

# Ensure project modules are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "data_driven"))
sys.path.insert(0, os.path.join(_project_root, "sim_T"))

import PO


def main():
    MaxIter = 12
    PopSize = 15

    lb, ub, dim = PO.get_search_bounds()

    t_start = time.perf_counter()
    best_pos, best_score, convergence = PO.puma_optimize(
        PopSize,
        MaxIter,
        lb,
        ub,
        dim,
        PO.stelmor_sim_CostFunction,
        BatchCostFunction=PO.stelmor_batch_cost_function,
    )
    elapsed = time.perf_counter() - t_start

    print()
    print("=" * 60)
    print(f"OPTIMIZATION COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Best Score: {best_score:.4f}")
    print(f"ORT:     {best_pos[0]:.0f} °C")
    for i in range(1, 11):
        print(f"SPEED{i:2d}: {best_pos[i]:.3f} m/s")
    for i in range(11, 17):
        print(f"FAN{i-10}:    {best_pos[i]:.0f} %")
    print(f"Convergence: {[f'{c:.3f}' for c in convergence]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
