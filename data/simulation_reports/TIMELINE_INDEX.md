# Experimental Simulation Reports & Benchmarks — Chronological Timeline

This directory archives all formal simulation experiments, benchmarking trials, and high-resolution performance plots for the **AI-Based Traffic Light Management System**, ordered by experimental phase and evaluation date.

---

## 1. Experimental Overview & Summary Table

| Phase | Evaluation Date | Network Scope | Scenario Tested | Key Excel Report | Key Outcome / Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **2026-08-24 – 2026-08-25** | Single 4-Way Junction | Baseline vs. AI Optimization (1-Hour & 24-Hour full-day cycle) | [`24hour_benchmark_3modes_comparison.xlsx`](1_2026-08-24_Single_Junction/24hour_benchmark/24hour_benchmark_3modes_comparison.xlsx) | **42.8% reduction in Average Waiting Time (AWT)**; wait time variance reduced by 61.4%. |
| **Phase 2** | **2026-08-27** | 2-Junction Arterial Corridor | Isolated vs. Coordinated Green Wave (5-Hour rush period) | [`5hour_2junction_3methods_comparison.xlsx`](2_2026-08-27_Two_Junction_Corridor/5hour_2junction_3methods_comparison.xlsx) | **Arterial platoon synchronization** prevented queue spillback; increased corridor throughput by 19.3%. |
| **Phase 3** | **2026-09-17** | 4-Junction Grid Network | 2x2 Network Grid under distributed load balancing (5-Hour trial) | [`5hour_4junction_3methods_comparison.xlsx`](3_2026-09-17_Four_Junction_Grid/5hour_4junction_3methods_comparison.xlsx) | **Multi-junction coordination** maintained 100% deadlock prevention and stabilized max queue ceilings across all 4 intersections. |

---

## 2. Directory Breakdown

### [`1_2026-08-24_Single_Junction/`](1_2026-08-24_Single_Junction/)
- **`1hour_simulations/`**:
  - `simulation_1hour_U150_D40_L35_R30_BASELINE.xlsx`: Fixed-Time control under heavy demand.
  - `simulation_1hour_U150_D40_L35_R30_OPTIMIZED.xlsx`: Adaptive Neuro-Fuzzy control response.
  - `simulation_1hour_U150_D40_L35_R30.xlsx`: Comparative vehicle discharge logs.
- **`24hour_benchmark/`**:
  - `24hour_benchmark_3modes_comparison.xlsx`: Full 24-hour comparative analysis between Fixed-Time, Dynamic Cycle, and AI Hybrid mode.
  - `24hour_benchmark_3modes_comparison_v2.xlsx`: Supplementary validation run.
- **`figures/`**:
  - `figure1_hourly_awt_comparison.png`: Hourly Average Waiting Time curve across 24 hours.
  - `figure2_hourly_max_wait_comparison.png`: Hourly Maximum Waiting Time ceiling.
  - `figure3_wait_time_variance_boxplot.png`: Wait time distribution & variance boxplot.
  - `figure4_throughput_capacity_comparison.png`: Hourly vehicle discharge capacity comparison.

### [`2_2026-08-27_Two_Junction_Corridor/`](2_2026-08-27_Two_Junction_Corridor/)
- **Reports**:
  - `5hour_two_junction_benchmark_comparison.xlsx`: Initial 5-hour arterial synchronization test.
  - `5hour_2junction_3methods_comparison.xlsx`: 3-method benchmark (Fixed vs. Decentralized vs. Coordinated Green Wave).
- **`figures/`**:
  - `figure1_two_junction_5hour_awt.png` & `figure1_two_junction_hourly_awt.png`: Arterial waiting time progression.
  - `figure2_two_junction_5hour_max_wait.png` & `figure2_two_junction_max_wait_ceiling.png`: Maximum wait times along the corridor.
  - `figure3_two_junction_5hour_throughput.png` & `figure3_two_junction_total_throughput.png`: Arterial throughput comparison.

### [`3_2026-09-17_Four_Junction_Grid/`](3_2026-09-17_Four_Junction_Grid/)
- **Reports**:
  - `5hour_4junction_3methods_comparison.xlsx`: Comprehensive 5-hour 4-junction (J1, J2, J3, J4) multi-axis simulation.
- **`figures/`**:
  - `figure1_four_junction_5hour_awt.png`: 4-Junction grid network average waiting times.
  - `figure2_four_junction_5hour_max_wait.png`: Queue starvation ceiling across 16 approaches.
  - `figure3_four_junction_5hour_throughput.png`: Total grid throughput (vehicles successfully cleared).
