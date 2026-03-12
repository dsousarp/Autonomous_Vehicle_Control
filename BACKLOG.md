# DeePC Autonomous Vehicle Control — Backlog

This backlog reflects the project's trajectory toward a **USV (Unmanned Surface Vehicle)**
use case. Items are grouped by theme and ordered by priority within each group.
Regularization (`lambda_g`, `lambda_y`) is treated as a robustness margin, not as the
primary nonlinearity mitigation strategy — the items below address the root causes directly.

---

## Priority 1 — Robustness to Nonlinearity and Disturbances

These are the highest-value additions for the USV case. Regularization alone cannot
bridge regime changes, structured ocean disturbances, or dynamics that drift outside
the offline data trajectory.

### 1.1 Online Sliding Hankel Window
**Why:** The offline Hankel matrix captures dynamics at one operating point. As sea state,
heading, and speed change, the Hankel matrix must track current conditions.

- Replace the fixed offline dataset with a rolling buffer of the most recent `T_data` samples
- Re-build (or rank-1 update) the Hankel matrices after each control step
- Add a persistent excitation monitor — warn or pause updates if the incoming data is
  insufficiently rich
- Tune the window length as a trade-off: too short loses diversity, too long retains
  stale dynamics

### 1.2 Disturbance Feedforward Estimator
**Why:** Wind, current, and waves are persistent and correlated — not zero-mean noise.
`lambda_y` slack absorbs them as model mismatch instead of compensating for them.

- At each step, compute the residual between DeePC's predicted output and the actual
  measured output
- Fit a simple first-order disturbance model (e.g. constant or sinusoidal) to the
  residual history
- Inject the estimated disturbance as a known offset into the QP reference trajectory
- Validate against a simulated steady beam current and head sea

### 1.3 Regime-Aware Regularization Scheduling
**Why:** A single `lambda_g` value is a blunt instrument. Low speed/low sea state needs
less regularization (trust the data); high speed or turning needs more.

- Define operating regimes by speed and turning rate: {low-speed straight, cruise
  straight, turning, high-sea-state}
- Schedule `lambda_g` and `lambda_y` as lookup tables over the operating point
- Compare RMSE vs. fixed regularization across stress tests

---

## Priority 2 — Computational Performance

At ~390 ms per step with Ts = 0.1 s, the controller is 4x over budget for real-time
deployment. This must be resolved before any hardware trials.

### 2.1 Swap CLARABEL → OSQP
- Replace the CLARABEL solver with OSQP, which is typically 5–10x faster for this
  problem class and supports warm-starting natively
- Keep SCS as the fallback
- Re-benchmark solve times across the 8-scenario stress suite

### 2.2 Sparse Hankel Exploitation
- Investigate sparsity structure in `U_p`, `U_f`, `Y_p`, `Y_f` blocks
- Pass explicit sparsity hints to the solver
- Profile whether Hankel construction or QP solve dominates runtime

### 2.3 Horizon Sensitivity Study
- Sweep `N` ∈ {5, 10, 15, 20} and `Tini` ∈ {2, 3, 5} against RMSE and solve time
- Identify the Pareto-optimal (N, Tini) pair for the USV scenario
- Document the result in the version README

---

## Priority 3 — Constraint Handling

### 3.1 Output Constraints (Obstacle / Geofence Avoidance)
- Add convex spatial constraints on `y_f` (e.g. half-plane exclusion zones)
- Test with static obstacles placed on the sinusoidal reference path
- Extend to a simple velocity-obstacle formulation for moving targets (COLREGS-aware)

### 3.2 Input Rate Constraints
- Add `delta_dot_max` and `a_dot_max` constraints (rate of change of steering and
  acceleration)
- These are physically necessary for actuator protection on a real vessel
- Reformulate as constraints on `u_f[k+1] - u_f[k]`

### 3.3 Soft vs. Hard Constraint Audit
- Document which constraints are hard (always satisfied) vs. soft (penalized violation)
- For the USV case, geofence and actuator limits should be hard; output tracking should
  remain soft

### 3.4 Solver Failure Fallback
- Handle solver failures gracefully with a fallback control law (e.g. hold previous input,
  safe stop, or simple PID takeover)

---

## Priority 4 — Plant and Scenario Upgrades

### 4.1 Upgrade to USV Dynamic Model (Fossen 3-DOF)
- Replace the kinematic bicycle model with a Fossen-style surface vessel:
  `M * nu_dot + C(nu) * nu + D(nu) * nu = tau + tau_env`
- States: surge `u`, sway `v`, yaw rate `r`
- Outputs: `x`, `y`, heading `psi` (or keep heading hidden for realism)
- This increases the nonlinearity the DeePC must handle and stress-tests Priority 1 items

### 4.2 Environmental Disturbance Simulation
- Add steady current (constant force offset in world frame)
- Add wind (speed + direction, mapped to force via Beaufort approximation)
- Add wave disturbance (first-order Markov process or JONSWAP spectrum)
- Use these as the primary benchmark for the disturbance feedforward estimator (1.2)

### 4.3 Waypoint-Following Reference Generator
- Replace the sinusoidal reference with a waypoint list and LOS (Line-of-Sight) guidance
- Generate smooth `(x_ref, y_ref, v_ref)` trajectories from waypoints using cubic splines
- This is the reference format needed for real mission execution

### 4.4 Additional Reference Trajectories
- Lane change, figure-8, parking maneuvers (for land vehicle benchmarks)

---

## Priority 5 — Baselines and Benchmarking

### 5.1 PID Baseline
- Implement a decoupled PID for lateral (heading error) and longitudinal (speed error)
  control
- Run identical stress suite scenarios
- Report RMSE, constraint violations, and compute time side-by-side with DeePC

### 5.2 MPC Baseline (Model-Based)
- Implement a linear MPC using the linearized bicycle / Fossen model at the nominal
  operating point
- This establishes the ceiling: how much does knowing the model help?
- Gap between MPC and DeePC quantifies the cost of model-free operation

### 5.3 Automated Version Comparison Dashboard
- Extend `comparison/compare_versions.py` to include all metrics from the above baselines
- Generate a single HTML or PDF report with trajectory overlays, RMSE tables, and
  solve-time distributions

---

## Priority 6 — Data Collection and Dataset Management

### 6.1 Persistent Excitation Verification (Quantitative)
- Compute and log the condition number of the Hankel matrix after data collection
- Warn if below a configurable threshold
- Expose this as a pre-flight check for hardware deployment

### 6.2 Dataset Versioning
- Save offline datasets as `.npz` with metadata (timestamp, excitation type, vessel
  config, sea state)
- Allow the controller to load a pre-collected dataset instead of re-collecting at startup
- Useful for hardware where excitation maneuvers are costly

### 6.3 Multi-Trajectory Dataset Merging
- Collect multiple short excitation trajectories and merge their Hankel contributions
- Improves coverage of the operating envelope without one very long excitation run

### 6.4 Investigate Effect of T_data on Tracking Quality
- Sweep `T_data` values and measure impact on RMSE and solver conditioning

---

## Priority 7 — Code Quality

- [ ] Unit tests for Hankel construction, PE check, and simulator
- [ ] Integration test: run full simulation and assert tracking error bounds
- [ ] Type checking with mypy
- [ ] CI pipeline (GitHub Actions)
- [ ] Logging instead of print statements

---

## Future Extensions

These are longer-term directions beyond the current USV focus:

- [ ] **DeePC-Hunt** — hybrid DeePC formulation combining data-driven and model-based elements (Lygeros et al.)
- [ ] **Multi-domain vehicles** — extend to air (quadrotor/UAV) vehicle models beyond USV
- [ ] **Navigation + tracking** — add path planning / navigation layer on top of the trajectory tracking controller
- [ ] **Nonlinear DeePC variants** — explore formulations that relax the LTI assumption directly
- [ ] **MIMO validation** — test with higher-dimensional systems
- [ ] Real-time plotting / live simulation visualization
- [ ] Integration with ROS2 or CARLA simulator

---

## Known Issues

| ID | Description | Severity | Affects |
|----|-------------|----------|---------|
| K1 | Solve time ~390ms exceeds Ts=0.1s realtime budget | High | All versions |
| K2 | Willems' lemma assumes LTI; bicycle model is nonlinear | Medium | v1_baseline |
| K3 | No disturbance compensation; steady currents cause tracking bias | Medium | v1_baseline |
| K4 | T_data=200 is only ~1.9x the PE minimum; marginal for nonlinear regimes | Low | v1_baseline |
| K5 | No rate constraints on inputs; actuator wear risk on hardware | Low | v1_baseline |
| K6 | CLARABEL solver does not support `max_iter` parameter — solver settings limited | Low | v1_baseline |
| K7 | First few control steps show larger tracking error (warm-up transient from zero-input init) | Low | v1_baseline |

---

## Completed

| Version | Item |
|---------|------|
| v1 | Core DeePC with L2 regularization |
| v1 | Parametric CVXPY QP with warm-starting |
| v1 | PRBS + multisine mixed excitation |
| v1 | 8-scenario stress test suite |
| v1 | Cross-version comparison infrastructure |
| v1 | Hidden heading state (realistic output model) |
| v1 | Nonlinear vehicle simulator (kinematic bicycle) |
| v1 | Centralized config (single dataclass) |
| v1 | Sinusoidal reference tracking (100% optimal solves) |

---

## Changelog

### v0.1.0 (2026-03-05)
- Initial implementation of DeePC for autonomous vehicle trajectory tracking
- Kinematic bicycle model simulator
- PRBS + multisine + uniform random data collection
- CVXPY parametric QP with CLARABEL solver and SCS fallback
- Sinusoidal reference trajectory tracking (150 steps, 100% optimal)
- Three output plots: trajectory, tracking errors, control inputs
