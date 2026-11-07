# Real-Time XPBD Cloth & Soft-Body Physics Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Numba](https://img.shields.io/badge/Numba-JIT_Accelerated-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A real-time 3D physics simulation framework for deformable bodies and cloth dynamics implemented in Python.

To overcome Python's native execution overhead in compute-heavy particle loops, this engine leverages **Numba (LLVM JIT compilation)** and vectorized **NumPy** arrays, achieving real-time frame rates for thousands of interacting constraints and particles.

---

## Key Technical Features

* **Extended Position-Based Dynamics (XPBD):** Implements time-step and substep independent constraint compliance (`α`). Unlike traditional PBD or basic Mass-Spring systems, material stiffness does not artificially scale with solver iteration count or frame rate fluctuations.
* **Substepping Solver Architecture:** Decouples the main rendering frame time (`Δt`) into smaller physics substeps (`Δt_s = Δt / N_substeps`), significantly improving convergence stability under high tension without sacrificing performance.
* **O(N) Spatial Hashing Collision Detection:** Replaces naive `O(N²)` broad-phase particle-particle and self-collision checks with a custom 3D uniform grid spatial hash table, enabling real-time self-collision resolution.
* **Numba JIT Acceleration:** Core solver loops—including distance/bending constraint projection, Verlet integration, and spatial grid queries—are compiled directly to machine code with `@njit(fastmath=True, parallel=True)`.
* **Interactive Constraints & Forces:** Supports structural, shear, and bending distance constraints, dynamic pinning, aerodynamic drag/wind forces, and analytical primitive collisions (spheres, planes).

---

## Mathematical Formulation

The solver resolves a system of non-linear constraints **C(x) = 0** by iteratively updating particle positions **x** along the constraint gradients **∇C(x)**.

For each constraint with inverse mass matrix **M⁻¹** and time-scaled compliance parameter **α̃ = α / Δt²**, the Lagrange multiplier increment **Δλ** and position correction **Δx** are computed as:

```text
          -C(x) - α̃ · λ
Δλ = ─────────────────────────
     ∇C(x) · M⁻¹ · ∇C(x)ᵀ + α̃

Δx = M⁻¹ · ∇C(x)ᵀ · Δλ
