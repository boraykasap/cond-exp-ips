# Conditional Moment Estimation for Correlated Particle Systems

## Problem

Estimating conditional moments E[V | X = x] from particle data is the
key computational bottleneck for variance reduction in low Mach number
kinetic simulations. The choice of nonparametric estimator critically
affects bias, variance, and stability of the correlated process.

## This Repository

Particle simulations of planar Couette flow using the Fokker-Planck model,
comparing conditional moment estimators:

- **Binning** — hard cell assignment
- **Nadaraya-Watson** — kernel-weighted average  
- **Local Linear Regression** — local weighted least squares

Each estimator is evaluated in the context of a correlated parallel process
for variance reduction (Gorji et al. 2015).