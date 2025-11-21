
# Numerical Methods Development Agent

## 1. Purpose

The broad purpose is to implement double-point material point method based on the description in the article xi_paper.pdf and xi_implemention.pdf. Therfore every implementation should fall within the category of modifying and extending the current MPM code such that it become a double-point MPM that can simulate sediment-fluid interaction. 



## 2. Repository Context

- Language: Python (JAX-based numerical computing).

### Key directories:

- `hydraxmpm/`  
  – Main HydraxMPM library package.

- `hydraxmpm/common/`  
  – Shared foundations and typing infrastructure.  
    Common base classes.  
    Type aliases for particle/node arrays and mathematical tensors.

- `hydraxmpm/grid/`  
  – Background grid representation used in MPM.  
    Node masses, momenta, velocities, normals, and grid-level operations.

- `hydraxmpm/material_points/`  
  – Material point state and evolution.  
    Particle positions, velocities, deformation gradient, mass/volume, stresses.

- `hydraxmpm/shapefunctions/`  
  – Interpolation kernels and particle–grid transfer logic.  
    Linear, quadratic, and cubic B-spline basis functions.  
    Shape-function gradients and particle–node connectivity.

- `hydraxmpm/constitutive_laws/`  
  – Material models.  
    Drucker–Prager plasticity.  
    Modified Cam-Clay.  
    Newtonian fluids.  
    Base class for new constitutive laws.

- `hydraxmpm/forces/`  
  – External forces and boundary conditions.  
    Gravity.  
    Stick–slip boundaries.  
    Rigid body interactions.  
    Domain-related boundary handling.

- `hydraxmpm/solvers/`  
  – MPM time integration engines.  
    Base MPM solver.  
    Explicit USL and USL-APIC variants.

- `hydraxmpm/utils/`  
  – Supporting utilities.  
    Mathematical tensor helpers.  
    STL mesh sampling.  
    Domain-filling tools.  
    Callback and output management.  
    JAX helper utilities.

- `hydraxmpm/plotting/`  
  – Visualization utilities.  
    Plotting helpers.  
    Polyscope viewer for simulation outputs.

- `hydraxmpm/__init__.py`  
  – Top-level API exposure and runtime type-checking hook.


## Code Style and Consistency Requirements

All new code written by the agent must strictly follow the existing style of
the repository. Style consistency is mandatory and overrides any external
conventions.

The agent must:

-  Mirror the architecture of existing solvers or algorithms. Any new implementations
   match the design, naming conventions,
   internal function structure, and argument ordering used by existing
   implementations.
   
- Inspect the numerical and mathematical description of the double-point MPM algorithm for sediment-fluid system as described in the files xi_paper.pdf and xi_implementation.pdf

## 8. Safety and Reporting

- Do not delete files unless explicitly instructed.
- Do not modify `.git` or external configuration.










































































































































































































































