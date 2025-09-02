# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-


"""HydraxMPM library.

Built with JAX.
"""

# ruff: noqa: E402
# ruff: noqa: F401
from jaxtyping import install_import_hook

hook = install_import_hook("hydraxmpm", "typeguard.typechecked")

from .common.base import Base
from .common.types import TypeFloat, TypeFloatMatrix3x3, TypeFloatScalarPStack, TypeInt
from .constitutive_laws.constitutive_law import ConstitutiveLaw
from .constitutive_laws.druckerprager import DruckerPrager

# from .constitutive_laws.linearelastic import LinearIsotropicElastic
from .constitutive_laws.modifiedcamclay import ModifiedCamClay

# from .constitutive_laws.mu_i_rheology_incompressible import MuI_incompressible
from .constitutive_laws.newtonfluid import NewtonFluid
from .forces.boundary import Boundary
from .forces.force import Force
from .forces.gravity import Gravity
from .forces.rigidparticles import RigidParticles
from .forces.slipstickboundary import SlipStickBoundary
from .grid.grid import Grid
from .material_points.material_points import MaterialPoints
from .plotting import helpers, viewer

# from .sip_benchmarks.sip_benchmarks import (
#                            ConstantPressureShear,
#                            IsotropicCompression,
#                            TriaxialConsolidatedDrained,
#                            TriaxialConsolidatedUndrained,
# )
from .solvers.mpm_solver import MPMSolver

# from .solvers.sip_solver import SIPSolver
from .solvers.usl import USL
from .solvers.usl_apic import USL_APIC

# from .solvers.usl_asflip import USL_ASFLIP
from .utils.math_helpers import (
                           get_dev_strain,
                           get_dev_stress,
                           get_pressure,
                           get_pressure_stack,
                           get_q_vm,
                           get_q_vm_stack,
                           get_sym_tensor_stack,
                           get_volumetric_strain,
)
from .utils.mpm_callback_helpers import npz_to_vtk
from .utils.plot import make_plot

hook.uninstall()
del hook
