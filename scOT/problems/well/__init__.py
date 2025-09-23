"""Well dataset module."""

from .well_active_matter import WellActiveMatter
from .well_helmholtz_staircase import WellHelmholtzStaircase
from .well_gray_scott_reaction_diffusion import WellGrayScottReactionDiffusion
from .well_shear_flow import WellShearFlow
from .well_turbulent_radiative_layer_2d import WellTurbulentRadiativeLayer2D
from .well_rayleigh_benard import WellRayleighBenard
from .well_acoustic_scattering_maze import WellAcousticScatteringMaze
from .well_viscoelastic_instability import WellViscoelasticInstability

__all__ = [
    'WellActiveMatter',
    'WellHelmholtzStaircase',
    'WellGrayScottReactionDiffusion',
    'WellShearFlow',
    'WellTurbulentRadiativeLayer2D',
    'WellRayleighBenard',
    'WellAcousticScatteringMaze',
    'WellViscoelasticInstability'
]
