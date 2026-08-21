"""Sets up a scenario in which a single locust makes decisions about the
direction it wants to go based on static targets with certain geometry.

The package is split by concern:

  angles               angle wrapping and arc helpers
  targets              ``Targets``: geometry, apparent extent, occlusion
  angle_distributions  the warp / weight distribution family library
  perception_model     ``PerceptionModel``: egocentric -> neural, and rho
  neural_band_model    ``NeuralBandModel``: gamma dynamics, equilibria, stability
  _nbm_plots           its rasters and diagnostic plots        (mixin)
  _nbm_basins          its heading-basin wheels                (mixin)
  _nbm_walkers         its SDE walker ensemble                 (mixin)

Everything the rest of the repo imports is re-exported here, so both
``import decision_model as model`` and ``from decision_model import Targets``
reach the same objects they always did.  ``FAMILY_INFO`` in particular is
re-exported by reference, not copied: ``weighting_analysis/anti_foveal.py``
registers extra families by updating it in place.
"""

from . import angle_distributions
from .angles import convert_angles, _smallest_enclosing_arc  # noqa: F401
from .angle_distributions import FAMILY_INFO
from .targets import Targets
from .perception_model import PerceptionModel, _ReadOnlyParams  # noqa: F401
from .neural_band_model import NeuralBandModel

__all__ = [
    'Targets',
    'PerceptionModel',
    'NeuralBandModel',
    'angle_distributions',
    'FAMILY_INFO',
    'convert_angles',
]
