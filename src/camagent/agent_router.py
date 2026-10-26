"""Combined ecological tool implementation used by the LLM-facing registry.

This module deliberately contains no natural-language intent classifier or
keyword-to-tool routing function. ``WildlifeAgent.plan`` sends the user's
question and planner-visible tool schemas to the configured language model;
``ToolRegistry`` uses this class only as the concrete implementation behind
the tools that the model selected.
"""

from .movement_tools import MovementTools
from .quality_tools import QualityTools
from .spatial_tools import SpatialTools
from .temporal_tools import TemporalTools


class EcologicalToolset(SpatialTools, TemporalTools, QualityTools, MovementTools):
    """Concrete ecological functions exposed through the LLM-facing registry."""

    pass


# Backward compatibility for external imports created before the class was
# renamed.  This alias does not perform natural-language routing.
AgentRouter = EcologicalToolset
