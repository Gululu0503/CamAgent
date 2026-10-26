__all__ = [
    "WildlifeAgent",
    "AgentRunPolicy",
    "ToolRegistry",
    "VisualizationTools",
    "EcologicalToolset",
    "AgentRouter",
    "convert_species_csv_directory_to_camtrap_dp",
    "run_ecology_figure_suite",
    "run_pipeline",
    "run_agent_contract_stress_matrix",
]


def __getattr__(name):
    if name in {"WildlifeAgent", "AgentRunPolicy"}:
        from .llm_agent import AgentRunPolicy, WildlifeAgent

        return {"WildlifeAgent": WildlifeAgent, "AgentRunPolicy": AgentRunPolicy}[name]
    if name == "ToolRegistry":
        from .tool_registry import ToolRegistry

        return ToolRegistry
    if name == "VisualizationTools":
        from .visualization_tools import VisualizationTools

        return VisualizationTools
    if name in {"EcologicalToolset", "AgentRouter"}:
        from .agent_router import AgentRouter, EcologicalToolset

        return {
            "EcologicalToolset": EcologicalToolset,
            "AgentRouter": AgentRouter,
        }[name]
    if name == "convert_species_csv_directory_to_camtrap_dp":
        from .camtrap_conversion import convert_species_csv_directory_to_camtrap_dp

        return convert_species_csv_directory_to_camtrap_dp
    if name == "run_ecology_figure_suite":
        from .ecology_figures import run_ecology_figure_suite

        return run_ecology_figure_suite
    if name == "run_pipeline":
        from .research_pipeline import run_pipeline

        return run_pipeline
    if name == "run_agent_contract_stress_matrix":
        from .agent_stress import run_agent_contract_stress_matrix

        return run_agent_contract_stress_matrix
    raise AttributeError(f"module 'camagent' has no attribute {name!r}")
