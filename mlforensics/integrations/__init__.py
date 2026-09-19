"""Optional ecosystem integrations."""

__all__ = [
    "MLflowAdapter",
    "OpenLineageAdapter",
    "WandBAdapter",
    "export_dvc_dependencies",
    "lineage_event",
    "read_dvc_dependencies",
]


def __getattr__(name: str):
    """Load an integration module only when one of its public names is used."""
    if name in {"MLflowAdapter"}:
        from .mlflow import MLflowAdapter

        return MLflowAdapter
    if name in {"WandBAdapter"}:
        from .wandb import WandBAdapter

        return WandBAdapter
    if name in {"OpenLineageAdapter", "lineage_event"}:
        from .openlineage import OpenLineageAdapter, lineage_event

        return {"OpenLineageAdapter": OpenLineageAdapter, "lineage_event": lineage_event}[name]
    if name in {"export_dvc_dependencies", "read_dvc_dependencies"}:
        from .dvc import export_dvc_dependencies, read_dvc_dependencies

        return {
            "export_dvc_dependencies": export_dvc_dependencies,
            "read_dvc_dependencies": read_dvc_dependencies,
        }[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
