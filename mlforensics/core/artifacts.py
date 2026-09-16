"""Backward-compatible import location for artifact stores."""

from .store import ArtifactStore, FsspecArtifactStore, LocalArtifactStore

__all__ = ["ArtifactStore", "FsspecArtifactStore", "LocalArtifactStore"]
