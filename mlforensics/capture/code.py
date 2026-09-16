"""Code provenance capture compatibility module."""

from .git import capture_git, capture_git_diff, capture_git_metadata, git_diff, git_metadata

__all__ = ["capture_git", "capture_git_diff", "capture_git_metadata", "git_diff", "git_metadata"]
