"""
Backward-compatible shim for artifact helpers.

Use `LLMBenchCore.ArtifactStore` for new code.
"""

from .ArtifactStore import (  # noqa: F401
  ArtifactStore,
  CadArtifactStore,
  ModelArtifactStore,
  file_content_changed,
  write_if_changed,
)
