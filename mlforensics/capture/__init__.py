"""Evidence capture helpers."""

import sys
from types import ModuleType

from .capsule import RunCapsule
from .context import (
    CHILD_CAPSULE_ENV,
    RUN_ID_ENV,
    CaptureContext,
    CaptureRunner,
    CaptureSession,
    capture,
    capture_rng_state,
    child_capture_environment,
    load_child_capsule,
    run_command,
)
from .data import (
    fingerprint_dataset,
    fingerprint_directory,
    fingerprint_file,
    fingerprint_paths,
    infer_schema,
)
from .dependencies import capture_dependency_inventory, dependency_inventory
from .environment import (
    DEFAULT_ENV_ALLOWLIST,
    capture_dependencies,
    capture_env,
    capture_environment,
    capture_environment_variables,
    capture_hardware,
    capture_python_environment,
    capture_python_info,
    capture_runtime,
)
from .git import (
    capture_git,
    capture_git_diff,
    capture_git_metadata,
    git_commits,
    git_diff,
    git_metadata,
    git_root,
)
from .system import capture_hardware_metadata, capture_system_metadata
from .training import TrainingCapture, capture_training, restore_training_state

__all__ = [
    "CaptureSession",
    "CaptureContext",
    "CaptureRunner",
    "RUN_ID_ENV",
    "CHILD_CAPSULE_ENV",
    "RunCapsule",
    "capture",
    "capture_dependencies",
    "capture_dependency_inventory",
    "dependency_inventory",
    "capture_environment",
    "capture_environment_variables",
    "capture_env",
    "DEFAULT_ENV_ALLOWLIST",
    "capture_python_environment",
    "capture_python_info",
    "capture_runtime",
    "capture_git",
    "capture_git_metadata",
    "capture_git_diff",
    "capture_hardware",
    "capture_hardware_metadata",
    "capture_system_metadata",
    "capture_rng_state",
    "child_capture_environment",
    "load_child_capsule",
    "fingerprint_dataset",
    "fingerprint_directory",
    "fingerprint_file",
    "fingerprint_paths",
    "git_commits",
    "git_diff",
    "git_root",
    "git_metadata",
    "infer_schema",
    "run_command",
    "TrainingCapture",
    "capture_training",
    "restore_training_state",
]


class _CallableCaptureModule(ModuleType):
    """Keep ``mlforensics.capture()`` convenient without masking the module."""

    def __call__(self, **kwargs):
        return capture(**kwargs)


# A package attribute named ``capture`` is a natural top-level convenience,
# but a plain function would make ``import mlforensics.capture.context`` bind
# incorrectly after importing the root package.  A callable module supports
# both forms while retaining all normal submodule import semantics.
sys.modules[__name__].__class__ = _CallableCaptureModule
