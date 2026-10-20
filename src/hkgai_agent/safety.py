"""Workspace confinement shared by all filesystem tools."""

from pathlib import Path, PureWindowsPath


class SafetyError(ValueError):
    """An input violates a workspace safety boundary."""


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise SafetyError("Workspace must be an existing directory")

    def resolve(self, relative_path: str, *, allow_root: bool = False) -> Path:
        if not isinstance(relative_path, str) or not relative_path or "\x00" in relative_path:
            raise SafetyError("A nonempty workspace-relative path is required")
        normalized = relative_path.replace("\\", "/")
        windows_path = PureWindowsPath(relative_path)
        if normalized.startswith("/") or windows_path.drive or windows_path.root:
            raise SafetyError("Absolute and drive-relative paths are forbidden")
        parts = normalized.split("/")
        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
        }
        if any(
            part == ".."
            or ":" in part
            or part.rstrip(" .").split(".")[0].upper() in reserved
            or (part not in {"", "."} and part.endswith((" ", ".")))
            for part in parts
        ):
            raise SafetyError("Traversal, device names, and ambiguous path components are forbidden")
        try:
            candidate = (self.root / normalized).resolve()
            candidate.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as error:
            raise SafetyError("Path escapes the workspace or cannot be resolved") from error
        if candidate == self.root and not allow_root:
            raise SafetyError("A file path inside the workspace is required")
        return candidate
