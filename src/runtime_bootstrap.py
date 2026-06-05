from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import re
import sys
from pathlib import Path


def _normalized_dist_name(value: str) -> str:
    return value.lower().replace("-", "_")


def _patch_importlib_metadata(venv_site_packages: Path) -> None:
    original_distributions = importlib_metadata.distributions
    entry_points_type = importlib_metadata.EntryPoints

    def parse_dist_info_name(path_name: str) -> tuple[str, str] | None:
        match = re.match(r"^(?P<name>.+)-(?P<version>\d[^\\]*)\.dist-info$", path_name)
        if match is None:
            return None
        return match.group("name"), match.group("version")

    def safe_distributions(**kwargs):
        return original_distributions(path=[str(venv_site_packages)])

    def safe_distribution(name: str):
        target = _normalized_dist_name(name)
        for dist in safe_distributions():
            parsed = parse_dist_info_name(getattr(dist, "_path", Path("")).name)
            if parsed is None:
                continue
            dist_name, _ = parsed
            if _normalized_dist_name(dist_name) == target:
                return dist
        raise importlib_metadata.PackageNotFoundError(name)

    def safe_version(name: str) -> str:
        target = _normalized_dist_name(name)
        for dist_info_path in venv_site_packages.glob("*.dist-info"):
            parsed = parse_dist_info_name(dist_info_path.name)
            if parsed is None:
                continue
            dist_name, dist_version = parsed
            if _normalized_dist_name(dist_name) == target:
                return dist_version
        raise importlib_metadata.PackageNotFoundError(name)

    def safe_packages_distributions() -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for dist in safe_distributions():
            parsed = parse_dist_info_name(getattr(dist, "_path", Path("")).name)
            if parsed is None:
                continue
            dist_name, _ = parsed
            try:
                top_level = dist.read_text("top_level.txt")
            except OSError:
                top_level = None
            if not top_level:
                continue
            for package_name in [line.strip() for line in top_level.splitlines() if line.strip()]:
                mapping.setdefault(package_name, []).append(dist_name)
        return mapping

    def safe_entry_points(**params):
        return entry_points_type(()).select(**params)

    importlib_metadata.distributions = safe_distributions
    importlib_metadata.distribution = safe_distribution
    importlib_metadata.version = safe_version
    importlib_metadata.packages_distributions = safe_packages_distributions
    importlib_metadata.entry_points = safe_entry_points


def bootstrap_runtime() -> None:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    try:
        base_prefix = Path(sys.base_prefix).resolve()
        venv_site_packages = (Path(sys.prefix) / "Lib" / "site-packages").resolve()
    except OSError:
        return

    filtered_paths: list[str] = []
    for entry in sys.path:
        if not entry:
            filtered_paths.append(entry)
            continue

        try:
            resolved = Path(entry).resolve()
        except OSError:
            filtered_paths.append(entry)
            continue

        if resolved == base_prefix:
            continue

        filtered_paths.append(entry)

    sys.path[:] = filtered_paths
    _patch_importlib_metadata(venv_site_packages)