"""Runtime feature toggle state with auto-save to config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.core.config import (
    DEFAULT_CONFIG_PATH,
    FEATURE_DISPLAY_NAMES,
    FEATURE_YAML_KEYS,
    AppConfig,
    save_features,
)

FeatureChangeCallback = Callable[[str, bool], None]

# Ordered list of toggle keys (FeatureConfig attribute names).
FEATURE_KEYS: tuple[str, ...] = tuple(FEATURE_YAML_KEYS.keys())


class FeatureStateManager:
    """Single authority for runtime feature flags; mutates ``config.features`` in place."""

    def __init__(
        self,
        config: AppConfig,
        config_path: Path | str | None = None,
        on_change: FeatureChangeCallback | None = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        self._on_change = on_change

    def is_enabled(self, key: str) -> bool:
        attr = self._resolve_key(key)
        return bool(getattr(self.config.features, attr))

    def set_enabled(self, key: str, value: bool, *, persist: bool = True) -> None:
        """Set a feature flag, optionally persist, and run change callbacks."""
        attr = self._resolve_key(key)
        current = getattr(self.config.features, attr)
        if current == value:
            return

        setattr(self.config.features, attr, value)

        if persist:
            self.persist()

        if self._on_change is not None:
            self._on_change(attr, value)

    def toggle(self, key: str) -> bool:
        """Flip a feature flag and return the new state."""
        attr = self._resolve_key(key)
        new_value = not getattr(self.config.features, attr)
        self.set_enabled(attr, new_value)
        return new_value

    def apply_many(self, changes: dict[str, bool]) -> list[str]:
        """Apply multiple flag changes in one persist cycle. Returns changed keys."""
        changed: list[str] = []
        for key, value in changes.items():
            attr = self._resolve_key(key)
            if getattr(self.config.features, attr) == value:
                continue
            setattr(self.config.features, attr, value)
            changed.append(attr)

        if not changed:
            return changed

        self.persist()
        if self._on_change is not None:
            for attr in changed:
                self._on_change(attr, getattr(self.config.features, attr))

        return changed

    def persist(self) -> None:
        """Write current feature flags to config.yaml."""
        save_features(self.config, self.config_path)

    def as_dict(self) -> dict[str, bool]:
        return {key: self.is_enabled(key) for key in FEATURE_KEYS}

    def active_features(self) -> list[str]:
        return [
            FEATURE_DISPLAY_NAMES[key]
            for key in FEATURE_KEYS
            if self.is_enabled(key)
        ]

    def summary(self) -> str:
        enabled = self.active_features()
        return ", ".join(enabled) if enabled else "none"

    def format_list(self) -> str:
        lines = []
        for key in FEATURE_KEYS:
            label = FEATURE_DISPLAY_NAMES[key]
            state = "on" if self.is_enabled(key) else "off"
            lines.append(f"  {label}: {state}")
        return "\n".join(lines)

    @staticmethod
    def _resolve_key(key: str) -> str:
        normalized = key.strip().lower().replace("-", "_")
        if normalized == "sources":
            normalized = "show_sources"
        if normalized not in FEATURE_YAML_KEYS:
            valid = ", ".join(FEATURE_DISPLAY_NAMES.values())
            raise KeyError(f"Unknown feature '{key}'. Valid features: {valid}")
        return normalized
