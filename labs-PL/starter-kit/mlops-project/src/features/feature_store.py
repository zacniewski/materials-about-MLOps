"""A minimal offline feature store backed by Parquet and a JSON registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class FileFeatureStore:
    """Persist named feature groups as Parquet with a small local registry."""

    def __init__(self, root: Path = Path("data/feature_store")) -> None:
        self.root = root
        self.registry_path = root / "registry.json"

    def write(
        self, name: str, frame: pd.DataFrame, entity_key: str = "customer_id"
    ) -> Path:
        if entity_key not in frame.columns:
            raise ValueError(f"Entity key {entity_key!r} is missing")
        if frame[entity_key].duplicated().any():
            raise ValueError("Entity keys must be unique within a feature group")
        self.root.mkdir(parents=True, exist_ok=True)
        output = self.root / f"{name}.parquet"
        frame.to_parquet(output, index=False)
        registry = (
            json.loads(self.registry_path.read_text(encoding="utf-8"))
            if self.registry_path.exists()
            else {}
        )
        registry[name] = {
            "path": output.name,
            "entity_key": entity_key,
            "columns": list(frame.columns),
            "updated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        }
        self.registry_path.write_text(
            json.dumps(registry, indent=2) + "\n", encoding="utf-8"
        )
        return output

    def read(self, name: str, entity_ids: list[int] | None = None) -> pd.DataFrame:
        if not self.registry_path.exists():
            raise KeyError(f"Feature group {name!r} is not registered")
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if name not in registry:
            raise KeyError(f"Feature group {name!r} is not registered")
        metadata = registry[name]
        frame = pd.read_parquet(self.root / metadata["path"])
        if entity_ids is not None:
            frame = frame[frame[metadata["entity_key"]].isin(entity_ids)]
        return frame
