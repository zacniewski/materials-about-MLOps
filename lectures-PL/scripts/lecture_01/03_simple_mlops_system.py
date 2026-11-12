"""Wykład 1, sekcja 8 – Prosty system MLOps od zera.

Minimalny cykl życia modelu: trening → ewaluacja → rejestracja w rejestrze
→ promocja do produkcji (tylko jeśli nowy model jest lepszy) → predykcja.
Rejestr jest zapisywany na dysku (JSON + pliki .joblib).

Uruchomienie:
    python lectures/scripts/lecture_01/03_simple_mlops_system.py
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

REGISTRY_DIR = Path(__file__).parent / "outputs" / "model_registry"


@dataclass
class ModelVersion:
    name: str
    version: str
    metrics: dict
    params: dict
    artifact_path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    status: str = "staging"  # staging | production | archived


class SimpleModelRegistry:
    """Prosty rejestr modeli przechowywany w pliku JSON."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = root / "registry.json"
        self.models: dict[str, list[ModelVersion]] = {}
        if self.index.exists():
            raw = json.loads(self.index.read_text())
            self.models = {k: [ModelVersion(**v) for v in vs] for k, vs in raw.items()}

    def _save(self) -> None:
        data = {k: [asdict(v) for v in vs] for k, vs in self.models.items()}
        self.index.write_text(json.dumps(data, indent=2))

    def register(self, name: str, model, metrics: dict, params: dict) -> ModelVersion:
        version = str(len(self.models.get(name, [])) + 1)
        path = self.root / f"{name}_v{version}.joblib"
        joblib.dump(model, path)
        mv = ModelVersion(name, version, metrics, params, str(path))
        self.models.setdefault(name, []).append(mv)
        self._save()
        print(f"✅ Zarejestrowano: {name} v{version}  metrics={metrics}")
        return mv

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        return next((m for m in self.models.get(name, []) if m.status == "production"), None)

    def promote_if_better(self, name: str, version: str, metric: str = "f1") -> None:
        candidate = next(m for m in self.models[name] if m.version == version)
        current = self.get_production_model(name)
        if current and current.metrics[metric] >= candidate.metrics[metric]:
            print(f"⏸️  v{version} ({metric}={candidate.metrics[metric]}) nie jest lepszy od "
                  f"produkcyjnego v{current.version} ({current.metrics[metric]}) – zostaje w staging")
            return
        if current:
            current.status = "archived"
        candidate.status = "production"
        self._save()
        print(f"🚀 Wdrożono: {name} v{version} → produkcja")


def main() -> None:
    X, y = make_classification(n_samples=3000, n_features=20, n_informative=8, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Każde uruchomienie skryptu zaczyna od czystego rejestru (demo)
    if REGISTRY_DIR.exists():
        for f in REGISTRY_DIR.iterdir():
            f.unlink()
    registry = SimpleModelRegistry(REGISTRY_DIR)

    candidates = [
        (LogisticRegression(max_iter=500), {"model": "LogisticRegression"}),
        (RandomForestClassifier(n_estimators=100, random_state=42), {"model": "RF", "n_estimators": 100}),
        (GradientBoostingClassifier(n_estimators=10, max_depth=1, random_state=42),
         {"model": "GBM", "n_estimators": 10, "max_depth": 1}),
    ]
    for model, params in candidates:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        metrics = {"accuracy": round(accuracy_score(y_te, pred), 4),
                   "f1": round(f1_score(y_te, pred), 4)}
        mv = registry.register("churn-model", model, metrics, params)
        registry.promote_if_better("churn-model", mv.version)
        print()

    prod = registry.get_production_model("churn-model")
    print(f"📦 Model produkcyjny: v{prod.version} {prod.params} {prod.metrics}")
    model = joblib.load(prod.artifact_path)
    print(f"🔮 Predykcja dla 5 nowych próbek: {model.predict(X_te[:5]).tolist()}")
    print(f"📁 Rejestr: {registry.index}")


if __name__ == "__main__":
    main()
