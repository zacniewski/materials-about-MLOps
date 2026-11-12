"""Wykład 4, sekcje 4–6 – Kompilacja pipeline'u KFP v2 do YAML (bez klastra).

Definiuje lekkie komponenty KFP, pipeline z warunkiem `dsl.If/dsl.Else`
i kompiluje go do pliku IR YAML. Następnie analizuje wygenerowany YAML:
lista zadań, ich zależności i parametry wejściowe. Kubernetes NIE jest
potrzebny – kompilacja jest czysto lokalna (wystarczy pakiet `kfp`).

Uruchomienie:
    python lectures/scripts/lecture_04/03_kfp_compile.py
"""

from pathlib import Path

import yaml
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

OUT = Path(__file__).parent / "outputs"
PKGS = ["pandas", "pyarrow", "scikit-learn", "joblib"]


@dsl.component(base_image="python:3.11-slim", packages_to_install=PKGS)
def ingest_data(n_samples: int, output_dataset: Output[Dataset]):
    import pandas as pd
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    df.to_parquet(output_dataset.path, index=False)


@dsl.component(base_image="python:3.11-slim", packages_to_install=PKGS)
def train_model(input_dataset: Input[Dataset], n_estimators: int,
                output_model: Output[Model], metrics: Output[Metrics]) -> float:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet(input_dataset.path)
    X, y = df.drop(columns="target"), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    metrics.log_metric("auc_roc", auc)
    joblib.dump(model, output_model.path)
    return auc


@dsl.component(base_image="python:3.11-slim")
def deploy_model(input_model: Input[Model]):
    print(f"Wdrażam model z {input_model.uri}")


@dsl.component(base_image="python:3.11-slim")
def send_notification(message: str):
    print(message)


@dsl.pipeline(name="churn-pipeline-demo", description="Demo wykładu 4")
def churn_pipeline(n_samples: int = 10000, n_estimators: int = 100,
                   auc_threshold: float = 0.80):
    ingest_task = ingest_data(n_samples=n_samples)
    train_task = train_model(input_dataset=ingest_task.outputs["output_dataset"],
                             n_estimators=n_estimators)
    train_task.set_cpu_limit("2").set_memory_limit("4G").set_retry(num_retries=2)
    with dsl.If(train_task.outputs["Output"] >= auc_threshold, name="auc-ok"):
        deploy_model(input_model=train_task.outputs["output_model"])
    with dsl.Else(name="auc-too-low"):
        send_notification(message="Model odrzucony – AUC poniżej progu")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "churn_pipeline.yaml"
    compiler.Compiler().compile(pipeline_func=churn_pipeline, package_path=str(path))
    print(f"✓ Skompilowano: {path} ({path.stat().st_size / 1024:.1f} KB)")

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    print("\nParametry pipeline'u:")
    for name, p in spec["root"]["inputDefinitions"]["parameters"].items():
        print(f"  {name:<14} {p['parameterType']:<12} domyślnie={p.get('defaultValue')}")

    def walk(dag: dict, indent: int = 1):
        for task, t in dag.get("tasks", {}).items():
            deps = t.get("dependentTasks", [])
            cond = t.get("triggerPolicy", {}).get("condition")
            print("  " * indent + f"- {task}  deps={deps}" + (f"  if: {cond}" if cond else ""))
            comp = spec["components"][t["componentRef"]["name"]]
            if "dag" in comp:
                walk(comp["dag"], indent + 1)

    print("\nZadania w DAG (z zagnieżdżonymi grupami warunkowymi):")
    walk(spec["root"]["dag"])
    print(f"\nLiczba executorów (obrazów do uruchomienia): {len(spec['deploymentSpec']['executors'])}")


if __name__ == "__main__":
    main()
