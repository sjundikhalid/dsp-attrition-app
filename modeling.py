import os
import warnings
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature

try:
    import dagshub
except Exception:
    dagshub = None

warnings.filterwarnings("ignore")
load_dotenv()


def init_dagshub(repo_owner: str, repo_name: str, use_dagshub: bool = False):
    """Initialize DagsHub integration (optional).

    If use_dagshub is True and dagshub package is available, call dagshub.init(...)
    and set MLflow tracking URI to the returned tracking uri.
    """
    if not use_dagshub:
        return None
    if dagshub is None:
        raise RuntimeError("dagshub package not installed. Install it or set USE_DAGSHUB=0.")

    # dagshub.init handles setting up remote tracking automatically when mlflow=True
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    # DagsHub will have set MLflow tracking URI via environment; return it for visibility
    return mlflow.get_tracking_uri()


def run_rf_model_mlflow(df: pd.DataFrame,
                        experiment_name: str = "attrition_prediction",
                        use_dagshub: bool = False,
                        repo_owner: str = "sjundikhalid",
                        repo_name: str = "dsp-attrition-app"):
    """Train RandomForest, log params/metrics/model to MLflow.

    Parameters:
    - df: DataFrame with target column 'Attrition'
    - experiment_name: MLflow experiment name
    - use_dagshub: if True, try to initialize DagsHub remote logging
    - repo_owner/repo_name: DagsHub repo info
    """
    # Optionally initialize DagsHub (this will set tracking URI and auth env vars)
    tracking_uri = None
    if use_dagshub:
        tracking_uri = init_dagshub(repo_owner=repo_owner, repo_name=repo_name, use_dagshub=True)

    # If MLFLOW_TRACKING_URI is set in env, mlflow.get_tracking_uri() will return it.
    if tracking_uri is None:
        # Allow override via env var MLFLOW_TRACKING_URI
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        if env_uri:
            mlflow.set_tracking_uri(env_uri)

    # Setup MLflow experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = MlflowClient()
    if experiment is None:
        experiment_id = client.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        print(f"Experiment '{experiment_name}' already exists (id={experiment.experiment_id})")

    # Prepare data
    if 'Attrition' not in df.columns:
        raise ValueError("DataFrame harus berisi kolom 'Attrition' sebagai target")

    y = df['Attrition']
    X = df.drop(['Attrition'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    # Start run
    with mlflow.start_run(run_name="rf-default-model", experiment_id=experiment.experiment_id) as run:
        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)
        y_pred = model_rf.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        model_params = model_rf.get_params()
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"tuned_{param_name}", param_value)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        model_signature = infer_signature(model_input=X_train, model_output=y_train)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                registered_model_name="rf_model",
                input_example=X_train.head(1),
                signature=model_signature
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                signature=model_signature,
                input_example=X_train.head(1)
            )

        # Print run metadata
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))


def example_dagshub_test_log():
    """Example small function that uses mlflow logging via DagsHub init.

    This mirrors the snippet you provided and is safe to call after init.
    """
    # simple test run
    with mlflow.start_run():
        mlflow.log_param('parameter name', 'value')
        mlflow.log_metric('metric name', 1)


if __name__ == "__main__":
    # Configuration via environment variables (convenient for CI or local runs)
    USE_DAGSHUB = os.getenv('USE_DAGSHUB', '0') in ('1', 'true', 'True')
    DAGSHUB_OWNER = os.getenv('DAGSHUB_OWNER', 'sjundikhalid')
    DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'dsp-attrition-app')

    dataset = "data/data_full_dsp1.csv"
    df = pd.read_csv(dataset)

    if USE_DAGSHUB:
        print('Initializing DagsHub...')
        try:
            uri = init_dagshub(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, use_dagshub=True)
            print('DagsHub initialized, tracking uri:', uri)
            # optional small test log
            example_dagshub_test_log()
        except Exception as e:
            print('Failed to init DagsHub:', e)

    # Run model training and logging
    run_rf_model_mlflow(df, use_dagshub=USE_DAGSHUB, repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO)