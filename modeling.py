import os
import warnings
import sys
import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")
load_dotenv()

def run_rf_model_mlflow(df):
    dagshub.init(repo_owner='sjundikhalid', repo_name='dsp-attrition-app', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/sjundikhalid/dsp-attrition-app.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    experiment_name = "attrition_prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = None
    client = MlflowClient()
    if experiment is None:
        experiment_id = client.create_experiment(name = experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
        experiment = mlflow.get_experiment(experiment_id)
    else:
        print(f"Experiment '{experiment_name}' already exists")

    y = df['Attrition']
    X = df.drop(['Attrition'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

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
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))

if __name__ == "__main__":
    dataset = "data/data_full_dsp1.csv"
    df = pd.read_csv(dataset)
    run_rf_model_mlflow(df)