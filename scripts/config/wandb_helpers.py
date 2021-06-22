import wandb
import requests
import io
from copy import copy
import os

for var in ['HER2BDL_HOME', 'HER2BDL_DATASETS', 'HER2BDL_EXPERIMENTS', 'HER2BDL_EXTRAS']:
    os.environ[var] = "" #"debug"

columns = [
        'TPR - Class 0', 'FP - Class 2+', 'TPR Macro', 'TN - Class 1+',
        'PPV - Class 2+', 'F1 - Class 3+', 'Class Stat', 'F1 - Class 2+',
        'Confusion Matrix', 'Overall Stat', 'ROC Curve', 'ACC - Class 0', 
        'TN - Class 0', 'Examples', 'High Uncertainty', 
        'Uncertainty By Class', 'Uncertainty', 'Low Uncertainty',
        'PPV - Class 0', 'FN - Class 2+', 'ACC - Class 3+',
        'FN - Class 3+', 'PPV Macro', 'PPV Micro', 'FN - Class 1+',
        'TPR - Class 2+', 'F1 - Class 0', 'mean_mutual information',
        'ERR Macro', 'mean_predictive entropy', 'TP - Class 1+',
        'TN - Class 3+', 'FP - Class 0', 'TPR - Class 1+',
        'TP - Class 0', 'FN - Class 0', 'PPV - Class 1+', 'TP - Class 2+',
        'FP - Class 3+', 'ACC - Class 2+', 'TPR Micro', 'TP - Class 3+',
        'F1 - Class 1+', 'ACC - Class 1+', 'TPR - Class 3+', 
        'TN - Class 2+', 'ACC Macro', 'mean_variation-ratio', 
        'PPV - Class 3+', 'FP - Class 1+'
]

table_types = ["Overall Stat", "Class Stat"]


def add_best_to_summary(run_id, target="val_loss", op="min"):
    api = wandb.Api()
    run = api.run(f"sborquez/her2bdl/{run_id}")
    hist_df = run.history()
    if op == "min":
        best_index = hist_df[target].argmin()
    else:
        best_index = hist_df[target].argmax()
    best_row = hist_df.loc[best_index]
    for c in columns:
        print(f"Best - {c}", type(best_row[c]))
        if isinstance(best_row[c], dict):
            continue
        else:
            run.summary[f"best_{c}"] = best_row[c]
    run.summary.update()


import pandas as pd
import numpy as np
def load_experiments_ids(csv="D:/sebas/Projects/her2bdl/wandb_export_2021-06-20T00_23_38.029-04_00.csv"):
    df = pd.read_csv(csv)
    ids = df["ID"]
    return ids.tolist()

def _load_npy(run, name):
    response = requests.get(run.file(name).url)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content), allow_pickle=True) 
    return data.tolist()

def add_legend_agg(run_id):
    api = wandb.Api()
    run = api.run(f"sborquez/her2bdl/{run_id}")
    if run.config["aggregation"]["parameters"]["uncertainty_metric"] == "predictive_entropy":
        u = "H"
    else:
        u = "I"
    d = run.config["model"]["hyperparameters"]["mc_dropout_rate"]
    if run.config["aggregation"]["method"] == "ThresholdAggregator":
        t = run.config["aggregation"]["parameters"]["threshold"]
        if  t == 2:
            method = f"B-{d}"
        elif t in (0.1, 0.5):
            method = f"Tl_{u}-{d}"
        else:
            method = f"Th_{u}-{d}"
    else:
        method = f"M_{u}-{d}"
    run.config.update({"legend": method})
    run.save()

def add_contest_evaluation(run_id, confidence_measurement="mutual information"):
    api = wandb.Api()
    run = api.run(f"sborquez/her2bdl/{run_id}")
    agg_predictions = _load_npy(run, "aggregated_predictions.npy")
    agg_uncertainty = _load_npy(run, "aggregated_uncertainty.npy")
    agg_data = _load_npy(run, "aggregated_data.npy")

    # Setup Evaluation Table
    evaluation_table = {
        "group": agg_data["group"],
        "y_true": agg_data["y_true"],
        "y_pred" : agg_predictions["y_pred"],
        "predictive entropy": agg_uncertainty["predictive entropy"],
        "mutual information": agg_uncertainty["mutual information"]
    }
    del agg_data
    del agg_uncertainty
    del agg_predictions
    evaluation_table = pd.DataFrame(evaluation_table)
    # Agreement Points
    evaluation_table["agreement points"] = eval_agreement_points(
        y_true=evaluation_table["y_true"], 
        y_pred=evaluation_table["y_pred"]
    )
    # Weighted Confidence
    evaluation_table["weighted confidence"] = eval_weighted_confidence(
        y_true=evaluation_table["y_true"], 
        y_pred=evaluation_table["y_pred"],
        uncertainty =  evaluation_table[confidence_measurement]
    )
    evaluation_table["weighted confidence 2"] = eval_weighted_confidence(
        y_true=evaluation_table["y_true"], 
        y_pred=evaluation_table["y_pred"],
        uncertainty =  evaluation_table["predictive entropy"]
    )
    # Combined
    evaluation_table["combined"] = eval_combined(
        evaluation_table["agreement points"], 
        evaluation_table["weighted confidence"]
    )
    evaluation_table["combined 2"] = eval_combined(
        evaluation_table["agreement points"], 
        evaluation_table["weighted confidence 2"]
    )
    # Update summary
    model_evaluation = dict(evaluation_table[
        ["agreement points","weighted confidence","combined", "weighted confidence 2", "combined 2"]
    ].sum())
    for key, value in model_evaluation.items():
        run.summary[key] = value
    # artifact = wandb.Artifact("evaluation_table", type="results")
    # artifact.add(wandb.Table(evaluation_table), "Evaluation Table")
    # run.upsert_artifact(artifact)
    # run.finish_artifact(artifact)
    run.summary.update()

agreement_points = np.array([
    [15, 15, 10, 0],
    [15, 15, 10, 0],
    [2.5, 2.5, 15, 5],
    [0, 0, 10, 15]
])

def eval_agreement_points(y_true, y_pred):
    return pd.Series(agreement_points[y_true, y_pred])

def eval_weighted_confidence(y_true, y_pred, uncertainty, normalizer=1/np.log(4)):
    confidence_norm = 1 - uncertainty*normalizer
    ps_is_correct = (y_pred == y_true)
    wc_correct = \
        ps_is_correct*((2*confidence_norm - np.power(confidence_norm, 2))/2.0 )\
        + ~ps_is_correct*((1 - np.power(confidence_norm, 2))/2.0)
    return pd.Series(wc_correct)

def eval_combined(agreement_points, weighted_confidence):
    return agreement_points * weighted_confidence
