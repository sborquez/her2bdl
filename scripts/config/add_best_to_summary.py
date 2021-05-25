import wandb
from copy import copy

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


def update(run_id, target="val_loss", op="min"):
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

