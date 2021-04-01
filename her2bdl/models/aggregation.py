from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Aggregator(ABC):
    def __init__(self, uncertainty_metric):
        self.uncertainty_metric = uncertainty_metric
        self.tissue_results_columns = (\
            "CaseNo", "label", "CaseNo_label", "row", "col",
            TARGET, "prediction", uncertainty_metric
        )

    def __add_column(self, patch_results):
        patch_results = patch_results.copy()
        patch_results["CaseNo_label"] = patch_results.apply(
            lambda row: f"{int(row['CaseNo'])}_{int(row['label'])}", axis = 1
        )
        return patch_results

    @abstractmethod
    def __call__(self, patch_results, **kwargs):
        patch_results = self.__add_column(patch_results)
        pass
        return pd.DataFrame(tissue_results, columns=self.tissue_results_columns)

class ThresholdAggregator(Aggregator):
    def __init__(self, uncertainty_metric, threshold):
        super().__init__(uncertainty_metric)
        self.t = threshold

    def __call__(self, patch_results):
        patch_results = self.__add_column(patch_results)
        tissue_results = []
        for CaseNo_label, df in patch_results.groupby("CaseNo_label"):
            # threshold vote
            centain_predictions = df[df[self.uncertainty_metric] < self.t]
            prediction = centain_predictions["prediction"].mode()
            prediction = int(prediction[0]) if len(prediction) == 1 else None
            uncertainty = centain_predictions[self.uncertainty_metric].median()
            # Update results
            row_ = df.iloc[0]
            tissue_results.append((
                row_["CaseNo"], row_["label"], row_["CaseNo_label"], 
                row_["row"], row_["col"], row_[TARGET], prediction,
                uncertainty
            ))
        return pd.DataFrame(tissue_results, columns=self.tissue_results_columns)

class MixtureAggregator(Aggregator):
    def __call__(self, patch_results, patch_predictive_distributions):
        patch_results = self.__add_column(patch_results)
        tissue_results = []       
        for CaseNo_label, df in patch_results.groupby("CaseNo_label"):
            # mixture
            weights = 1/(df[self.uncertainty_target].to_numpy() + 1e-16)
            tissue_predictive_distribution = \
                (1./weights.sum())*np.dot(weights, patch_predictive_distributions[df.index])
            prediction = tissue_predictive_distribution.argmax()
            uncertainty = df[self.uncertainty_target].median()
            # Update results
            row_ = df.iloc[0]
            tissue_results.append((
                row_["CaseNo"], row_["label"], row_["CaseNo_label"], 
                row_["row"], row_["col"], row_[TARGET], prediction,
                uncertainty
            ))
        return pd.DataFrame(tissue_results, columns=self.tissue_results_columns)