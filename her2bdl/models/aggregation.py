from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import warnings
from ..data.constants import TARGET
from ..models.uncertainty import predictive_entropy, mutual_information, variation_ratio

class Aggregator(ABC):
    def __init__(self, uncertainty_metric):
        self.uncertainty_metric = uncertainty_metric.replace("_", " ")

    @abstractmethod
    def __call__(self, predictions_results, uncertainty_results):
        pass

class ThresholdAggregator(Aggregator):
    def __init__(self, uncertainty_metric, threshold):
        super().__init__(uncertainty_metric)
        self.t = threshold

    def __call__(self, predictions_results, uncertainty_results):
        # Filter by a uncertainty threshold
        uncertainty = uncertainty_results[self.uncertainty_metric]
        threshold_selector = uncertainty < self.t
        num_classes = predictions_results['y_predictive_distribution'].shape[-1]
        if np.any(threshold_selector): 
            selected = {
                k:v[threshold_selector] for k,v in predictions_results.items()
            }
            # Evaluate predictive distribution by meaning selected predictions
            y_predictive_distribution = selected['y_predictive_distribution'].mean(axis=0)
            y_pred = y_predictive_distribution.argmax()
            y_predictions_samples = selected['y_predictions_samples'].reshape((-1, num_classes))
            prediction_agg_result = {
                'y_pred' : y_pred,
                'y_predictive_distribution': y_predictive_distribution,
                'y_predictions_samples': y_predictions_samples
            }
            # Recalculate uncertainties
            uncertainty_agg_result = {
                'predictive entropy': predictive_entropy(y_predictive_distribution, is_sample=True),
                'mutual information': mutual_information(y_predictive_distribution, y_predictions_samples, is_sample=True),
                'variation-ratio': variation_ratio(y_predictions_samples, is_sample=True)

            }
        else:            
            warnings.warn(f"No predictions with uncertainty '{self.uncertainty_metric}' lower than {self.t} ")
            prediction_agg_result = {
                'y_pred' : None,
                'y_predictive_distribution': np.array(num_classes*[None]),
                'y_predictions_samples': np.array(num_classes*[None]).reshape((1, num_classes))
            }
            # Recalculate uncertainties
            uncertainty_agg_result = {
                'predictive entropy': None,
                'mutual information': None,
                'variation-ratio': None

            }
        return prediction_agg_result, uncertainty_agg_result

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