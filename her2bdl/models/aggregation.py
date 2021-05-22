from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import warnings
from ..data.constants import TARGET
from ..models.uncertainty import predictive_entropy, mutual_information, variation_ratio

class Aggregator(ABC):
    def __init__(self, uncertainty_metric, by="CaseNo_label"):
        self.uncertainty_metric = uncertainty_metric.replace("_", " ")
        self.by=by

    def predict_with_aggregation(self, dataset, predictions_results, uncertainty_results, include_data=False, verbose=0, **kwargs):
        """
        Aggregate predictions with epistemic uncertainty results for dataset.
        Parameters
        ----------
        dataset : `keras.utils.Sequence` or `ParallelMapDataset`
            Data generator or TF_Dataset.
        predictions_results : `dict`
            Predictions for the dataset.
        uncertainty_results : `dict`
            Epistemic uncertainty for the dataset.
        include_data    : `bool``
            Returns images and y_true from dataset.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            (`dict`, `dict`)
                Aggregated prediction and uncertainty results for dataset.
        """
        aggregated_data = {"group": [], "df": [], 'y_true': []} #, 'X': []}
        aggregated_map  = {"prediction": [], "uncertainty": []}
        aggregated_predictions = {k:[] for k in predictions_results.keys()}
        aggregated_uncertainty = {k:[] for k in uncertainty_results.keys()}
        iterator = dataset.get_partition(predictions_results, uncertainty_results, by=self.by)
        if verbose>0:
            total = dataset.get_n_partitions(by=self.by)
            iterator = tqdm(iterator, total=total)
        for group_partition in iterator:
            group, group_df, group_predictions_results, group_uncertainty_results = group_partition
            prediction_agg_result, uncertainty_agg_result = self(group_predictions_results, group_uncertainty_results)
            # Aggregated predictions
            for k,v in prediction_agg_result.items():
                aggregated_predictions[k].append(v)
            for k,v in uncertainty_agg_result.items():
                aggregated_uncertainty[k].append(v)
            if include_data:
                y_true = group_df[TARGET].unique()[0]
                aggregated_data["group"].append(group)
                aggregated_data["df"].append(group_df)
                aggregated_data["y_true"].append(y_true)
                #aggregated_data["X"].append(y_true)
                #aggregated_map["prediction"].append()
                #aggregated_map["uncertainty"].append()
        aggregated_data["group"] = np.array(aggregated_data["group"])
        aggregated_data["y_true"] = np.array(aggregated_data["y_true"])
        aggregated_predictions = {k:np.array(v) for k,v in aggregated_predictions.items()}
        aggregated_uncertainty = {k:np.array(v) for k,v in aggregated_uncertainty.items()}
        if include_data: return aggregated_data, aggregated_predictions, aggregated_uncertainty
        return aggregated_predictions, aggregated_uncertainty
    
    @abstractmethod
    def __call__(self, predictions_results, uncertainty_results):
        pass

class ThresholdAggregator(Aggregator):
    def __init__(self, uncertainty_metric, threshold, by="CaseNo_label"):
        super().__init__(uncertainty_metric, by=by)
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
            y_predictive_distribution_samples = selected['y_predictive_distribution']
            y_predictive_distribution = y_predictive_distribution_samples.mean(axis=0)
            y_pred = y_predictive_distribution.argmax()
            #y_predictions_samples = selected['y_predictions_samples'].reshape((-1, num_classes))
            prediction_agg_result = {
                'y_pred' : y_pred,
                'y_predictive_distribution': y_predictive_distribution,
                'y_predictions_samples': y_predictive_distribution_samples
            }
            # Recalculate uncertainties
            uncertainty_agg_result = {
                'predictive entropy': predictive_entropy(y_predictive_distribution, is_sample=True),
                'mutual information': mutual_information(y_predictive_distribution, y_predictive_distribution_samples, is_sample=True),
                'variation-ratio': variation_ratio(y_predictive_distribution_samples, is_sample=True)
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
    def get_weights(self, uncertainty):
        w = np.exp(-1*uncertainty)
        w /= np.sum(w)
        return w

    def __call__(self, predictions_results, uncertainty_results):
        # Filter by a uncertainty threshold
        uncertainty = uncertainty_results[self.uncertainty_metric]
        num_classes = predictions_results['y_predictive_distribution'].shape[-1]
        weights = self.get_weights(uncertainty)
        # Evaluate predictive distribution by meaning selected predictions
        y_predictive_distribution_samples = predictions_results['y_predictive_distribution']
        y_predictive_distribution = np.average(y_predictive_distribution_samples, weights=weights, axis=0)
        y_pred = y_predictive_distribution.argmax()
        #y_predictions_samples = predictions_results['y_predictions_samples'].reshape((-1, num_classes))
        prediction_agg_result = {
            'y_pred' : y_pred,
            'y_predictive_distribution': y_predictive_distribution,
            'y_predictions_samples': y_predictive_distribution_samples
        }
        # Recalculate uncertainties
        uncertainty_agg_result = {
            'predictive entropy': predictive_entropy(y_predictive_distribution, is_sample=True),
            'mutual information': mutual_information(y_predictive_distribution, y_predictive_distribution_samples, is_sample=True, weights=weights),
            'variation-ratio': variation_ratio(y_predictive_distribution_samples, is_sample=True, weights=weights)
        }

        return prediction_agg_result, uncertainty_agg_result