"""
Utility tools for train/evaluate experiments
================================================

Collection of functions shared between train scripts.
"""


import numpy as np
import yaml
import wandb
from wandb.keras import WandbCallback
    
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def load_config_file(config_filepath, mode="train"):
    with open(config_filepath) as yaml_file:
        base_config = yaml.safe_load(yaml_file)
    


def setup_callbacks():
    callbacks =[]
    return callbacks

class WandbClassificationCallback(WandbCallback):
    """
    WandbCallback extension for include extra plots and summaries.

    original from: https://wandb.ai/mathisfederico/wandb_features/reports/Visualizing-Confusion-Matrices-With-W-B--VmlldzoxMzE5ODk
    """

    def __init__(self, monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=False, log_weights=False, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=[], data_type=None, predictions=1, generator=None,
                 input_type=None, output_type=None, log_evaluation=False,
                 validation_steps=None, class_colors=None, log_batch_frequency=None,
                 log_best_prefix="best_", 
                 log_confusion_matrix=False,
                 confusion_examples=0, confusion_classes=5):
        
        super().__init__(monitor=monitor,
                        verbose=verbose, 
                        mode=mode,
                        save_weights_only=save_weights_only,
                        log_weights=log_weights,
                        log_gradients=log_gradients,
                        save_model=save_model,
                        training_data=training_data,
                        validation_data=validation_data,
                        labels=labels,
                        data_type=data_type,
                        predictions=predictions,
                        generator=generator,
                        input_type=input_type,
                        output_type=output_type,
                        log_evaluation=log_evaluation,
                        validation_steps=validation_steps,
                        class_colors=class_colors,
                        log_batch_frequency=log_batch_frequency,
                        log_best_prefix=log_best_prefix)
                        
        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_examples = confusion_examples
        self.confusion_classes = confusion_classes
               
    def on_epoch_end(self, epoch, logs={}):
        if self.generator:
            self.validation_data = next(self.generator)

        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)
        
        if self.log_confusion_matrix:
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(self._log_confusion_matrix(), commit=False)                    

        if self.input_type in ("image", "images", "segmentation_mask") or self.output_type in ("image", "images", "segmentation_mask"):
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                if self.confusion_examples > 0:
                    wandb.log({'confusion_examples': self._log_confusion_examples(
                                                    confusion_classes=self.confusion_classes,
                                                    max_confused_examples=self.confusion_examples)}, commit=False)
                if self.predictions > 0:
                    wandb.log({"examples": self._log_images(
                        num_images=self.predictions)}, commit=False)

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
                        epoch, self.monitor, self.best, self.current))
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current
        
    def _log_confusion_matrix(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        n_confused = np.sum(confmatrix)
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.labels, 'y': self.labels, 'z': confmatrix,
                                 'hoverongaps':False, 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

        confdiag = confdiag.astype('float')
        n_right = np.sum(confdiag)
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.labels, 'y': self.labels, 'z': confdiag,
                               'hoverongaps':False, 'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        n_total = n_right + n_confused
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
        fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
        yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}

        fig.update_layout(title={'text':'Confusion matrix', 'x':0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
        
        return {'confusion_matrix': wandb.data_types.Plotly(fig)}

    def _log_confusion_examples(self, rescale=255, confusion_classes=5, max_confused_examples=3):
            x_val = self.validation_data[0]
            y_val = self.validation_data[1]
            y_val = np.argmax(y_val, axis=1)
            y_pred = np.argmax(self.model.predict(x_val), axis=1)

            # Grayscale to rgb
            if x_val.shape[-1] == 1:
                x_val = np.concatenate((x_val, x_val, x_val), axis=-1)

            confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
            np.fill_diagonal(confmatrix, 0)

            def example_image(class_index, x_val=x_val, y_pred=y_pred, y_val=y_val, labels=self.labels, rescale=rescale):
                image = None
                title_text = 'No example found'
                color = 'red'

                right_predicted_images = x_val[np.logical_and(y_pred==class_index, y_val==class_index)]
                if len(right_predicted_images) > 0:
                    image = rescale * right_predicted_images[0]
                    title_text = 'Predicted right'
                    color = 'rgb(46, 184, 46)'
                else:
                    ground_truth_images = x_val[y_val==class_index]
                    if len(ground_truth_images) > 0:
                        image = rescale * ground_truth_images[0]
                        title_text = 'Example'
                        color = 'rgb(255, 204, 0)'

                return image, title_text, color

            n_cols = max_confused_examples + 2
            subplot_titles = [""] * n_cols
            subplot_titles[-2:] = ["y_true", "y_pred"]
            subplot_titles[max_confused_examples//2] = "confused_predictions"
            
            n_rows = min(len(confmatrix[confmatrix > 0]), confusion_classes)
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
            for class_rank in range(1, n_rows+1):
                indx = np.argmax(confmatrix)
                indx = np.unravel_index(indx, shape=confmatrix.shape)
                if confmatrix[indx] == 0:
                    break
                confmatrix[indx] = 0

                class_pred, class_true = indx[0], indx[1]
                mask = np.logical_and(y_pred==class_pred, y_val==class_true)
                confused_images = x_val[mask]

                # Confused images
                n_images_confused = min(max_confused_examples, len(confused_images))
                for j in range(n_images_confused):
                    fig.add_trace(go.Image(z=rescale*confused_images[j],
                                        name=f'Predicted: {self.labels[class_pred]} | Instead of: {self.labels[class_true]}',
                                        hoverinfo='name', hoverlabel={'namelength' :-1}),
                                row=class_rank, col=j+1)
                    fig.update_xaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j+1, mirror=True)
                    fig.update_yaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j+1, mirror=True)

                # Comparaison images
                for i, class_index in enumerate((class_true, class_pred)):
                    col = n_images_confused+i+1
                    image, title_text, color = example_image(class_index)
                    fig.add_trace(go.Image(z=image, name=self.labels[class_index], hoverinfo='name', hoverlabel={'namelength' :-1}), row=class_rank, col=col)    
                    fig.update_xaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True, title_text=title_text)
                    fig.update_yaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True, title_text=self.labels[class_index])

            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            
            return wandb.data_types.Plotly(fig)
