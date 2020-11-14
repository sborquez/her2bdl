from nose.tools import raises, timed, eq_, ok_
from nose import with_setup
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from her2bdl.data import get_generators_from_tf_Dataset
from shutil import rmtree
import pathlib

# NOSE_DATA_PATH = "./.test_data"
NOSE_DATA_PATH = pathlib.Path(__file__).parent.joinpath(".extras", "test_data").resolve()
train_dataset, steps_per_epoch = (None, None)
val_dataset, validation_steps  = (None, None)
image, label = (None, None)


def setup_module():
    global train_dataset, steps_per_epoch
    global val_dataset, validation_steps
    global image, label
    # Shutup TF >:c
    tf.get_logger().setLevel(0)
    tf.autograph.set_verbosity(0)

    # Prepare dataset
    dataset_target = "simple"
    input_shape = (224, 224, 3)
    batch_size = 3
    num_classes = 10
    label_mode = "categorical"
    train_, val_ = get_generators_from_tf_Dataset(
        dataset_target, input_shape, batch_size,
        num_classes=num_classes, label_mode=label_mode,
        validation_split="sample", preprocessing= {"rescale": None}, data_dir=NOSE_DATA_PATH)
    (train_dataset, steps_per_epoch) = train_
    (val_dataset, validation_steps)  = val_
    for image, label in train_dataset.take(1).cache().repeat(): break


def teardown_module():
    global train_dataset, steps_per_epoch
    global val_dataset, validation_steps
    # Clean dataset
    #rmtree(NOSE_DATA_PATH)
    del train_dataset
    del val_dataset


"""
MCDropout Tests
===============
"""
y_predictions_samples = None
y_predictive_distribution = None

def predictive_testing_values():
    global y_predictions_samples
    global y_predictive_distribution
    T = 50
    y_predictions_samples = np.array([
        [[1.0, 0]] * T,                  # Low uncertainty and High confidence
        [[1/2, 1/2]] * T,                # High uncertainty and Low confidence
        [[1.0, 0], [0.0, 1.0]] * (T//2), # High uncertainty and High confidence
    ])
    y_predictive_distribution = y_predictions_samples.mean(axis=1)

@with_setup(setup=predictive_testing_values)
def test_MCDropout_predictive_entropy():
    from her2bdl.models import ModelMCDropout
    expected = np.array([0.0, 0.693, 0.693])
    H = ModelMCDropout.predictive_entropy(None, None, y_predictive_distribution)
    H = np.round(H, 3)
    ok_(np.all(expected == H), f"Entropy is not equal to expected value: H={H}")
    
@with_setup(setup=predictive_testing_values)
def test_MCDropout_variation_ratios():
    from her2bdl.models import ModelMCDropout
    vr = np.array([
        ModelMCDropout.variation_ratios(None, None, y_predictions_samples)
         for _ in range(100)
    ])
    vr = np.round(vr.mean(axis=0), 2)
    expected_mean = np.array([0.0, 0.33, 0.5])
    eq_(expected_mean[0], vr[0])
    ok_((0 < vr[1]) and (vr[1] <= 0.5))
    eq_(expected_mean[2], vr[2])

@with_setup(setup=predictive_testing_values)
def test_MCDropout_mutual_information():
    from her2bdl.models import ModelMCDropout
    expected = np.array([0.0, 0.0, 0.693])
    I = ModelMCDropout.mutual_information(None, None, y_predictive_distribution, y_predictions_samples)
    I = np.round(I, 3)
    ok_(np.all(expected == I), f"Mutual Information is not equal to expected value: I={I}")
    

"""
SimpleClassifier Tests
======================
"""

def build_model(model_constructor, model_parameters):
    model = model_constructor(**model_parameters)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam')
    return model

def test_SimpleClassifierMCDropout_output():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.5,
        }
    )
    # Model Output shape
    global image, label
    output_1 = model.predict(image)
    eq_(output_1.shape, label.shape)
    del model
    K.clear_session()


def test_SimpleClassifierMCDropout_stochastic():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.2,
        }
    )
    # Model Stochastic
    global image
    output_1 = model.predict(image)
    output_2 = model.predict(image)
    ok_(np.any(output_1 != output_2), "Models output is deterministic.")
    del model
    K.clear_session()

def test_SimpleClassifierMCDropout_uncertainty():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.5,
        }
    )
    # Model Uncertainty
    global image
    uncertainty = model.uncertainty(x=image, batch_size=16)
    del model
    K.clear_session()
    expected_columns = ['predictive entropy', 'mutual information', 'variation-ratios']
    expected_shape   = (len(image), len(expected_columns))
    eq_(expected_columns, list(uncertainty.columns))
    eq_(expected_shape, uncertainty.shape)


def test_SimpleClassifierMCDropout_fit():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.2,
        }
    )
    # Fit Model
    global train_dataset, steps_per_epoch
    global val_dataset, validation_steps
    history = model.fit(train_dataset, 
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, 
        validation_steps=validation_steps,
        epochs=1, 
        workers=1
    )
    evaluation = model.evaluate(val_dataset, steps=validation_steps)
    del model	
    K.clear_session()


"""
EfficientNetMCDropout Tests
==========================
"""

def test_EfficientNetMCDropout_output():
    from her2bdl.models import EfficientNetMCDropout
    model = build_model(
        model_constructor = EfficientNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "efficient_net_weights": 'imagenet'
        }
    )    
    # Model Output shape
    global image, label
    output_1 = model.predict(image)
    eq_(output_1.shape, label.shape)
    del model
    K.clear_session()


def test_EfficientNetMCDropout_output():
    from her2bdl.models import EfficientNetMCDropout
    model = build_model(
        model_constructor = EfficientNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "efficient_net_weights": 'imagenet'
        }
    )
    # Model Stochastic
    global image
    output_1 = model.predict(image)
    output_2 = model.predict(image)
    ok_(np.any(output_1 != output_2), "Models output is deterministic.")
    del model
    K.clear_session()

def test_EfficientNetMCDropout_uncertainty():
    from her2bdl.models import EfficientNetMCDropout
    model = build_model(
        model_constructor = EfficientNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "efficient_net_weights": 'imagenet'
        }
    )
    # Model Uncertainty
    global image
    uncertainty = model.uncertainty(x=image, batch_size=8)
    del model
    K.clear_session()
    expected_columns = ['predictive entropy', 'mutual information', 'variation-ratios']
    expected_shape   = (len(image), len(expected_columns))
    eq_(expected_columns, list(uncertainty.columns))
    eq_(expected_shape, uncertainty.shape)


def test_EfficientNetMCDropout_fit():
    from her2bdl.models import EfficientNetMCDropout
    model = build_model(
        model_constructor = EfficientNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "num_classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "efficient_net_weights": 'imagenet'
        }
    )
    # Fit Model
    global train_dataset, steps_per_epoch
    global val_dataset, validation_steps
    history = model.fit(train_dataset, 
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, 
        validation_steps=validation_steps,
        epochs=1, 
        workers=1
    )
    evaluation = model.evaluate(val_dataset, steps=validation_steps)
    del model
    K.clear_session()