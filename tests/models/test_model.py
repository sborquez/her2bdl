from nose.tools import assert_equals, raises, timed, eq_, ok_
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
    batch_size = 8
    num_clasess = 10
    label_mode = "categorical"
    train_, val_ = get_generators_from_tf_Dataset(
        dataset_target, input_shape, batch_size,
        num_classes=num_clasess, label_mode=label_mode,
        validation_split="sample", preprocessing= {"rescale": None}, data_dir=NOSE_DATA_PATH)
    (train_dataset, steps_per_epoch) = train_
    (val_dataset, validation_steps)  = val_
    for image, label in train_dataset.take(1).cache().repeat(): break


def teardown_module():
    global train_dataset, steps_per_epoch
    global val_dataset, validation_steps
    # Clean dataset
    rmtree(NOSE_DATA_PATH)
    del train_dataset
    del val_dataset


def build_model(model_constructor, model_parameters):
    model = model_constructor(**model_parameters)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam')
    return model


def test_SimpleClassifier_output():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "classes": 10,
            "mc_dropout_rate": 0.2,
        }
    )
    # Model Output shape
    global image, label
    output_1 = model.predict(image)
    eq_(output_1.shape, label.shape)
    del model
    K.clear_session()


def test_SimpleClassifier_uncertainty():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "classes": 10,
            "mc_dropout_rate": 0.2,
        }
    )
    # Model Uncertainty
    global image
    output_1 = model.predict(image)
    output_2 = model.predict(image)
    ok_(np.any(output_1 != output_2), "Models output is deterministic.")
    del model
    K.clear_session()


def test_SimpleClassifier_fit():
    from her2bdl.models import SimpleClassifierMCDropout
    model = build_model(
        model_constructor = SimpleClassifierMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "classes": 10,
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


def test_EfficentNetMCDropout_output():
    from her2bdl.models import EfficentNetMCDropout
    model = build_model(
        model_constructor = EfficentNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "weights": 'imagenet'
        }
    )    
    # Model Output shape
    global image, label
    output_1 = model.predict(image)
    eq_(output_1.shape, label.shape)
    del model
    K.clear_session()


def test_EfficentNetMCDropout_uncertainty():
    from her2bdl.models import EfficentNetMCDropout
    model = build_model(
        model_constructor = EfficentNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "weights": 'imagenet'
        }
    )
    # Model Uncertainty
    global image
    output_1 = model.predict(image)
    output_2 = model.predict(image)
    ok_(np.any(output_1 != output_2), "Models output is deterministic.")
    del model
    K.clear_session()


def test_EfficentNetMCDropout_fit():
    from her2bdl.models import EfficentNetMCDropout
    model = build_model(
        model_constructor = EfficentNetMCDropout, 
        model_parameters = {
            "input_shape" : (224, 224, 3),
            "classes": 10,
            "mc_dropout_rate": 0.2,
            "base_model": "B0", 
            "weights": 'imagenet'
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