#%%
import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import pathlib
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg') 

from her2bdl import *

# Training Parameters
#print(MODELS)
SEED = 420
BATCH_SIZE = 32
EPOCHS = 1
VALIDATION_SPLIT=0.2
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_RESCALE = None
#IMAGE_FOLDER = "D:/sebas/Desktop/simple_images/classes"
DATASET_TARGET = "simple"
#MODEL = "SimpleClassifierMCDropout"
MODEL = "EfficentNetMCDropout"
INCLUDE_UNCERTAINTY_test = True

input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
#num_classes = len(list(data_dir.glob('*/')))
num_classes = 10

#data_dir = pathlib.Path(IMAGE_FOLDER)
model_constructor = MODELS[MODEL]
model_parameters = {
    "input_shape" : input_shape,
    "num_classes": num_classes,
    "mc_dropout_rate": 0.2,
    #"base_model": "B0", 
    #"weights": 'imagenet'
}

# train_, val_ = get_generators_from_directory(
#     IMAGE_FOLDER, input_shape, BATCH_SIZE, 
#     rescale=IMG_RESCALE, 
#     validation_split=VALIDATION_SPLIT
# )

train_, val_ = get_generators_from_tf_Dataset(
    DATASET_TARGET, input_shape, BATCH_SIZE, 
    preprocessing={'rescale' : IMG_RESCALE}, 
    validation_split=VALIDATION_SPLIT
)

#
(train_dataset, steps_per_epoch) = train_
(val_dataset, validation_steps)  = val_


#%%
from PIL import Image
for image, label in train_dataset.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
  img = image.numpy()[0].astype(np.uint8)
  img = Image.fromarray(img)
  #img.show()
  #plt.show()

# %%
model = model_constructor(**model_parameters)
model.build(input_shape=(None, *input_shape))
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam')
model.summary()
output = model.predict(image)
print("Model output shape:", output.shape)
#%%
if INCLUDE_UNCERTAINTY_test:
    output_1 = model.predict(image)
    output_2 = model.predict(image)
    print("is stochastic:", np.any(output_1 != output_2))
# %%
history = model.fit(train_dataset, 
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset, 
    validation_steps=validation_steps,
    epochs=EPOCHS, 
    workers=1
)
targets     = []
predictions = []
for image, label in val_dataset.take(validation_steps):
    target = label.numpy().argmax(axis=-1)
    targets.append(target)
    pred = model.predict(image).argmax(axis=-1)
    predictions.append(pred)
targets = np.hstack(targets)
predictions = np.hstack(predictions)
cm = confusion_matrix(targets, predictions)

evaluation = model.evaluate(val_dataset, steps=validation_steps)

print("Confusion matrix")
print(cm)
print("Evaluation:", evaluation)
#%%
# TODO: Add uncertainty matrix
# TODO: Add uncertainty evaluation

