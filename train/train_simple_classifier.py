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
EPOCHS = 5
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_RESCALE = None
IMAGE_FOLDER = "D:/sebas/Desktop/simple_images/classes"
#MODEL = "SimpleClassifierMCDropout"
MODEL = "EfficentNetMCDropout"
INCLUDE_UNCERTAINTY_test = True

input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
data_dir = pathlib.Path(IMAGE_FOLDER)
num_classes = len(list(data_dir.glob('*/')))
model_constructor = MODELS[MODEL]
model_parameters = {
    "input_shape" : input_shape,
    "classes": num_classes,
    "mc_dropout_rate": 0.2,
    #"base_model": "B0", 
    #"weights": 'imagenet'
}

#STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# def preprocess(img):
#     img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
#     return img.astype(float) / 255.0

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                              #horizontal_flip=True,
                                              validation_split=0.2,
                                              rescale=IMG_RESCALE)
                                              #preprocessing_function=preprocess)

train_generator = image_generator.flow_from_directory(
    directory=str(data_dir),
    batch_size=BATCH_SIZE,
    shuffle=True,
    #class_mode="categorical",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    subset='training'
)

val_generator = image_generator.flow_from_directory(
    directory=str(data_dir),
    batch_size=BATCH_SIZE,
    shuffle=True,
    #class_mode="categorical",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    subset='validation'
)

def callable_iterator(generator):
    for img_batch, targets_batch in generator:
        yield img_batch, targets_batch

train_dataset = tf.data.Dataset.from_generator(lambda: callable_iterator(train_generator),
                                            output_types=(tf.float32, tf.float32), output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3],
                                    [None, num_classes]))
val_dataset = tf.data.Dataset.from_generator(lambda: callable_iterator(val_generator),
                        output_types=(tf.float32, tf.float32), output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3],
                                    [None, num_classes]))
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

if INCLUDE_UNCERTAINTY_test:
    output_1 = model.predict(image)
    output_2 = model.predict(image)
    print("is stochastic:", np.any(output_1 != output_2))
# %%
history = model.fit(train_dataset, 
    steps_per_epoch=len(train_generator),
    validation_data=val_dataset, 
    validation_steps=len(val_generator),
    epochs=EPOCHS, 
    workers=1
)

#%%
targets     = []
predictions = []
for image, label in val_dataset.take(len(val_generator)):
    target = label.numpy().argmax(axis=-1)
    targets.append(target)
    pred = model.predict(image).argmax(axis=-1)
    predictions.append(pred)
targets = np.hstack(targets)
predictions = np.hstack(predictions)
cm = confusion_matrix(targets, predictions)

evaluation = model.evaluate(val_dataset, steps=len(val_generator))

print("Confusion matrix")
print(cm)
print("Evaluation:", evaluation)
#%%
# TODO: Add uncertainty matrix
# TODO: Add uncertainty evaluation

