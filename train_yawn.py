import math
import os
import pickle
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

from yawn_train.convert_dataset_video_to_mouth_img import MOUTH_AR_THRESH


# https://medium.com/edureka/tensorflow-image-classification-19b63b7bfd95
# https://www.tensorflow.org/hub/tutorials/tf2_text_classification
# https://keras.io/examples/vision/image_classification_from_scratch/
# https://www.kaggle.com/darthmanav/dog-vs-cat-classification-using-cnn
# Input Layer: It represent input image data. It will reshape image into single diminsion array. Example your image is 100x100=10000, it will convert to (100,1) array.
# Conv Layer: This layer will extract features from image.
# Pooling Layer: This layer reduces the spatial volume of input image after convolution.
# Fully Connected Layer: It connect the network from a layer to another layer
# Output Layer: It is the predicted values layer.

def create_model(input_shape) -> keras.Model:
    # Note that when using the delayed-build pattern (no input shape specified),
    # the model gets built the first time you call `fit`, `eval`, or `predict`,
    # or the first time you call the model on some input data.
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # Layer 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # Layer 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # Layer 3
    model.add(layers.Flatten())  # Fully connected layer
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))  # Fully connected layer
    model.add(layers.Dropout(0.5))  # Fully connected layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Fully connected layer
    # compile model
    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_model_old(input_shape, num_classes) -> keras.Model:
    # Let's then add a Flatten layer that flattens the input image, which then feeds into the next layer, a Dense layer, or fully-connected layer, with 128 hidden units. Finally, because our goal is to perform binary classification, our final layer will be a sigmoid, so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is of class 1 (class 1 being grass and class 0 being dandelion).
    # model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    #                                   tf.keras.layers.Dense(128, activation=tf.nn.relu),
    #                                  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    model = keras.Model(inputs, outputs)

    # configure the specifications for model training. We will train our model with the binary_crossentropy loss. We will use the Adam optimizer. Adam is a sensible optimization algorithm because it automates learning-rate tuning for us
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])  # , 'mse', 'mae', 'mape'])
    return model


# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# OR disable cuda
# tf.config.experimental.set_visible_devices([], 'GPU')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from numba import cuda
# cuda.select_device(0)

print(device_lib.list_local_devices())
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

print('Tensorflow: ' + tf.__version__)
print('Keras: ' + tf.keras.__version__)

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set.
    # On Kaggle this is always the case.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

MOUTH_PREPARE_FOLDER = './mouth_state'

MOUTH_PREPARE_TRAIN_FOLDER = f'{MOUTH_PREPARE_FOLDER}/train'
MOUTH_PREPARE_TEST_FOLDER = f'{MOUTH_PREPARE_FOLDER}/test'
MOUTH_PREPARE_VAL_FOLDER = f'{MOUTH_PREPARE_FOLDER}/val'

MOUTH_FOLDER = "./mouth_state"
MOUTH_OPENED_FOLDER = f"{MOUTH_FOLDER}/opened"
MOUTH_CLOSED_FOLDER = f"{MOUTH_FOLDER}/closed"

# Hyperparameters
EPOCH = 1
BATCH_SIZE = 8

IMAGE_DIMENSION = 100
COLOR_CHANNELS = 1
IMAGE_PAIR_SIZE = (IMAGE_DIMENSION, IMAGE_DIMENSION)

OUTPUT_FOLDER = f"./out_epoch_{EPOCH}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TRAIN_HISTORY_DICT_DUMP = f'{OUTPUT_FOLDER}/train_history_dict.txt'
ONNX_MODEL_PATH = f"{OUTPUT_FOLDER}/yawn_model_onnx_{EPOCH}.onnx"
KERAS_MODEL_PATH = f"{OUTPUT_FOLDER}/yawn_model_{EPOCH}.h5"
TFLITE_QUANT_PATH = f"{OUTPUT_FOLDER}/yawn_model_quant_{EPOCH}.tflite"
TFLITE_FLOAT_PATH = f"{OUTPUT_FOLDER}/yawn_model_float_{EPOCH}.tflite"
TFLITE_FLOAT_PATH2 = f"{OUTPUT_FOLDER}/yawn_model_float2_{EPOCH}.tflite"
SAVED_MODEL = f"{OUTPUT_FOLDER}/saved_mouth_model_{EPOCH}"

print('First 10 opened images')
opened_eye_names = os.listdir(MOUTH_OPENED_FOLDER)
print(opened_eye_names[:10])
print('First 10 closed images')
closed_eye_names = os.listdir(MOUTH_CLOSED_FOLDER)
print(closed_eye_names[:10])


def show_img_preview():
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4
    # Index for iterating over images
    pic_index = 0
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    pic_index += 8
    next_eye_opened_pic = [os.path.join(MOUTH_OPENED_FOLDER, fname)
                           for fname in opened_eye_names[pic_index - 8:pic_index]]
    next_closed_eye_pic = [os.path.join(MOUTH_CLOSED_FOLDER, fname)
                           for fname in closed_eye_names[pic_index - 8:pic_index]]
    for i, img_path in enumerate(next_eye_opened_pic + next_closed_eye_pic):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        # sp.axis('Off')  # Don't show axes (or gridlines)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        img_filename = os.path.basename(img_path)
        filename_only = os.path.splitext(img_filename)[0]
        img_threshold = filename_only.split("_")
        conf = float(img_threshold[2])
        is_opened = "opened" if conf >= MOUTH_AR_THRESH else "closed"

        img = mpimg.imread(img_path)
        plt.xlabel("{} ({})".format(img_filename, is_opened, color='blue'))
        plt.imshow(img, cmap="gray")
    plt.savefig(f"{OUTPUT_FOLDER}/plot_img_overview.png")
    plt.show()


show_img_preview()

# TRAINING_DATA_DIR = str(base_dir)
# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

# Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network.
# All images will be rescaled by 1./255
print('Create Train Image Data Generator')
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)  # set validation split
train_generator = train_datagen.flow_from_directory(
    MOUTH_PREPARE_TRAIN_FOLDER,  # source directory for training images
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    shuffle=True,
    class_mode='binary',
    target_size=IMAGE_PAIR_SIZE  # All images will be resized to IMAGE_SHAPE
)
class_indices = train_generator.class_indices
print(class_indices)  # {'closed': 0, 'opened': 1}
class_names = list(class_indices.keys())
class_idx = list(class_indices.values())

print('Create Validation Image Data Generator')
valid_generator = train_datagen.flow_from_directory(
    MOUTH_PREPARE_VAL_FOLDER,
    class_mode='binary',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=False,  # no shuffle as we use it to predict on test data that must be in order
    target_size=IMAGE_PAIR_SIZE  # All images will be resized to IMAGE_SHAPE
)
print('Keys: ' + ','.join(map(str, list(valid_generator.class_indices.keys()))))
print('Values: ' + ','.join(map(str, list(valid_generator.class_indices.values()))))
print('First 10 images: ' + ','.join(map(str, valid_generator.filenames[:10])))

# print('Create Test Data Generator')
# test_generator = train_datagen.flow_from_directory(
#     MOUTH_PREPARE_TEST_FOLDER,
#     class_mode='binary',
#     color_mode='grayscale',
#     batch_size=BATCH_SIZE,
#     shuffle=False,  # no shuffle as we use it to predict on test data
#     target_size=image_size  # All images will be resized to IMAGE_SHAPE
# )
number_of_examples = len(valid_generator.filenames)
number_of_generator_calls = math.ceil(number_of_examples / (1.0 * BATCH_SIZE))
# 1.0 above is to skip integer division
test_labels = []
for i in range(0, int(number_of_generator_calls)):
    test_labels.extend(np.array(valid_generator[i][1]))
test_labels = [int(round(x)) for x in test_labels]  # convert 1.0 to 1, 0.0 to 0.
print(f'Test labels, size={len(test_labels)}, first 100:')
print(test_labels[:100])

print('Resolve test images')
test_images = valid_generator.filenames
test_images[:] = [f'{MOUTH_PREPARE_VAL_FOLDER}/{x}' for x in test_images]

print('Construct model')

# Create a basic model instance
input_shape = (IMAGE_DIMENSION, IMAGE_DIMENSION, COLOR_CHANNELS)
model = create_model(input_shape)  # create_model(input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 1), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
# Display the model's architecture
model.summary()

history = model.fit(train_generator,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=valid_generator)
# save history for later analysis and plot
with open(TRAIN_HISTORY_DICT_DUMP, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#  a graph of accuracy and loss over time
# plot the training and validation loss for comparison, as well as the training and validation accuracy
history_dict = history.history
history_dict.keys()


# plot diagnostic learning curves
def summarize_diagnostics(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot Epochs / Training and validation loss
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss (Cross Entropy Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{OUTPUT_FOLDER}/plot_epochs_loss.png")
    plt.show()

    # Plot Epochs / Training and validation accuracy
    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy (Classification Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{OUTPUT_FOLDER}/plot_epochs_accuracy.png")
    plt.show()


summarize_diagnostics(history_dict)

# evaluate the accuracy of our model:
final_loss, final_accuracy = model.evaluate(valid_generator)
print("Final loss: {:.2f}".format(final_loss))
print("Final accuracy: {:.2f}%".format(final_accuracy * 100))

STEP_SIZE_TEST = valid_generator.n // valid_generator.batch_size
valid_generator.reset()
predictions = model.predict(valid_generator, verbose=1)
fpr, tpr, _ = roc_curve(valid_generator.classes, predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f"{OUTPUT_FOLDER}/plot_roc.png")
plt.show()


def plot_image(i, predictions_item, true_label_id, images):
    true_label_id, img = true_label_id[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img_filename = os.path.basename(img)
    img_obj = mpimg.imread(img)
    plt.imshow(img_obj, cmap="gray")
    # predicted_label_id = np.argmax(predictions_item)  # take class with highest confidence
    predicted_confidence = np.max(predictions_item)
    predicted_score = 100 * predicted_confidence

    is_mouth_opened = True if predicted_confidence >= 0.2 else False
    # classes taken from input data
    predicted_label_id = class_indices['opened' if is_mouth_opened else 'closed']

    predicted_class = class_names[predicted_label_id]
    if predicted_label_id == true_label_id:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}, {} {:2.0f}% ({})".format(img_filename, predicted_class,
                                             predicted_score,
                                             class_names[true_label_id]),
               color=color)


def plot_value_array(predictions_array):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    predicted_confidence = np.max(predictions_array)
    thisplot = plt.bar(range(1), predicted_confidence, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 10
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    idx = random.choice(range(len(predictions)))
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(idx, predictions[idx], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(predictions[idx])
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/plot_predictions.png")
plt.show()


# Predicting the classes of images
# predictions = model.predict_generator(valid_generator)
# print('predictions shape:', predictions.shape)

def predict_image(model, input_img):
    loaded_img = keras.preprocessing.image.load_img(
        input_img, target_size=IMAGE_PAIR_SIZE, color_mode="grayscale"
    )
    img_array = keras.preprocessing.image.img_to_array(loaded_img)
    # scale pixel values to [0, 1]
    img_array = img_array.astype('float32')
    img_array /= 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    model_predictions = model.predict(img_array)
    score = model_predictions[0]
    print(
        "This image (%s) is %.2f percent closed mouth and %.2f percent opened mouth."
        % (input_img, 100 * score, 100 * (1 - score))
    )


closed_mouth_folder = f"{MOUTH_PREPARE_VAL_FOLDER}/closed/"
closed_mouth_img = os.listdir(closed_mouth_folder)[0]
closed_mouth_img = os.path.join(closed_mouth_folder, closed_mouth_img)
img = mpimg.imread(closed_mouth_img)
plt.imshow(img, cmap="gray")
plt.show()
predict_image(model, closed_mouth_img)

opened_mouth_folder = f"{MOUTH_PREPARE_VAL_FOLDER}/opened/"
opened_mouth_img = os.listdir(opened_mouth_folder)[0]
opened_mouth_img = os.path.join(opened_mouth_folder, opened_mouth_img)
img = mpimg.imread(opened_mouth_img)
plt.imshow(img, cmap="gray")
plt.show()
predict_image(model, opened_mouth_img)

# saved model
tf.keras.models.save_model(
    model,
    SAVED_MODEL,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save(KERAS_MODEL_PATH)

# Convert keras to onnx
# import keras2onnx
# keras_model = keras.models.load_model(KERAS_MODEL_PATH)
# onnx_model = keras2onnx.convert_keras(keras_model, ONNX_MODEL_PATH)

# https://github.com/ysh329/deep-learning-model-convertor
os.system("python -m tf2onnx.convert \
        --saved-model {saved_model} \
        --output {onnx}".format(saved_model=SAVED_MODEL, onnx=ONNX_MODEL_PATH))

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
open(TFLITE_QUANT_PATH, "wb").write(tflite_quant_model)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_float_model = converter.convert()
open(TFLITE_FLOAT_PATH, "wb").write(tflite_float_model)

# Create a concrete function from the SavedModel
model = tf.saved_model.load(SAVED_MODEL)
concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# Specify the input shape
concrete_func.inputs[0].set_shape([1, IMAGE_DIMENSION, IMAGE_DIMENSION, COLOR_CHANNELS])
# Convert the model and export
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Only for float16
tflite_model = converter.convert()
open(TFLITE_FLOAT_PATH2, "wb").write(tflite_model)
