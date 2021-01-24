import os
import sys

# for Jupyter paths support, include other modules
from tensorflow.python.keras.callbacks import CSVLogger

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.python.client import device_lib

from yawn_train import train_utils
from yawn_train.model_config import MOUTH_AR_THRESH, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS, IMAGE_PAIR_SIZE

# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# OR disable cuda
# tf.config.experimental.set_visible_devices([], 'GPU')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from numba import cuda
# cuda.select_device(0)

# find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)
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
MOUTH_PREPARE_TRAIN_FOLDER = os.path.join(MOUTH_PREPARE_FOLDER, 'train')
MOUTH_PREPARE_TEST_FOLDER = os.path.join(MOUTH_PREPARE_FOLDER, 'test')
MOUTH_PREPARE_VAL_FOLDER = os.path.join(MOUTH_PREPARE_FOLDER, 'val')

MOUTH_FOLDER = "./mouth_state"
MOUTH_OPENED_FOLDER = os.path.join(MOUTH_FOLDER, 'opened')
MOUTH_CLOSED_FOLDER = os.path.join(MOUTH_FOLDER, 'closed')

# Hyperparameters
EPOCH = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.001

EARLY_STOP = False
TRAIN_LITE_MODEL = False
INCLUDE_OPTIMIZER_WEIGHTS = False
IS_PRUNE_MODEL = False
IS_EVALUATE_TFLITE = False

MODEL_PREFIX = 'lite' if TRAIN_LITE_MODEL else 'pro'
OUTPUT_FOLDER = f"./out_epoch_{EPOCH}_{MODEL_PREFIX}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PLOT_PREDICTIONS = os.path.join(OUTPUT_FOLDER, 'plot_predictions.png')
PLOT_MODEL_ARCH_PATH = os.path.join(OUTPUT_FOLDER, 'plot_model_arch.png')
PLOT_IMAGE_FREQ_PATH = os.path.join(OUTPUT_FOLDER, 'plot_dataset_frequency.png')
PLOT_IMAGE_PREVIEW = os.path.join(OUTPUT_FOLDER, 'plot_img_overview.png')
TRAIN_HISTORY_CSV = os.path.join(OUTPUT_FOLDER, 'train_history.csv')
ONNX_MODEL_PATH = os.path.join(OUTPUT_FOLDER, f"yawn_model_onnx_{EPOCH}.onnx")
KERAS_MODEL_PATH = os.path.join(OUTPUT_FOLDER, f"yawn_model_{EPOCH}.h5")
FROZEN_MODEL_PATH = os.path.join(OUTPUT_FOLDER, f"yawn_model_{EPOCH}.pb")
KERAS_PRUNE_MODEL_PATH = os.path.join(OUTPUT_FOLDER, f"yawn_model_prune_{EPOCH}.h5")
TFLITE_QUANT_PATH = os.path.join(OUTPUT_FOLDER, f"yawn_model_quant_{EPOCH}.tflite")
TFLITE_FLOAT_PATH = os.path.join(OUTPUT_FOLDER, f"yawn_model_float_{EPOCH}.tflite")
TFLITE_FLOAT_PATH2 = os.path.join(OUTPUT_FOLDER, f"yawn_model_float2_{EPOCH}.tflite")
SAVED_MODEL = os.path.join(OUTPUT_FOLDER, f"saved_mouth_model_{EPOCH}")
TFJS_MODEL = os.path.join(OUTPUT_FOLDER, f"tfjs_model_{EPOCH}")

print('First 10 opened images')
opened_eye_img_paths = train_utils.listdir_fullpath(MOUTH_OPENED_FOLDER)
opened_eye_img_names = [os.path.basename(f) for f in opened_eye_img_paths[:10]]
print(opened_eye_img_names)
print()

print('First 10 closed images')
closed_eye_img_paths = train_utils.listdir_fullpath(MOUTH_CLOSED_FOLDER)
closed_eye_img_names = [os.path.basename(f) for f in closed_eye_img_paths[:10]]
print(closed_eye_img_names)
print()

train_utils.plot_freq_imgs(
    PLOT_IMAGE_FREQ_PATH,
    opened_eye_img_paths,
    closed_eye_img_paths
)

train_utils.show_img_preview(
    PLOT_IMAGE_PREVIEW,
    opened_eye_img_paths, closed_eye_img_paths,
    MOUTH_AR_THRESH
)

# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
# Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network.
# All images will be rescaled by 1./255
print('Create Train Image Data Generator')
# construct the image generator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,  # randomly rotate image
    rescale=1. / 255,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",  # zoom_range will corrupt img
    preprocessing_function=train_utils.add_noise)
train_generator = train_datagen.flow_from_directory(
    MOUTH_PREPARE_TRAIN_FOLDER,  # source directory for training images
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    shuffle=True,
    class_mode='binary',
    target_size=IMAGE_PAIR_SIZE  # All images will be resized to IMAGE_SHAPE
)

print('Preview 20 images from train generator')
train_utils.plot_dataget_first_20(train_generator)

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

print('Create model')
# Create a basic model instance
input_shape = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS)
model = train_utils.create_compiled_model_lite(input_shape, LEARNING_RATE) \
    if TRAIN_LITE_MODEL else train_utils.create_compiled_model(input_shape, LEARNING_RATE)
keras.utils.plot_model(model, show_shapes=True)
# Display the model's architecture
model.summary()

# use CSVLogger callback to save training history
csv_logger = CSVLogger(TRAIN_HISTORY_CSV, append=True, separator=';')
# updatelr = LearningRateScheduler(train_utils.lr_scheduler, verbose=1)
printlr = train_utils.printlearningrate()

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
callbacks = [
    csv_logger,
    # updatelr,
    printlr
]
if EARLY_STOP:
    callbacks.append(es_callback)
history = model.fit(train_generator,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=valid_generator,
                    callbacks=callbacks)

#  a graph of accuracy and loss over time
# plot the training and validation loss for comparison, as well as the training and validation accuracy
history_dict = history.history
history_dict.keys()

plot_accuracy_path = f"{OUTPUT_FOLDER}/plot_epochs_accuracy.png"
plot_loss_path = f"{OUTPUT_FOLDER}/plot_epochs_loss.png"
plot_lr_path = f"{OUTPUT_FOLDER}/plot_lr.png"
train_utils.summarize_diagnostics(
    history_dict,
    plot_accuracy_path,
    plot_loss_path,
    plot_lr_path
)

# evaluate the accuracy of our model:
final_loss, final_accuracy, f1_score, precision, recall = model.evaluate(valid_generator, verbose=1)
print("Final loss: {:.2f}".format(final_loss))
print("Final accuracy: {:.2f}%".format(final_accuracy * 100))
print("F1: {:.2f}".format(f1_score))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

STEP_SIZE_TEST = valid_generator.n // valid_generator.batch_size
valid_generator.reset()
predictions = model.predict(valid_generator, verbose=1)

train_utils.plot_roc(
    f"{OUTPUT_FOLDER}/plot_roc.png",
    valid_generator.classes, predictions
)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 10
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    idx = random.choice(range(len(predictions)))
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    is_correct_pred = train_utils.plot_image(idx, predictions[idx], test_labels, test_images, class_names,
                                             class_indices)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    train_utils.plot_value_array(predictions[idx], is_correct_pred)
plt.tight_layout()
plt.savefig(PLOT_PREDICTIONS)
plt.show()

# Predicting the classes of some images
for class_name in class_names:  # opened, closed
    train_utils.predict_random_test_img(model, MOUTH_PREPARE_TEST_FOLDER, class_name)

# saved model
tf.keras.models.save_model(
    model,
    SAVED_MODEL,
    overwrite=True,
    include_optimizer=INCLUDE_OPTIMIZER_WEIGHTS,
    save_format=None,
    signatures=None,
    options=None
)

# Plot model architecture
tf.keras.utils.plot_model(
    model,
    to_file=PLOT_MODEL_ARCH_PATH,
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

train_utils.export_pb(SAVED_MODEL, FROZEN_MODEL_PATH)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
# Set include_optimizer=False to reduce output model size (e.g. 19.7mb -> 9.8mb)
model.save(KERAS_MODEL_PATH, include_optimizer=INCLUDE_OPTIMIZER_WEIGHTS)
print('Saved keras model to:', KERAS_MODEL_PATH)

if IS_PRUNE_MODEL:
    # prune model
    print('Prune model')
    train_utils.prune_model(model, train_generator, BATCH_SIZE, valid_generator, KERAS_PRUNE_MODEL_PATH)

train_utils.convert_tf2onnx(SAVED_MODEL, ONNX_MODEL_PATH)

# convert to js format
train_utils.export_tf_js(model, TFJS_MODEL)

train_utils.export_tflite_quant(TFLITE_QUANT_PATH, SAVED_MODEL)
if IS_EVALUATE_TFLITE:
    train_utils.evaluate_tflite_quant(TFLITE_QUANT_PATH, test_images, test_labels)

train_utils.export_tflite_floating(TFLITE_FLOAT_PATH, SAVED_MODEL)
if IS_EVALUATE_TFLITE:
    train_utils.evaluate_tflite_float(TFLITE_FLOAT_PATH, test_images, test_labels)

# Create a concrete function from the SavedModel
train_utils.export_tflite_floating2(
    TFLITE_FLOAT_PATH2,
    SAVED_MODEL,
    [1, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS]
)
