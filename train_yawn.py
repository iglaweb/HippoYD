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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
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

OUTPUT_FOLDER = f"./out_epoch_{EPOCH}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IS_PRUNE_MODEL = False
IS_EVALUATE_TFLITE = False

TRAIN_HISTORY_CSV = f'{OUTPUT_FOLDER}/train_history.csv'
ONNX_MODEL_PATH = f"{OUTPUT_FOLDER}/yawn_model_onnx_{EPOCH}.onnx"
KERAS_MODEL_PATH = f"{OUTPUT_FOLDER}/yawn_model_{EPOCH}.h5"
FROZEN_MODEL_PATH = f"{OUTPUT_FOLDER}/yawn_model_{EPOCH}.pb"
KERAS_PRUNE_MODEL_PATH = f"{OUTPUT_FOLDER}/yawn_model_prune_{EPOCH}.h5"
TFLITE_QUANT_PATH = f"{OUTPUT_FOLDER}/yawn_model_quant_{EPOCH}.tflite"
TFLITE_FLOAT_PATH = f"{OUTPUT_FOLDER}/yawn_model_float_{EPOCH}.tflite"
TFLITE_FLOAT_PATH2 = f"{OUTPUT_FOLDER}/yawn_model_float2_{EPOCH}.tflite"
SAVED_MODEL = f"{OUTPUT_FOLDER}/saved_mouth_model_{EPOCH}"
TFJS_MODEL = f"{OUTPUT_FOLDER}/tfjs_model_{EPOCH}"

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
input_shape = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS)
model = train_utils.create_model(
    input_shape)  # create_model(input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 1), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
# Display the model's architecture
model.summary()

# use CSVLogger callback to save training history
csv_logger = CSVLogger(TRAIN_HISTORY_CSV, append=True, separator=';')
history = model.fit(train_generator,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=valid_generator,
                    callbacks=[csv_logger])

#  a graph of accuracy and loss over time
# plot the training and validation loss for comparison, as well as the training and validation accuracy
history_dict = history.history
history_dict.keys()

train_utils.summarize_diagnostics(history_dict, OUTPUT_FOLDER)

# evaluate the accuracy of our model:
final_loss, final_accuracy = model.evaluate(valid_generator, verbose=1)
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


def plot_image(i, predictions_item, true_label_id, images) -> bool:
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
    is_correct_prediction = predicted_label_id == true_label_id
    if is_correct_prediction:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}, {} {:2.0f}% ({})".format(img_filename, predicted_class,
                                             predicted_score,
                                             class_names[true_label_id]),
               color=color)
    return is_correct_prediction


def plot_value_array(predictions_array, is_correct_prediction: bool):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    predicted_confidence = np.max(predictions_array)
    thisplot = plt.bar(range(1), predicted_confidence, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    if is_correct_prediction:
        color = 'blue'
    else:
        color = 'red'
    thisplot[predicted_label].set_color(color)


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 10
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    idx = random.choice(range(len(predictions)))
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    is_correct_pred = plot_image(idx, predictions[idx], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(predictions[idx], is_correct_pred)
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

train_utils.export_pb(SAVED_MODEL, FROZEN_MODEL_PATH)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
# Set include_optimizer=False to reduce output model size (e.g. 19.7mb -> 9.8mb)
model.save(KERAS_MODEL_PATH, include_optimizer=False)
print('Saved keras model to:', KERAS_MODEL_PATH)

if IS_PRUNE_MODEL:
    # prune model
    print('Prune model')
    train_utils.prune_model(model, train_generator, BATCH_SIZE, valid_generator, KERAS_PRUNE_MODEL_PATH)

train_utils.convert_tf2onnx(SAVED_MODEL, ONNX_MODEL_PATH)

# convert to js format
try:
    import tensorflowjs as tfjs
except ImportError:
    tfjs = None
if tfjs:
    tfjs.converters.save_keras_model(model, TFJS_MODEL)
    print('Saved TFJS model to:', TFJS_MODEL)
else:
    print("You could convert tfjs right now, if you had tensorflowjs module.")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
with open(TFLITE_QUANT_PATH, "wb") as w:
    w.write(tflite_quant_model)
print('Saved quantized TFLite model to:', TFLITE_QUANT_PATH)

if IS_EVALUATE_TFLITE:
    # test tflite quality
    print('Evaluate quant TFLite model')
    interpreter_quant = tf.lite.Interpreter(model_content=tflite_quant_model)
    interpreter_quant.allocate_tensors()
    test_accuracy_tflite_q = train_utils.evaluate_model(interpreter_quant, test_images, test_labels)
    print('Quantized TFLite test_accuracy:', test_accuracy_tflite_q)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_float_model = converter.convert()
with open(TFLITE_FLOAT_PATH, "wb") as w:
    w.write(tflite_float_model)
print('Saved floating TFLite model to:', TFLITE_FLOAT_PATH)

if IS_EVALUATE_TFLITE:
    # test tflite quality
    print('Evaluate float TFLite model')
    interpreter_float = tf.lite.Interpreter(model_content=tflite_float_model)
    interpreter_float.allocate_tensors()
    test_accuracy_tflite_f = train_utils.evaluate_model(interpreter_float, test_images, test_labels)
    print('Floating TFLite test_accuracy:', test_accuracy_tflite_f)

# Create a concrete function from the SavedModel
model = tf.saved_model.load(SAVED_MODEL)
concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# Specify the input shape
concrete_func.inputs[0].set_shape([1, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS])
# Convert the model and export
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Only for float16
tflite_model = converter.convert()
with open(TFLITE_FLOAT_PATH2, "wb") as w:
    w.write(tflite_model)
print('Saved TFLite floating16 to:', TFLITE_FLOAT_PATH2)
