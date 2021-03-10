import os
import shutil
import sys

# for Jupyter paths support, include other modules
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from yawn_train.src.train_dnn_model import ModelType
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

from yawn_train.src import train_utils
from yawn_train.src.model_config import MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS, \
    IMAGE_PAIR_SIZE

# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print('Tensorflow: ' + tf.__version__)
print('Keras: ' + tf.keras.__version__)

MODEL_PATH = '/Users/igla/Downloads/out_epoch_80_lite-5/yawn_model_80.h5'  # '/Users/igla/Downloads/out_epoch_80_lite-3/yawn_model_80.h5'
MOUTH_PREPARE_TEST_FOLDER = '/Users/igla/Downloads/Kaggle Drowsiness_dataset/yawn_with_faces'  # '/Users/igla/Downloads/mouth_state_new10_full/test'
THRESHOLD_CONF = 0.4
BATCH_SIZE = 32

TRAIN_MODEL = ModelType.LITE  # FULL, LITE, MOBILENET_V2
MODEL_PREFIX = TRAIN_MODEL.value
OUTPUT_FOLDER = f"./out_evaluation_kaggle_faces"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PATH_INCORRECT_PREDS = '/Users/igla/Downloads/YAWN_DS_DATASET/output_kaggle/'

PLOT_CONF_MATRIX_NORMALIZED = os.path.join(OUTPUT_FOLDER, 'plot_conf_matrix_normalize.png')
PLOT_CONF_MATRIX = os.path.join(OUTPUT_FOLDER, 'plot_conf_matrix.png')
PLOT_PREDICTIONS = os.path.join(OUTPUT_FOLDER, 'plot_predictions.png')
PLOT_MODEL_ARCH_PATH = os.path.join(OUTPUT_FOLDER, 'plot_model_arch.png')
PLOT_IMAGE_FREQ_PATH = os.path.join(OUTPUT_FOLDER, 'plot_dataset_frequency.png')
PLOT_IMAGE_PREVIEW = os.path.join(OUTPUT_FOLDER, 'plot_img_overview.png')

# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
# Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network.
# All images will be rescaled by 1./255
print('Create Train Image Data Generator')
# construct the image generator for data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255)

print('Create Test Data Generator')
test_generator = train_datagen.flow_from_directory(
    MOUTH_PREPARE_TEST_FOLDER,
    class_mode='binary',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    # don't shuffle
    shuffle=False,  # no shuffle as we use it to predict on test data
    # use same size as in training
    target_size=IMAGE_PAIR_SIZE  # All images will be resized to IMAGE_SHAPE
)

print('Preview 20 images from test generator')
train_utils.plot_data_generator_first_20(test_generator)

class_indices = test_generator.class_indices
print(class_indices)  # {'closed': 0, 'opened': 1}
class_names = list(class_indices.keys())
class_idx = list(class_indices.values())

print('Keys: ' + ','.join(map(str, list(test_generator.class_indices.keys()))))
print('Values: ' + ','.join(map(str, list(test_generator.class_indices.values()))))
print('First 10 images: ' + ','.join(map(str, test_generator.filenames[:10])))

number_of_examples = len(test_generator.filenames)
number_of_generator_calls = math.ceil(number_of_examples / (1.0 * BATCH_SIZE))
# 1.0 above is to skip integer division
test_labels = []
for i in range(0, int(number_of_generator_calls)):
    test_labels.extend(np.array(test_generator[i][1]))
test_labels = [int(round(x)) for x in test_labels]  # convert 1.0 to 1, 0.0 to 0.
print(f'Test labels, size={len(test_labels)}, first 100:')
print(test_labels[:100])

print('Resolve test images')
test_images = test_generator.filenames
test_images[:] = [f'{MOUTH_PREPARE_TEST_FOLDER}/{x}' for x in test_images]

# Create a basic model instance
input_shape = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS)

model = keras.models.load_model(MODEL_PATH)
# Check its architecture
model.summary()

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)

y_pred = predictions > 0.4
cnt = np.count_nonzero(y_pred)

train_utils.plot_roc(
    f"{OUTPUT_FOLDER}/plot_roc.png",
    test_generator.classes,
    predictions
)

train_utils.show_pred_actual_lables(
    PLOT_PREDICTIONS,
    predictions,
    test_labels,
    test_images,
    class_names,
    class_indices
)

# Predicting the classes of some images
for class_name in class_names:  # opened, closed
    train_utils.predict_random_test_img(model, MOUTH_PREPARE_TEST_FOLDER, class_name, True)


def evaluate_image(i, predictions_item, true_label_id, images, class_indices) -> (bool, bool, float):
    true_label_id, img = true_label_id[i], images[i]
    predicted_confidence = np.max(predictions_item)
    predicted_confidence = round(predicted_confidence, 2)
    is_mouth_opened = True if predicted_confidence >= THRESHOLD_CONF else False
    # classes taken from input data
    predicted_label_id = class_indices['opened' if is_mouth_opened else 'closed']
    is_correct_prediction = predicted_label_id == true_label_id
    return is_correct_prediction, is_mouth_opened, predicted_confidence


actual = np.array(test_labels)
predicted = []
for idx in range(len(predictions)):
    predicted.append(1 if (predictions[idx] >= THRESHOLD_CONF) else 0)
predicted = np.array(predicted)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(actual, predicted)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(actual, predicted)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(actual, predicted)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(actual, predicted)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(actual, predicted)
print('Cohens kappa: %f' % kappa)

import seaborn as sn
import pandas as pd

matrix = confusion_matrix(actual, predicted)
print(matrix)

# not normalize
index = ['closed', 'opened']
columns = ['closed', 'opened']
cm_df = pd.DataFrame(matrix, columns, index)
plt.figure()
sn.heatmap(cm_df, annot=True)
plt.savefig(PLOT_CONF_MATRIX)
plt.show()

# Normalise
cmn = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
target_names = ['closed', 'opened']
sn.set(font_scale=1.4)  # for label size
sn.heatmap(cmn, ax=ax, annot=True, annot_kws={"size": 16}, fmt='.2f', xticklabels=target_names,
           yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(PLOT_CONF_MATRIX_NORMALIZED)
plt.show(block=False)

print('Detect invalid predictions')
for idx in range(len(predictions)):
    is_correct_pred, mouth_opened, pred_conf = evaluate_image(idx, predictions[idx], test_labels, test_images,
                                                              class_indices)
    mouth_opened_str = '1' if mouth_opened else '0'
    if is_correct_pred is False:
        im_bad = test_images[idx]
        img_filename = os.path.basename(im_bad)
        sub_path = 'opened' if mouth_opened else 'closed'
        dst_folder = path = os.path.join(PATH_INCORRECT_PREDS, sub_path)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, str(pred_conf) + '_' + img_filename)
        shutil.copy(im_bad, dst_path)
