import os
import sys

from tensorflow import keras

from yawn_train.src.train_dnn_model import DNNTrainer, ModelType

# for Jupyter paths support, include other modules

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import os

import tensorflow as tf
from tensorflow.python.client import device_lib

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

MAX_IMAGE_HEIGHT = 100
MAX_IMAGE_WIDTH = 100
IMAGE_PAIR_SIZE = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
dnn_trainer = DNNTrainer(
    img_size=IMAGE_PAIR_SIZE,
    grayscale=False,
    data_folder='./../mouth_state',
    use_gpu=True,
    epochs=1,
    batch_size=32,
    learning_rate_opt=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
    train_model=ModelType.VGG16,
    include_optimizer=False,
    is_prune_model=False,
    model_name_prefix='yawn'
)
dnn_trainer.run_training()
