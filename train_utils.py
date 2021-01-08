import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import backend as K

# plot diagnostic learning curves
from yawn_train.model_config import IMAGE_PAIR_SIZE


# https://medium.com/edureka/tensorflow-image-classification-19b63b7bfd95
# https://www.tensorflow.org/hub/tutorials/tf2_text_classification
# https://keras.io/examples/vision/image_classification_from_scratch/
# https://www.kaggle.com/darthmanav/dog-vs-cat-classification-using-cnn
# Input Layer: It represent input image data. It will reshape image into single diminsion array. Example your image is 100x100=10000, it will convert to (100,1) array.
# Conv Layer: This layer will extract features from image.
# Pooling Layer: This layer reduces the spatial volume of input image after convolution.
# Fully Connected Layer: It connect the network from a layer to another layer
# Output Layer: It is the predicted values layer.


def summarize_diagnostics(history_dict, output_folder):
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

    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    plt.title('Training and validation loss (Cross Entropy Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f"{output_folder}/plot_epochs_loss.png")
    plt.show()

    # Plot Epochs / Training and validation accuracy
    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    plt.title('Training and validation accuracy (Classification Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{output_folder}/plot_epochs_accuracy.png")
    plt.show()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


def evaluate_model(interpreter, test_images, test_labels):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    floating_model = interpreter.get_input_details()[0]['dtype'] == np.float32

    # Run predictions on ever y image in the "test" dataset.
    prediction_confs = []
    for i, test_image in enumerate(test_images):
        if i % 1000 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.

        # load image by path
        loaded_img = keras.preprocessing.image.load_img(
            test_image, target_size=IMAGE_PAIR_SIZE, color_mode="grayscale"
        )
        img_array = keras.preprocessing.image.img_to_array(loaded_img)
        img_array = img_array.astype('float32')

        if floating_model:
            # Normalize to [0, 1]
            image_frame = img_array / 255.0
            images_data = np.expand_dims(image_frame, 0).astype(np.float32)  # or [img_data]
        else:  # 0.00390625 * q
            images_data = np.expand_dims(img_array, 0).astype(np.uint8)  # or [img_data]
        interpreter.set_tensor(input_index, images_data)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        pred_confidence = np.argmax(output()[0])

        is_mouth_opened = True if pred_confidence >= 0.2 else False
        # classes taken from input data
        class_indices = {'closed': 0, 'opened': 1}
        predicted_label_id = class_indices['opened' if is_mouth_opened else 'closed']
        prediction_confs.append(predicted_label_id)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_confs = np.array(prediction_confs)
    accuracy = (prediction_confs == test_labels).mean()
    return accuracy


def prune_model(model, train_generator, batch_size: int, valid_generator, output_keras_prune):
    import tensorflow_model_optimization as tfmot

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    # Compute end step to finish pruning after 10 epochs.
    epochs = 10
    num_images = len(train_generator.filenames)
    end_step = np.ceil(num_images / (1.0 * batch_size)).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
    model_for_pruning.summary()

    logdir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]
    model_for_pruning.fit(train_generator,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=1,
                          validation_data=valid_generator,
                          callbacks=callbacks)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(
        valid_generator, verbose=1)
    # print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    tf.keras.models.save_model(model_for_export, output_keras_prune, include_optimizer=False)
    print('Saved pruned Keras model to:', output_keras_prune)


def convert_tf2onnx(saved_model, onnx_path):
    # Convert keras to onnx
    # import keras2onnx
    # keras_model = keras.models.load_model(KERAS_MODEL_PATH)
    # onnx_model = keras2onnx.convert_keras(keras_model, ONNX_MODEL_PATH)

    # https://github.com/ysh329/deep-learning-model-convertor
    os.system("python -m tf2onnx.convert \
            --saved-model {saved_model} \
            --output {onnx}".format(saved_model=saved_model, onnx=onnx_path))
    print('Saved onnx model to:', onnx_path)


def export_pb(saved_model, output_path):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = tf.saved_model.load(saved_model)
    full_model = tf.function(lambda x: model(x))
    f = full_model.get_concrete_function(
        tf.TensorSpec(shape=[None, 100, 100, 1],
                      dtype=tf.float32))
    f2 = convert_variables_to_constants_v2(f)
    graph_def = f2.graph.as_graph_def()
    # Export frozen graph
    with tf.io.gfile.GFile(output_path, 'wb') as f:
        f.write(graph_def.SerializeToString())
    print('Saved frozen model to:', output_path)
