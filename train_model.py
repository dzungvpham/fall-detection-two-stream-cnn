SEED = 47
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
import math
import h5py
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, Concatenate
from keras.engine.input_layer import Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

DATA_PATH = "datasets/FDD/fdd.hdf5" # Change dataset path here
WEIGHTS_PATH = "weights/weights.hdf5" # Change weight path here

def create_base_model(prefix = None, image_size = 224):
    base_model = MobileNetV2(
        input_shape = (image_size, image_size, 3), alpha = 1.0, depth_multiplier = 1,
        include_top = False, weights = "imagenet"
    )

    # Prefix all layers' names to avoid conflict
    if prefix != None:
        for layer in base_model.layers:
            layer.name = prefix + "_" + layer.name

    # Freeze base model's layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add top FC layers
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(256)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.5)(out)

    return base_model, out

def create_model(weights_path = None, image_size = 224, show_summary = False):
    spatial_stream, spatial_output = create_base_model(prefix = "spatial", image_size = image_size)
    temporal_stream, temporal_output = create_base_model(prefix = "temporal", image_size = image_size)
    out = Concatenate()([spatial_output, temporal_output])

    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.5)(out)

    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.5)(out)

    predictions = Dense(1, activation = "sigmoid")(out)
    model = Model(inputs = [spatial_stream.input, temporal_stream.input], outputs = predictions)

    if weights_path != None:
        model.load_weights(weights_path)
    if show_summary:
        model.summary()
    return model

def create_two_inputs_gen(X1, X2, labels, batch_size = 32, shuffle = True):
    gen = ImageDataGenerator(preprocessing_function = preprocess_input)
    X1_gen = gen.flow(X1, labels, seed = SEED, batch_size = batch_size, shuffle = shuffle)
    X2_gen = gen.flow(X2, seed = SEED, batch_size = batch_size, shuffle = shuffle)
    while True:
        X1_batch = X1_gen.next()
        X2_batch = X2_gen.next()
        yield [X1_batch[0], X2_batch], X1_batch[1]

def train_model(model = None):
    if not model:
        print("No model supplied!")
        return

    # Tune hyperparameters here
    EPOCHS = 1000
    LEARNING_RATE = 0.01
    DECAY = 0.001
    MOMENTUM = 0.95
    BATCH_SIZE = 48

    # Load data
    data = h5py.File(DATA_PATH, "r")
    X_rgb_train = data["data"]["rgb"]["train"]
    X_mhi_train = data["data"]["mhi"]["train"]
    y_train = data["labels"]["train"]
    X_rgb_val = data["data"]["rgb"]["val"]
    X_mhi_val = data["data"]["mhi"]["val"]
    y_val = data["labels"]["val"]

    # Initialize data generators
    train_gen = create_two_inputs_gen(
        X_rgb_train, X_mhi_train, y_train,
        batch_size = BATCH_SIZE, shuffle = True
    )
    validation_gen = create_two_inputs_gen(
        X_rgb_val, X_mhi_val, y_val,
        batch_size = BATCH_SIZE, shuffle = False
    )

    # Initialize callbacks
    checkpointer = ModelCheckpoint(
        filepath = "tmp/weights.hdf5", monitor = "val_acc",
        verbose = 1, save_best_only = True
    )
    early_stopper = EarlyStopping(
        monitor = "val_acc", min_delta = 0.001, patience = 60, verbose = 1
    )
    tensorboard = TensorBoard(log_dir = "logs/", write_graph = False)

    # Create optimizer and compile model
    optimizer = SGD(lr = LEARNING_RATE, decay = DECAY, momentum = MOMENTUM)
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

    # Fit model
    history = model.fit_generator(
        train_gen, epochs = EPOCHS, steps_per_epoch = math.ceil(len(X_rgb_train)/BATCH_SIZE),
        validation_data = validation_gen, validation_steps = math.ceil(len(X_rgb_val)/BATCH_SIZE),
        verbose = 2, callbacks = [checkpointer, early_stopper, tensorboard],
        class_weight = {0: 1.0, 1: 1.0} # 1 for fall, 0 for no fall
    )

    data.close()
    return model

def evaluate_model(model = None, validation = False):
    if model == None:
        print("Error: No model specified!")
        return

    # Load data
    data = h5py.File(DATA_PATH, "r")
    target = "val" if validation else "test"
    X_rgb_target = np.array(data["data"]["rgb"][target], dtype = np.float32)
    X_mhi_target = np.array(data["data"]["mhi"][target], dtype = np.float32)
    y_target = np.array(data["labels"][target])

    # Initialize data generator
    BATCH_SIZE = 32
    data_gen = create_two_inputs_gen(X_rgb_target, X_mhi_target, y_target, batch_size = BATCH_SIZE, shuffle = False)

    # Predict and calculate metrics
    predictions = model.predict_generator(data_gen, steps = math.ceil(len(y_target)/BATCH_SIZE), verbose = 1)
    predictions = np.array(np.round(predictions), dtype = bool)
    predictions = np.reshape(predictions, (len(predictions,)))
    y_target = np.array(y_target, dtype = bool)

    acc = np.average(predictions == y_target)
    sensitivity = np.sum(predictions & y_target) / np.sum(y_target) # Recall or true pos
    specificity = np.sum(~ (predictions | y_target)) / np.sum(~ y_target) # True neg
    precision = np.sum(predictions & y_target) / np.sum(predictions)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    return {
        "acc": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score
    }

    data.close()

if __name__ == "__main__":
    train_model(create_model())
    # print(evaluate_model(create_model(WEIGHTS_PATH)))
