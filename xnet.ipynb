import tensorflow as tf
import os
import random
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras_tuner import Hyperband
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Input, Concatenate, UpSampling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

base_dir = r'/content/drive/MyDrive/MulticlassDataset'
os.makedirs(base_dir, exist_ok=True)

train_dir = r'/content/drive/MyDrive/MulticlassDataset/train/normal'
val_dir = r'/content/drive/MyDrive/MulticlassDataset/val/normal'
test_dir = r'/content/drive/MyDrive/MulticlassDataset/test/normal'

img_width, img_height = 224, 224
batch_size = 32
epochs = 50
learning_rate = 0.001
num_classes = 7
dropout_rate = 0.5

filter_sizes = [32, 64, 128]
kernel_sizes = [(3, 3), (5, 5)]
dropout_rates = [0.3, 0.5, 0.7]
learning_rates = [0.0001, 0.00001, 0.001]

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

def build_model(input_shape=(224, 224, 3), classes=7, kernel_size=3, filter_depth=(64, 128, 256)):
    img_input = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(img_input)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(batch1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1)

    conv2 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(pool1)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation("relu")(batch2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act2)

    conv3 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(pool2)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation("relu")(batch3)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(act3)
    conv4 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(up1)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation("relu")(batch4)

    up2 = UpSampling2D(size=(2, 2))(act4)
    conv5 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(up2)
    batch5 = BatchNormalization()(conv5)
    act5 = Activation("relu")(batch5)

    # Output layer
    conv6 = Conv2D(classes, (1, 1), activation="softmax", padding="same")(act5)
    output = tf.keras.layers.GlobalAveragePooling2D()(conv6)  # Directly reduce dimensions
    model = Model(img_input, output)

    return model

def random_search(num_trials=5):
    best_model = None
    best_acc = 0
    best_params = {}

    for trial in range(num_trials):
        # Randomly select hyperparameters for this trial
        filters = random.choice(filter_sizes)
        kernel_size = random.choice(kernel_sizes)
        dropout_rate = random.choice(dropout_rates)
        lr = random.choice(learning_rates)

        print(f"Trial {trial+1}: filters={filters}, kernel_size={kernel_size}, dropout_rate={dropout_rate}, learning_rate={lr}")

        # Use build_model instead of model
        model = build_model(input_shape=(img_width, img_height, 3), classes=num_classes)

        # Update optimizer with current learning rate
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model and evaluate
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, model_checkpoint]
        )

        # Get the validation accuracy
        val_acc = max(history.history['val_accuracy'])

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_params = {
                'filters': filters,
                'kernel_size': kernel_size,
                'dropout_rate': dropout_rate,
                'learning_rate': lr
            }

    return best_model, best_params, best_acc

checkpoint_path = os.path.join(base_dir, "best_model2.keras")
model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

best_model, best_params, best_acc = random_search()
