import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential

physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_folder = 'C:/Users/rishk/OneDrive/train_folder'
valid_folder = 'C:/Users/rishk/OneDrive/test_folder'

width = 448
height = 448
n_channels = 3

batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_folder,
    image_size=(width, height),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    valid_folder,
    image_size=(width, height),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False
)

n_categories = len(train_dataset.class_names)

print(n_categories)

class_names = train_dataset.class_names

with tf.device('/GPU:0'):
    optimizer = Adam(learning_rate=0.001)

    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(width, height, n_channels))

    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(n_categories, activation='softmax'))

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 4
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset
)

prediction_classes = np.array([])
true_classes = np.array([])

for x, y in valid_dataset:
    prediction_classes = np.concatenate([prediction_classes,
                                         np.argmax(model.predict(x), axis=-1)])
    true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])

print(classification_report(true_classes, prediction_classes, target_names=class_names))
