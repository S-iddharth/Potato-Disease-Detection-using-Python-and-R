import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7

import os
import zipfile

train_dir = 'input your train_directory'
validation_dir = 'input your validation_directory'
test_dir = 'input your test_directory'

def extract_zip(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

extract_zip(train_zip, train_dir)
extract_zip(validation_zip, validation_dir)
extract_zip(test_zip, test_dir)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

test_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print("Test accuracy:", test_acc)

import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image

def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        return filename

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

image_file = upload_image()

img_array = preprocess_image(image_file)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

class_labels = train_generator.class_indices
for class_label, index in class_labels.items():
    if index == predicted_class:
        print("Predicted class:", class_label)
        break

img = image.load_img(image_file, target_size=IMAGE_SIZE)
plt.imshow(img)
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt

img_path = input("Enter path of image: ")
img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  
img_array /= 255. 

predictions = model.predict(img_array)

class_labels = train_generator.class_indices.keys()

plt.figure(figsize=(10, 5))
plt.bar(class_labels, predictions[0])
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.title('Probability Distribution of Uploaded Image')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()