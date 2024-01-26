from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split

img_width, img_height = 128, 128
batch_size = 64
num_epochs = 100
model_save_path = '/content/drive/MyDrive/tomato/modified_cnn_model.h5'
train_dir = '/content/drive/MyDrive/tomato/train'
test_dir = '/content/drive/MyDrive/tomato/val'

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)

# No augmentation for vala and test and just normalization
normalize_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create separate data generators for training and testing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training', 
    seed=42
)

val_generator = normalize_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False, 
    subset='validation',
    seed=42
)

# Load the validation
test_generator = normalize_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

#Visualize data before training
sample_images, sample_labels = next(train_generator)
class_names = list(train_generator.class_indices.keys())
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    class_index = np.argmax(sample_labels[i])
    class_name = class_names[class_index]
    plt.title(f'Class: {class_name}')
    plt.axis('off')
plt.show()


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs, 
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

model.save(model_save_path)

#plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

samples = 9
test_predictions = model.predict(test_generator)
actual_labels = test_generator.classes
predicted_labels = np.argmax(test_predictions, axis=1)
class_names = list(train_generator.class_indices.keys())

plt.figure(figsize=(12, 6))
for i in range(samples):
    plt.subplot(3, 3, i + 1)
    rand_index = np.random.randint(0, len(actual_labels))
    plt.imshow(test_generator[0][0][rand_index])
    actual_class = class_names[actual_labels[rand_index]]
    predicted_class = class_names[predicted_labels[rand_index]]
    title = f'Actual: {actual_class}\nPredicted: {predicted_class}\
    plt.title(title)
    plt.axis('off')

plt.show()
