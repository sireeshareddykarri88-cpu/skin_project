import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Dataset path
dataset_path = "dataset"

# Image size
IMG_SIZE = 224
BATCH_SIZE = 32

# Load dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Add layers
x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(
    train_data.num_classes,
    activation='softmax'
)(x)

# Final model
model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save model
model.save("model/skin_model.h5")

print("MODEL TRAINED SUCCESSFULLY")