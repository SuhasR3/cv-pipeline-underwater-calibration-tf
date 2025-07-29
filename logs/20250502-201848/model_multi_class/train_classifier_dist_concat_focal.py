import os
import datetime
import random
import numpy as np
import glob
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2

# --- GPU Configuration ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)

# --- Paths and Params ---
model_name = 'model_multi_class/'
SAVE = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
output_folder = SAVE + model_name
output_log = output_folder + "Log/"
output_weight = output_folder + "Best/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_log, exist_ok=True)
os.makedirs(output_weight, exist_ok=True)

batch_size = 64
nb_epoch = 10
image_shape = (299, 299, 3)
IMAGE_FILE_PATH_DISTORTED = "dataset/dataset_discrete/"

# --- Label Classes ---
classes_focal = list(np.arange(40, 501, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)


# --- Custom Data Generator Class using tf.keras.utils.Sequence ---
class RotNetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_data, labels, num_focal, num_distortion, batch_size, shuffle=True):
        self.input_data = input_data
        self.labels = labels
        self.num_focal = num_focal
        self.num_distortion = num_distortion
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.input_data) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_inputs = self.input_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X_images = np.empty((len(batch_inputs), 299, 299, 3), dtype=np.float32)
        X_focal_onehot = np.empty((len(batch_inputs), self.num_focal), dtype=np.float32)
        y_distortion = np.empty((len(batch_inputs), self.num_distortion), dtype=np.float32)
        
        for i, ((img_path, focal_class), distortion_class) in enumerate(zip(batch_inputs, batch_labels)):
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                # Create a blank image as fallback
                image = np.zeros((299, 299, 3), dtype=np.uint8)
            else:
                image = cv2.resize(image, (299, 299))
            
            # Apply preprocessing
            X_images[i] = preprocess_input(image.astype(np.float32))
            
            # One-hot encode focal class
            X_focal_onehot[i] = to_categorical(focal_class, num_classes=self.num_focal)
            
            # One-hot encode distortion class
            y_distortion[i] = to_categorical(distortion_class, num_classes=self.num_distortion)
        
        # Return as a dictionary for inputs and a single numpy array for outputs
        return {"main_input": X_images, "concat_input": X_focal_onehot}, y_distortion
    
    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.input_data, self.labels))
            random.shuffle(combined)
            self.input_data, self.labels = zip(*combined)
            self.input_data, self.labels = list(self.input_data), list(self.labels)


# --- Data Path Parsing ---
def get_paths(base_path):
    def extract_labels(paths):
        labels_focal = []
        labels_distortion = []
        for path in paths:
            # Convert path to use forward slashes
            path = path.replace('\\', '/')
            try:
                focal = float((path.split('_f_')[1]).split('_d_')[0])
                distortion = float((path.split('_d_')[1]).split('.jpg')[0])
                labels_focal.append(classes_focal.index(focal))
                labels_distortion.append(classes_distortion.index(distortion))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse labels from path {path}. Error: {e}")
                continue
        return labels_focal, labels_distortion
    
    print(base_path)
    
    # Use forward slashes for path construction
    train_path = os.path.join(base_path, 'train')
    valid_path = os.path.join(base_path, 'valid')

    print(train_path)
    print(valid_path)

    train_path = train_path.replace("/","\\")
    valid_path = valid_path.replace("/","\\")

    print(train_path)
    print(valid_path)

    # Check if directories exist
    if not os.path.exists(train_path):
        raise ValueError(f"Training directory not found: {train_path}")
    if not os.path.exists(valid_path):
        raise ValueError(f"Validation directory not found: {valid_path}")
    
    # Get all jpg files from directories
    paths_train = sorted(glob.glob(os.path.join(train_path, "*.jpg")))
    paths_valid = sorted(glob.glob(os.path.join(valid_path, "*.jpg")))
    
    # Check if we found any images
    if not paths_train:
        raise ValueError(f"No training images found in {train_path}")
    if not paths_valid:
        raise ValueError(f"No validation images found in {valid_path}")
    
    print(f"Found {len(paths_train)} training images")
    print(f"Found {len(paths_valid)} validation images")
    print("----------------------end_paths----------------------\n")

    # Convert all paths to use forward slashes
    paths_train = [p.replace('\\', '/') for p in paths_train]
    paths_valid = [p.replace('\\', '/') for p in paths_valid]

    labels_focal_train, labels_distortion_train = extract_labels(paths_train)
    labels_focal_valid, labels_distortion_valid = extract_labels(paths_valid)

    # Verify we have labels for all images
    if len(labels_focal_train) != len(paths_train) or len(labels_distortion_train) != len(paths_train):
        raise ValueError("Mismatch between number of training images and labels")
    if len(labels_focal_valid) != len(paths_valid) or len(labels_distortion_valid) != len(paths_valid):
        raise ValueError("Mismatch between number of validation images and labels")

    input_train = list(zip(paths_train, labels_focal_train))
    input_valid = list(zip(paths_valid, labels_focal_valid))

    # Shuffle
    train_combined = list(zip(input_train, labels_distortion_train))
    random.shuffle(train_combined)
    input_train, labels_distortion_train = zip(*train_combined)

    val_combined = list(zip(input_valid, labels_distortion_valid))
    random.shuffle(val_combined)
    input_valid, labels_distortion_valid = zip(*val_combined)

    return list(input_train), list(labels_distortion_train), list(input_valid), list(labels_distortion_valid)


input_train, labels_train, input_valid, labels_valid = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(f"{len(input_train)} training samples")
print(f"{len(input_valid)} validation samples")

# --- Model Definition ---
image_input = Input(shape=image_shape, name='main_input')
focal_input = Input(shape=(len(classes_focal),), name='concat_input')

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=image_input)
features = Flatten(name='phi-flattened')(base_model.output)
concat = Concatenate(axis=-1)([features, focal_input])
final_output = Dense(len(classes_distortion), activation='softmax', name='output_distortion')(concat)

model = Model(inputs={'main_input': image_input, 'concat_input': focal_input}, outputs=final_output)

# Updated optimizer creation for compatibility with newer TensorFlow
optimizer = Adam(learning_rate=1e-5)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Save Model Structure ---
with open(os.path.join(output_folder, "model.json"), "w") as f:
    f.write(model.to_json())

# Optional: copy script to output
try:
    copyfile(__file__, os.path.join(output_folder, os.path.basename(__file__)))
except:
    pass  # In case __file__ is undefined (e.g., interactive mode)

# --- Callbacks ---
tensorboard = TensorBoard(log_dir=output_log)
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_weight, "weights_{epoch:02d}_{val_loss:.2f}.weights.h5"),
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True
)

# --- Generators ---
train_gen = RotNetDataGenerator(
    input_train, labels_train,
    num_focal=len(classes_focal),
    num_distortion=len(classes_distortion),
    batch_size=batch_size, shuffle=True
)

val_gen = RotNetDataGenerator(
    input_valid, labels_valid,
    num_focal=len(classes_focal),
    num_distortion=len(classes_distortion),
    batch_size=batch_size, shuffle=False
)

# --- Training ---
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=nb_epoch,
    callbacks=[tensorboard, checkpointer]
)