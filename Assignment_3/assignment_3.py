import argparse
import datetime
import json
import math
import os
import numpy as np
import tensorflow as tf
from keras import layers, metrics, callbacks, models, optimizers
from keras.applications import ResNet50
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

def prepare_dataset(dataset_root_dir):
    """ Load and prepare dataset from directory structure """
    class_names = ['0_real_images', '1_fake_images']
    all_image_paths = []
    all_labels = []

    # Scan dirs and prepare dataset with images and labels
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_root_dir, class_name)
        for img_name in os.listdir(class_path):
            all_image_paths.append(os.path.join(class_path, img_name))
            all_labels.append(label_idx) # 0 per real, 1 per fake

    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)

    print(f"[INFO] N. images: {len(all_image_paths)}")
    print(f"[INFO] Original class distribution: Real ({np.sum(all_labels == 0)}), Fake ({np.sum(all_labels == 1)})")
    return all_image_paths, all_labels

def split_dataset(dataset, p_train, p_test, p_val):
    """ Function that splits dataset based on train, validation, test percentages """
    assert p_train + p_test + p_val == 1, "Total percentage must be 1"

    all_image_paths, all_labels = dataset

    # First split: train_val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=p_test,
        stratify=all_labels,
        random_state=42
    )

    # Second split: train and validation
    # test_size: p_val / (1 - p_test)
    val_split_from_train_val = p_val / (1 - p_test)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_split_from_train_val,
        stratify=train_val_labels,
        random_state=42
    )

    print("\n[RESULT] Size of sets after stratified split:")
    print(f"Training:   {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    print(f"Test:       {len(test_paths)} images")

    print("\n[RESULT] Count per class in each dataset:")
    print(f"Training:   Real ({np.sum(train_labels == 0)}),\tFake ({np.sum(train_labels == 1)})")
    print(f"Validation: Real ({np.sum(val_labels == 0)}),\tFake ({np.sum(val_labels == 1)})")
    print(f"Test:       Real ({np.sum(test_labels == 0)}),\tFake ({np.sum(test_labels == 1)})")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def process_image(img_path, label, image_width=224, image_height=224):
    """ Process individual image """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [image_height, image_width])
    img = tf.cast(img, tf.float32)
    # Normalizzation 1./255 will be done by Rescaling Layer in model
    return img, label

def create_generators(batch_size, train_tuples, val_tuples, test_tuples):
    """ Create tf.data.Dataset with pre-processed images based on prepared dataset """
    train_ds = tf.data.Dataset.from_tensor_slices(train_tuples)
    val_ds = tf.data.Dataset.from_tensor_slices(val_tuples)
    test_ds = tf.data.Dataset.from_tensor_slices(test_tuples)

    # Applicaply mapping and configure batch
    AUTOTUNE = tf.data.AUTOTUNE

    shuffle_buffer_size = len(train_tuples[0])
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True) \
                        .map(process_image, num_parallel_calls=AUTOTUNE) \
                        .batch(batch_size) \
                        .prefetch(AUTOTUNE)
    val_ds = val_ds.map(process_image, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.map(process_image, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    print("\nSize of new tf.data.Dataset:")
    print(f"Training batches:   {len(train_ds)}")
    print(f"Validation batches: {len(val_ds)}")
    print(f"Test batches:       {len(test_ds)}")

    return train_ds, val_ds, test_ds

def compute_class_weights(train_ds):
    """ Compute class weights dinamically based on supports"""
    # Collect all labels from the training dataset
    print("[INFO] Collecting labels for class weight calculation...")
    all_train_labels = []
    # tf.data.Dataset.unbatch() returns a dataset of individual elements
    for _, labels in train_ds.unbatch():
        all_train_labels.append(labels.numpy())

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_train_labels),
        y=all_train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[RESULT] Calculated class weights: {class_weight_dict}")
    return class_weight_dict

def self_attention_block(inputs, channels_out):
    """ Create block CNN + Self-Attention + Skip Connection """
    
    # Spatial dimentions 
    H, W = inputs.shape[1], inputs.shape[2]
    C = inputs.shape[-1]
    C_ = channels_out

    # Proietions of Q, K, V with conv 1x1
    q = layers.Conv2D(C_, kernel_size=1, padding='same')(inputs)
    k = layers.Conv2D(C_, kernel_size=1, padding='same')(inputs)
    v = layers.Conv2D(C_, kernel_size=1, padding='same')(inputs)

    # Reshape for attention: (B, H*W, C')
    q_reshaped = layers.Reshape((H * W, C_))(q)
    k_reshaped = layers.Reshape((H * W, C_))(k)
    v_reshaped = layers.Reshape((H * W, C_))(v)
    
    # Calculate attention scores: Q @ K^T
    attention_scores = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([q_reshaped, k_reshaped])
    # Scale attention scores to prevent vanishing/exploding gradients
    scaling_factor = math.sqrt(C_) 
    attention_scores = layers.Lambda(lambda x: x / scaling_factor, name='scaled_attention_scores')(attention_scores)
    # Attention weights: softmax(Q @ K^T)
    attention_weights = layers.Softmax(axis=-1)(attention_scores)

    # Output: A @ V
    attention_output = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attention_weights, v_reshaped])

    # Reshape back to (H, W, C')
    attention_output = layers.Reshape((H, W, C_))(attention_output)

    # Proietion to C canals and skip connection
    attention_output = layers.Conv2D(C, kernel_size=1, padding='same')(attention_output)
    output = layers.Add()([inputs, attention_output])  # skip connection

    return output

def build_model(input_shape, data_augmentation_layers, attention, pre_trained_resnet):
    """ Build keras model based on data augmentation, attention and pre-trained params """
    inputs = tf.keras.Input(shape=input_shape)

    # Rescaling layer
    x = layers.Rescaling(1./255)(inputs)
    
    # Data Augmentation Layers
    x = data_augmentation_layers(x)

    resnet_backbone = None

    if pre_trained_resnet:
        print("[INFO] Building model with Pre-trained ResNet50 backbone")
        # Load ResNet50 backbone withoud head
        resnet_backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze the backbone layers to prevent damaging pre-trained model
        resnet_backbone.trainable = False
        
        x = resnet_backbone(x) # Pass the output of augmentation to the ResNet backbone

        if attention:
            print("[INFO] Adding Self-Attention block after ResNet backbone.")
            x = self_attention_block(x, channels_out=64)

    else:
        print("[INFO] Building Custom CNN model.")
        # Conv Block 1
        x = layers.Conv2D(32, (3, 3), activation="relu", padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        # Conv Block 2
        x = layers.Conv2D(64, (3, 3), activation="relu", padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        # Conv Block 3
        x = layers.Conv2D(128, (3, 3), activation="relu", padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

        if attention:
            x = self_attention_block(x, channels_out=64)

    # Global pooling and classification
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, resnet_backbone

def train_model(model: models.Model, resnet_backbone, train_ds, val_ds, epochs, class_weight_dict, best_model_name, logs_path):
    """ Function that train model with 2 phases if a ResNet is being used or 1 phase otherwise """
    custom_metrics = [
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
    ]

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy", metrics=custom_metrics)

    # Callbacks
    log_dir = os.path.join(logs_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=8, verbose=1, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint(best_model_name, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weight_dict
    )

    # If using resnet backbone, fine tuning unfreezing backbone
    if resnet_backbone is not None:
        print("[DEBUG] Fine-tuning (unfreezing resnet backbone)")
        resnet_backbone.trainable = True

        # Lower learning rate for fine tuning
        model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
                      loss="binary_crossentropy", metrics=custom_metrics)

        log_dir_phase2 = os.path.join(logs_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_phase2")
        callbacks_phase2 = [
            callbacks.TensorBoard(log_dir=log_dir_phase2, histogram_freq=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1, restore_best_weights=True), # Reset EarlyStopping for Phase 2
            callbacks.ModelCheckpoint(best_model_name, save_best_only=True, save_weights_only=False), # Save final fine-tuned model
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
        ]

        # Continue training from where Phase 1 left off
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs, # Total epochs
            initial_epoch=history_phase1.epoch[-1] + 1, # Start from the next epoch of Phase 1 end
            steps_per_epoch=len(train_ds),
            validation_steps=len(val_ds),
            callbacks=callbacks_phase2,
            class_weight=class_weight_dict
        )

def evaluate_model(test_ds, best_model_name):
    """ Load and evaluate :best_model_name with keras metrics and sklearn metrics """
    best_model = tf.keras.models.load_model(best_model_name)
    test_results_best = best_model.evaluate(test_ds, verbose=1)
    print(f"- Loss:      {test_results_best[0]:.4f}")
    print(f"- Accuracy:  {test_results_best[1]:.4f}")
    print(f"- Precision: {test_results_best[2]:.4f}")
    print(f"- Recall:    {test_results_best[3]:.4f}")

    # Generate confusion matrix and classification report
    y_true_best = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_proba_best = best_model.predict(test_ds)
    y_pred_best = (y_pred_proba_best > 0.5).astype(int)

    class_names = ['0_real_images', '1_fake_images']

    print("\n[RESULT] Confusion Matrix (best model):")
    print(confusion_matrix(y_true_best, y_pred_best))

    print("\n[RESULT] Classification Report (best model):")
    print(classification_report(y_true_best, y_pred_best, target_names=class_names))

def compute_pipeline(cfg):
    """Main function that:
    - prepare dataset
    - train cnn 
    - save best configuration
    - evaluate best model
    """
    dataset_root_dir = cfg["dataset_root_dir"]
    logs_path = cfg["logs_path"]
    image_height = cfg["image_height"]
    image_width = cfg["image_width"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    splits = cfg["splits"]
    attention = cfg["attention"]
    best_model_name = cfg["best_model_name"]
    pre_trained_resnet = cfg["pre_trained_resnet"]
    
    dataset = prepare_dataset(dataset_root_dir)
    train_tuples, val_tuples, test_tuples = split_dataset(dataset, **splits)
    train_ds, val_ds, test_ds = create_generators(batch_size, train_tuples, val_tuples, test_tuples)

    # Data Augmentation Layers
    data_augmentation_layers = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"), # Flip horizzontal
            layers.RandomRotation(0.1),      # Rotate to +/- 10% of 360°
            layers.RandomZoom(0.1),          # Zoom of 10%
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Traslation
            layers.RandomContrast(0.1),      # Contrast random of 10%
            layers.RandomBrightness(factor=0.1), # Brightness random of 10%
        ],
        name="data_augmentation",
    )
    class_weight_dict = compute_class_weights(train_ds)
    model, resnet_backbone = build_model((image_height, image_width, 3), data_augmentation_layers, attention, pre_trained_resnet)
    train_model(model, resnet_backbone, train_ds, val_ds, epochs, class_weight_dict, best_model_name, logs_path)
    evaluate_model(test_ds, best_model_name)

def do_inference(image_path, cfg):
    """ Inference function that uses best model to predict class of :image_path """

    model: models.Model = models.load_model(cfg["best_model_name"])
    model.summary()
    # Preprocess image to match model input (size, channels, normalization)
    image, _ = process_image(image_path, None, cfg["image_width"], cfg["image_height"])
    # Adding batch dimention
    image = tf.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_predicted = 1 if prediction[0][0] > 0.5 else 0
    class_names = ['0_real_images', '1_fake_images']
    print(f"[RESULT] Predicted class: {class_names[class_predicted]} (confidence: {prediction[0][0]:.4f})")
    
    return class_predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary classification with deep cnn")
    parser.add_argument("--config", type=str, default="default_config.json", help="Path to JSON config file")
    parser.add_argument("--path", type=str, help="Path to image file to classify")
    parser.add_argument("--train", action="store_true", help="Force model training")
    args = parser.parse_args()

    print("Started with following parameters:")
    print(f"\t - Config: {args.config}")
    print(f"\t - Path: {args.path}")

    print(f"[INFO] JSON config file: {args.config}")
    # Load custom configurations
    with open(args.config) as f:
        cfg = json.load(f)

    if args.train or not os.path.isfile(cfg["best_model_name"]):
        print("[INFO] Training neural network")
        compute_pipeline(cfg)
    else:
        print(f"[INFO] found model: {cfg['best_model_name']}")

    if args.path and os.path.isfile(args.path):
        print(f"[INFO] Inference on file: {args.path}")
        do_inference(args.path, cfg)
    else:
        print("[DEBUG] No image path found to classify")
