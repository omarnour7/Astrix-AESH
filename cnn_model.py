"""
GeoRadar — CNN Subsurface Classifier
Team Astrix · AESH 2026 · Green Radar Systems

Classifies preprocessed GPR A-scan envelopes into three classes:
    0 → Dry soil
    1 → Water-bearing layer
    2 → Rock formation

Architecture: 1-D CNN → GlobalAveragePooling → Dense softmax
Input shape:  (n_time_steps, 1)   — single-channel envelope
"""

import os
import json
import numpy as np

# ── Graceful import of TensorFlow ─────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not installed. Using mock model for demo.")

CLASS_NAMES = ["dry_soil", "water_bearing", "rock"]
N_CLASSES = len(CLASS_NAMES)


# ── Model definition ───────────────────────────────────────────────────────────

def build_cnn(input_length: int, n_classes: int = N_CLASSES,
              dropout_rate: float = 0.3) -> "keras.Model":
    """
    1-D Convolutional Neural Network for GPR A-scan classification.

    Architecture:
        Block 1: Conv1D(32, k=7) → BN → ReLU → MaxPool(2)
        Block 2: Conv1D(64, k=5) → BN → ReLU → MaxPool(2)
        Block 3: Conv1D(128, k=3) → BN → ReLU → MaxPool(2)
        Head:    GlobalAvgPool → Dropout → Dense(64, ReLU) → Dense(n_classes, softmax)
    """
    inp = keras.Input(shape=(input_length, 1), name="envelope_input")

    x = layers.Conv1D(32,  7, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64,  5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GeoRadar_CNN")
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    data_dir:   str = "data/synthetic_labels",
    model_path: str = "results/georadar_cnn.keras",
    epochs:     int = 50,
    batch_size: int = 32,
    val_split:  float = 0.2,
    test_split: float = 0.1,
    seed:       int = 42,
) -> dict:
    """
    Load dataset, train CNN, evaluate on held-out test set.
    Returns classification report dict.
    """
    if not TF_AVAILABLE:
        print("[MOCK] TF not available — returning dummy accuracy.")
        return {"accuracy": 0.87, "note": "mock result, install TensorFlow to train"}

    # Load data
    X = np.load(os.path.join(data_dir, "X.npy"))   # (N, T, 2)
    y = np.load(os.path.join(data_dir, "y.npy"))   # (N,)

    # Use envelope channel (already preprocessed) — channel 0 = real
    # For CNN input we use magnitude: sqrt(real^2 + imag^2)
    X_env = np.sqrt(X[:, :, 0] ** 2 + X[:, :, 1] ** 2)
    X_env = X_env / (X_env.max(axis=1, keepdims=True) + 1e-8)
    X_env = X_env[..., np.newaxis]   # (N, T, 1)

    # Train / val / test split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_split)
    n_val  = int(len(y) * val_split)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    X_tr, y_tr = X_env[train_idx], y[train_idx]
    X_v,  y_v  = X_env[val_idx],   y[val_idx]
    X_te, y_te = X_env[test_idx],  y[test_idx]

    print(f"Train: {len(X_tr)}  Val: {len(X_v)}  Test: {len(X_te)}")

    model = build_cnn(input_length=X_env.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
    ]

    history = model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_v, y_v),
        callbacks=cb_list,
        verbose=1,
    )

    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = np.argmax(model.predict(X_te), axis=1)
    report_str = classification_report(y_te, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True)
    print("\n── Test Set Classification Report ──")
    print(report_str)

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    with open(model_path.replace(".keras", "_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    with open("results/cnn_accuracy_report.txt", "w") as f:
        f.write(report_str)

    print(f"\nModel saved: {model_path}")
    return report_dict


# ── Inference helper ───────────────────────────────────────────────────────────

def predict_scan(envelope: np.ndarray, model_path: str = "results/georadar_cnn.keras"):
    """
    Predict subsurface class for a single preprocessed envelope.

    Args:
        envelope: 1-D float array (normalized A-scan envelope)

    Returns:
        class_name (str), confidence (float)
    """
    if not TF_AVAILABLE:
        return "water_bearing", 0.87   # mock

    model = keras.models.load_model(model_path)
    x = envelope[np.newaxis, :, np.newaxis].astype(np.float32)
    probs = model.predict(x, verbose=0)[0]
    cls = int(np.argmax(probs))
    return CLASS_NAMES[cls], float(probs[cls])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--data_dir",   type=str, default="data/synthetic_labels")
    parser.add_argument("--model_path", type=str, default="results/georadar_cnn.keras")
    args = parser.parse_args()

    if args.train:
        train(data_dir=args.data_dir, model_path=args.model_path, epochs=args.epochs)
    else:
        print("Pass --train to start training. Example:")
        print("  python cnn_model.py --train --epochs 50")
        model = build_cnn(input_length=10000)
        model.summary()
