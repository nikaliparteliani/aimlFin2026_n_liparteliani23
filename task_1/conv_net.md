# Convolutional Neural Networks (CNN)

## 1. Introduction

A **Convolutional Neural Network (CNN)** is a class of deep neural networks specifically designed for processing structured grid-like data, most notably images. Unlike traditional fully connected neural networks, CNNs leverage the spatial structure of input data by using learnable filters (kernels) that slide across the input, detecting local patterns such as edges, textures, and shapes. This makes CNNs significantly more efficient and effective for tasks involving visual recognition, natural language processing, and even network traffic analysis in cybersecurity.

CNNs were inspired by the biological visual cortex, where individual neurons respond to stimuli only in a restricted region of the visual field known as the **receptive field**. This architecture allows CNNs to build hierarchical representations — lower layers detect simple features (edges, corners), while deeper layers combine these into complex patterns (faces, objects, network attack signatures).

---

## 2. Architecture Overview

A typical CNN consists of the following layers:

```
Input → [Convolution → Activation → Pooling] × N → Flatten → Fully Connected → Output
```

### 2.1 Convolutional Layer

The core building block. A set of learnable filters convolve across the input, producing **feature maps**. Each filter detects a specific feature.

```
Filter (3×3):          Input Patch:         Output (dot product):
┌───┬───┬───┐         ┌───┬───┬───┐
│ 1 │ 0 │-1 │         │ 5 │ 3 │ 2 │
├───┼───┼───┤    *    ├───┼───┼───┤    =  (1×5+0×3+(-1)×2+1×1+0×7+(-1)×3+1×2+0×4+(-1)×1) = 2
│ 1 │ 0 │-1 │         │ 1 │ 7 │ 3 │
├───┼───┼───┤         ├───┼───┼───┤
│ 1 │ 0 │-1 │         │ 2 │ 4 │ 1 │
└───┴───┴───┘         └───┴───┴───┘
```

**Key parameters:**
- **Filters (Kernels):** Small matrices (e.g., 3×3, 5×5) that detect features
- **Stride:** Step size of the filter as it moves across the input
- **Padding:** Adding zeros around the input border to control output dimensions

### 2.2 Activation Function (ReLU)

After convolution, a non-linear activation function is applied element-wise. The most common is **ReLU (Rectified Linear Unit)**:

```
f(x) = max(0, x)

Input Feature Map:        After ReLU:
┌────┬────┬────┐         ┌────┬────┬────┐
│ -1 │  3 │  0 │         │  0 │  3 │  0 │
├────┼────┼────┤   →     ├────┼────┼────┤
│  5 │ -2 │  7 │         │  5 │  0 │  7 │
├────┼────┼────┤         ├────┼────┼────┤
│ -3 │  1 │ -4 │         │  0 │  1 │  0 │
└────┴────┴────┘         └────┴────┴────┘
```

### 2.3 Pooling Layer

Pooling reduces the spatial dimensions of the feature maps, decreasing computational load and providing translational invariance.

**Max Pooling (2×2, stride 2):**

```
Input (4×4):                    Output (2×2):
┌────┬────┬────┬────┐          ┌────┬────┐
│  1 │  3 │  2 │  1 │          │  5 │  7 │
├────┼────┼────┼────┤    →     ├────┼────┤
│  5 │  2 │  7 │  3 │          │  4 │  6 │
├────┼────┼────┼────┤          └────┴────┘
│  4 │  1 │  6 │  2 │
├────┼────┼────┼────┤
│  0 │  3 │  5 │  1 │
└────┴────┴────┴────┘
```

### 2.4 Fully Connected Layer

After several convolutional and pooling layers, the feature maps are **flattened** into a 1D vector and fed into fully connected (dense) layers for classification or regression.

### 2.5 Complete CNN Architecture Visualization

```
                    CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE
                    ==========================================

  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌────────┐
  │  INPUT  │    │  CONV +  │    │  MAX    │    │  CONV +  │    │  MAX    │    │FLATTEN │
  │ 28×28×1 │ →  │  ReLU    │ →  │ POOL   │ →  │  ReLU    │ →  │ POOL   │ →  │        │
  │         │    │ 26×26×32 │    │13×13×32│    │ 11×11×64 │    │ 5×5×64 │    │ [1600] │
  └─────────┘    └──────────┘    └─────────┘    └──────────┘    └─────────┘    └────┬───┘
                                                                                     │
                                                                                     ▼
                                                                              ┌──────────┐
                                                                              │  DENSE   │
                                                                              │  128     │
                                                                              │  (ReLU)  │
                                                                              └────┬─────┘
                                                                                   │
                                                                                   ▼
                                                                              ┌──────────┐
                                                                              │  OUTPUT  │
                                                                              │ Softmax  │
                                                                              │(Classes) │
                                                                              └──────────┘
```

---

## 3. How CNNs Learn

CNNs learn through **backpropagation** and **gradient descent**. During training:

1. **Forward Pass:** Input data passes through all layers, producing a prediction
2. **Loss Calculation:** The prediction is compared with the true label using a loss function (e.g., Cross-Entropy)
3. **Backward Pass:** Gradients of the loss are computed with respect to each filter weight
4. **Weight Update:** Filters are updated using an optimizer (e.g., Adam, SGD)

```
Training Loop Visualization:
═══════════════════════════════════════════════════════════════

  Input ──→ Forward Pass ──→ Prediction ──→ Loss Function
                                                │
                                                ▼
  Update Weights ◄── Compute Gradients ◄── Backpropagation
       │
       └──────────── Repeat until convergence ────────────────→

═══════════════════════════════════════════════════════════════
```

---

## 4. Key Advantages of CNNs

- **Parameter Sharing:** The same filter is applied across the entire input, drastically reducing the number of parameters compared to fully connected networks
- **Local Connectivity:** Each neuron connects only to a local region of the input, capturing spatial locality
- **Translation Invariance:** Pooling layers make the network robust to small shifts and distortions in input
- **Hierarchical Feature Learning:** Automatically learns features from low-level (edges) to high-level (objects)

```
Feature Hierarchy in CNNs:
══════════════════════════════════════════════════════

Layer 1 (Low-level):     Layer 2 (Mid-level):     Layer 3 (High-level):
┌─────────────────┐     ┌─────────────────┐      ┌─────────────────┐
│  ─  │  \  │  /  │     │ Corners, Curves │      │  Faces, Cars,   │
│  |  │  ── │  ·  │     │ Arcs, Circles   │      │  Houses, Animals│
│ Edges & Lines   │     │ Shapes & Textures│      │ Objects & Scenes│
└─────────────────┘     └─────────────────┘      └─────────────────┘

══════════════════════════════════════════════════════
```

---

## 5. Practical Example: CNN for Network Intrusion Detection (Cybersecurity)

### 5.1 Problem Statement

In cybersecurity, detecting malicious network traffic is critical. We can use a **1D CNN** to classify network traffic flows as either **normal** or **malicious** (e.g., DDoS attack, port scan, brute force). The CNN processes numerical features extracted from network packets (packet size, duration, protocol flags, byte counts, etc.) and learns patterns that distinguish attacks from legitimate traffic.

### 5.2 Dataset Description

We simulate a dataset inspired by the **CIC-IDS** and **NSL-KDD** intrusion detection datasets. Each sample represents a network connection with the following features:

| Feature Index | Feature Name     | Description                           |
|--------------|------------------|---------------------------------------|
| 0            | duration         | Connection duration (seconds)         |
| 1            | protocol_type    | Protocol (0=TCP, 1=UDP, 2=ICMP)     |
| 2            | src_bytes        | Bytes sent from source                |
| 3            | dst_bytes        | Bytes sent from destination           |
| 4            | flag             | Connection flag (encoded)             |
| 5            | count            | Connections to same host in 2s        |
| 6            | srv_count        | Connections to same service in 2s     |
| 7            | serror_rate      | SYN error rate                        |
| 8            | same_srv_rate    | Same service connection rate          |
| 9            | dst_host_count   | Destination host connection count     |

**Labels:** `0` = Normal Traffic, `1` = Malicious Traffic (Attack)

### 5.3 Python Implementation

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================
# 1. GENERATE SYNTHETIC NETWORK TRAFFIC DATA
# ============================================================
# Simulating data inspired by NSL-KDD / CIC-IDS datasets
np.random.seed(42)

n_samples = 5000
n_features = 10

# --- Normal traffic patterns ---
n_normal = 3000
normal_data = np.column_stack([
    np.random.exponential(scale=50, size=n_normal),        # duration
    np.random.choice([0, 1], size=n_normal, p=[0.8, 0.2]), # protocol (mostly TCP)
    np.random.normal(loc=500, scale=200, size=n_normal),    # src_bytes
    np.random.normal(loc=1000, scale=300, size=n_normal),   # dst_bytes
    np.random.randint(0, 5, size=n_normal),                 # flag
    np.random.poisson(lam=5, size=n_normal),                # count
    np.random.poisson(lam=4, size=n_normal),                # srv_count
    np.random.uniform(0, 0.1, size=n_normal),               # serror_rate (low)
    np.random.uniform(0.7, 1.0, size=n_normal),             # same_srv_rate (high)
    np.random.poisson(lam=20, size=n_normal),               # dst_host_count
])
normal_labels = np.zeros(n_normal)

# --- Malicious traffic patterns (DDoS, Port Scan, Brute Force) ---
n_attack = 2000
attack_data = np.column_stack([
    np.random.exponential(scale=2, size=n_attack),          # duration (short)
    np.random.choice([0, 1, 2], size=n_attack, p=[0.4, 0.2, 0.4]),  # more ICMP
    np.random.normal(loc=100, scale=50, size=n_attack),     # src_bytes (low)
    np.random.normal(loc=50, scale=30, size=n_attack),      # dst_bytes (low)
    np.random.randint(0, 5, size=n_attack),                 # flag
    np.random.poisson(lam=100, size=n_attack),              # count (very high!)
    np.random.poisson(lam=80, size=n_attack),               # srv_count (high)
    np.random.uniform(0.5, 1.0, size=n_attack),             # serror_rate (high!)
    np.random.uniform(0.0, 0.3, size=n_attack),             # same_srv_rate (low)
    np.random.poisson(lam=200, size=n_attack),              # dst_host_count (high)
])
attack_labels = np.ones(n_attack)

# Combine dataset
X = np.vstack([normal_data, attack_data])
y = np.concatenate([normal_labels, attack_labels])

# Clip negative values (bytes cannot be negative)
X = np.clip(X, 0, None)

print(f"Dataset shape: {X.shape}")
print(f"Normal samples: {n_normal}, Attack samples: {n_attack}")
print(f"Feature names: duration, protocol_type, src_bytes, dst_bytes, flag,")
print(f"              count, srv_count, serror_rate, same_srv_rate, dst_host_count")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for 1D CNN: (samples, timesteps, features)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# ============================================================
# 3. BUILD THE 1D CNN MODEL
# ============================================================
model = Sequential([
    # First Convolutional Block
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
           input_shape=(n_features, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # Second Convolutional Block
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 4. TRAIN THE MODEL
# ============================================================
history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ============================================================
# 5. EVALUATE THE MODEL
# ============================================================
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss:     {test_loss:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,
                          target_names=['Normal', 'Attack']))

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# ============================================================
# 6. VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training & Validation Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Training & Validation Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix', fontsize=14)
plt.colorbar(im)
classes = ['Normal', 'Attack']
tick_marks = [0, 1]
ax.set_xticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes)

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)

ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualizations saved: training_history.png, confusion_matrix.png")
```

### 5.4 Expected Output

After running the code, the model typically achieves **97-99% accuracy** on the test set, demonstrating that CNNs can effectively learn to distinguish between normal and malicious network traffic patterns.

```
Expected Results:
══════════════════════════════════════════
  Test Accuracy:  ~0.98+

  Classification Report:
  ┌──────────┬───────────┬────────┬──────────┐
  │  Class   │ Precision │ Recall │ F1-Score │
  ├──────────┼───────────┼────────┼──────────┤
  │  Normal  │   0.99    │  0.99  │   0.99   │
  │  Attack  │   0.98    │  0.98  │   0.98   │
  └──────────┴───────────┴────────┴──────────┘
══════════════════════════════════════════
```

### 5.5 Why CNNs Work for Intrusion Detection

```
Traditional Approach:               CNN-Based Approach:
┌────────────────────┐              ┌────────────────────┐
│  Manual Feature    │              │  Raw Network       │
│  Engineering       │              │  Features          │
│  (Expert needed)   │              │  (Automated)       │
└────────┬───────────┘              └────────┬───────────┘
         ▼                                   ▼
┌────────────────────┐              ┌────────────────────┐
│  Rule-based /      │              │  CNN Learns        │
│  Signature Match   │              │  Patterns Auto.    │
└────────┬───────────┘              └────────┬───────────┘
         ▼                                   ▼
┌────────────────────┐              ┌────────────────────┐
│  ✗ Misses new      │              │  ✓ Detects new     │
│    attack variants │              │    (zero-day)      │
│  ✗ Static rules    │              │    attack patterns │
│  ✗ High false +    │              │  ✓ Adaptive        │
└────────────────────┘              └────────────────────┘
```

The **1D CNN** treats the sequence of network features as a one-dimensional signal and applies convolutional filters to detect local patterns and correlations between adjacent features. This is particularly effective because attack traffic often exhibits correlated anomalies across multiple features simultaneously (e.g., high connection count + high error rate + low duration = DDoS attack).

---

## 6. Summary

Convolutional Neural Networks are powerful deep learning architectures that automatically learn hierarchical feature representations from structured data. While originally designed for image processing, their ability to detect patterns makes them invaluable in cybersecurity for tasks like intrusion detection, malware classification, and anomaly detection. The combination of convolutional layers, pooling, and non-linear activations allows CNNs to capture complex relationships in data that traditional methods often miss.

---

*References: LeCun et al. (1998) "Gradient-Based Learning Applied to Document Recognition"; CIC-IDS2017 Dataset; NSL-KDD Dataset*
