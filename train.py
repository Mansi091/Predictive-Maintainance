import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, roc_auc_score

np.random.seed(42)
tf.random.set_seed(42)

def train_model():
    dataset_path = 'rui-dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    target_col = 'Machine failure'
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_path = 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = num_neg / num_pos
    class_weights = {0: 1.0, 1: pos_weight}
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(feature_cols),)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model_path = 'model.keras'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=60,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    
    best_model = tf.keras.models.load_model(model_path)
    val_probs = best_model.predict(X_val_scaled, verbose=0).flatten()
    val_preds = (val_probs >= 0.5).astype(int)
    
    final_f1 = f1_score(y_val, val_preds, zero_division=0)
    final_recall = recall_score(y_val, val_preds, zero_division=0)
    final_precision = precision_score(y_val, val_preds, zero_division=0)
    final_auc = roc_auc_score(y_val, val_probs)
    
    print(" Final Validation Metrics ")
    print(classification_report(y_val, val_preds, target_names=['Normal', 'Failure']))
    print(f"ROC-AUC Score: {final_auc:.4f}")
    print(f"F1-Score: {final_f1:.4f}")
    print(f"Recall : {final_recall:.4f}")
    print(f"Precision: {final_precision:.4f}")

if __name__ == '__main__':
    train_model()
