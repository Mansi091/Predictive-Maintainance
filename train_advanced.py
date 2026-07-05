import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

DATA_PATH = "rui-dataset-engineered.csv"
MODEL_EXPORT_PATH = "model.pkl"
PLOTS_DIR = "static/eda"

def train_advanced_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Engineered dataset not found: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Power_Nm_RPM',
        'Temp_Difference_K'
    ]
    target_col = 'Machine failure'
    
    X = df[feature_cols]
    y = df[target_col]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_pr_aucs, xgb_pr_aucs = [], []
    oof_probs = np.zeros(len(df))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_probs = rf.predict_proba(X_val_scaled)[:, 1]
        
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        xgb = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            scale_pos_weight=scale_pos_weight, 
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(X_train_scaled, y_train)
        xgb_probs = xgb.predict_proba(X_val_scaled)[:, 1]
        
        oof_probs[val_idx] = xgb_probs
        
        rf_p, rf_r, _ = precision_recall_curve(y_val, rf_probs)
        xgb_p, xgb_r, _ = precision_recall_curve(y_val, xgb_probs)
        
        rf_pr_aucs.append(auc(rf_r, rf_p))
        xgb_pr_aucs.append(auc(xgb_r, xgb_p))
        
    print(f"RF Mean PR-AUC: {np.mean(rf_pr_aucs):.4f}")
    print(f"XGB Mean PR-AUC: {np.mean(xgb_pr_aucs):.4f}")
    
    precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    tuned_preds = (oof_probs >= best_threshold).astype(int)
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Precision: {precision_score(y, tuned_preds):.4f}")
    print(f"Recall: {recall_score(y, tuned_preds):.4f}")
    print(f"F1-Score: {f1_score(y, tuned_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, oof_probs):.4f}")
    print(f"PR-AUC: {auc(recalls, precisions):.4f}")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f"XGBoost (PR-AUC = {auc(recalls, precisions):.3f})", color='purple', lw=2)
    plt.axvline(x=recall_score(y, tuned_preds), color='red', linestyle='--')
    plt.scatter(recall_score(y, tuned_preds), precision_score(y, tuned_preds), color='red', marker='o', s=100)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150)
    plt.close()
    
    cm = confusion_matrix(y, tuned_preds)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Failure'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scale_pos_weight = (len(y) - sum(y)) / sum(y)
    
    final_xgb = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        scale_pos_weight=scale_pos_weight, 
        random_state=42,
        eval_metric='logloss'
    )
    final_xgb.fit(X_scaled, y)
    
    model_artifacts = {
        'model': final_xgb,
        'scaler': scaler,
        'threshold': float(best_threshold),
        'feature_cols': feature_cols
    }
    with open(MODEL_EXPORT_PATH, 'wb') as f:
        pickle.dump(model_artifacts, f)

if __name__ == "__main__":
    train_advanced_pipeline()
