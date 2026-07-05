import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "rui-dataset.csv"
OUTPUT_CSV_PATH = "rui-dataset-engineered.csv"
PLOTS_DIR = "static/eda"

def run_eda_and_feature_engineering():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    
    df['Power_Nm_RPM'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['Temp_Difference_K'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 8))
    numeric_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Power_Nm_RPM',
        'Temp_Difference_K',
        'Machine failure'
    ]
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap with Engineered Features", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='Temp_Difference_K', hue='Machine failure', fill=True, common_norm=False, palette="Set1", alpha=0.5)
    plt.title("Temperature Difference Distribution by Failure Status", fontsize=12)
    plt.xlabel("Process Temp - Air Temp [K]")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "temp_diff_distribution.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='Power_Nm_RPM', hue='Machine failure', fill=True, common_norm=False, palette="Set2", alpha=0.5)
    plt.title("Spindle Power Distribution by Failure Status", fontsize=12)
    plt.xlabel("Power (Torque * Speed)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "power_distribution.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    run_eda_and_feature_engineering()
