import mlflow
import pandas as pd

# =========================
# SETTINGS
# =========================

EXPERIMENT_NAME = "Market_Prediction_V3"

# =========================
# DISPLAY RESULTS
# =========================

def display_results():

    print("\nFetching MLflow results...")

    # -------------------------
    # GET EXPERIMENT
    # -------------------------

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"No experiment found with name '{EXPERIMENT_NAME}'")
        return

    # -------------------------
    # LOAD RUNS
    # -------------------------

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        print("No runs found.")
        return

    # Keep only successful runs
    if 'status' in runs.columns:
        runs = runs[runs['status'] == 'FINISHED']

    # -------------------------
    # SAFE COLUMN HANDLING
    # -------------------------

    required_columns = [
        'params.model_type',
        'metrics.directional_accuracy',
        'metrics.f1_score',
        'metrics.loss',
        'metrics.val_loss',
        'start_time'
    ]

    for col in required_columns:
        if col not in runs.columns:
            runs[col] = None

    display_df = runs[required_columns].copy()

    # -------------------------
    # KEEP LATEST RUN PER MODEL
    # -------------------------

    display_df = display_df.sort_values(
        by=['params.model_type', 'start_time'],
        ascending=[True, False]
    )

    display_df = display_df.drop_duplicates(
        subset=['params.model_type'],
        keep='first'
    )

    # -------------------------
    # SORT BY PERFORMANCE
    # -------------------------

    display_df = display_df.sort_values(
        by='metrics.directional_accuracy',
        ascending=False
    )

    # -------------------------
    # RENAME COLUMNS
    # -------------------------

    display_df.columns = [
        'Model',
        'Accuracy',
        'F1-Score',
        'Train Loss',
        'Validation Loss',
        'Last Run'
    ]

    # -------------------------
    # FORMAT OUTPUT SAFELY
    # -------------------------

    display_df['Accuracy'] = display_df['Accuracy'].apply(
        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
    )

    display_df['F1-Score'] = display_df['F1-Score'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )

    display_df['Train Loss'] = display_df['Train Loss'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )

    display_df['Validation Loss'] = display_df['Validation Loss'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------

    print("\n" + "=" * 80)
    print("MLFLOW MARKET PREDICTION RESULTS")
    print("=" * 80)

    print(display_df.to_string(index=False))

    print("=" * 80)

    # -------------------------
    # BEST MODEL
    # -------------------------

    best_model = display_df.iloc[0]

    print("\nBEST MODEL:")
    print(
        f"{best_model['Model']} | "
        f"Accuracy: {best_model['Accuracy']} | "
        f"F1: {best_model['F1-Score']}"
    )

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    display_results()