import mlflow
import pandas as pd

def find_high_accuracy():
    experiment_name = "Market_Prediction_V2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("No experiment found.")
        return
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("No runs found.")
        return
    
    if 'metrics.directional_accuracy' in runs.columns:
        high_runs = runs[runs['metrics.directional_accuracy'] > 0.7]
        if not high_runs.empty:
            print("Found high accuracy runs:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(high_runs[['params.model_type', 'metrics.directional_accuracy', 'start_time']])
        else:
            print("No runs found with > 70% accuracy.")
    else:
        print("No 'directional_accuracy' metric found in any runs.")

if __name__ == "__main__":
    find_high_accuracy()
