import mlflow

def display_results():
    experiment_name = "Market_Prediction_V2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"No experiment found with name '{experiment_name}'")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Sort by MSE metric
    if 'metrics.mse' in runs.columns:
        runs = runs.sort_values(by='metrics.mse', ascending=True)
    
    print("\n" + "="*50)
    print("MLFLOW EXPERIMENT RESULTS")
    print("="*50)
    
    # Select key columns
    cols = ['params.model_type', 'metrics.mse', 'start_time']
    display_df = runs[cols].copy()
    display_df.columns = ['Model Name', 'MSE (Val)', 'Run Time']
    
    print(display_df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    display_results()
