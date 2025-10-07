import wandb
import pandas as pd

api = wandb.Api()

# Get all runs from a project
runs = api.runs("itmo-university-it/ma_dqn_hypergraph_gat")

all_data = []
for run in runs:
    history = run.history()
    history['run_name'] = run.name
    all_data.append(history)

# Combine all runs
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv("rl_runs.csv", index=False)