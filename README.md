# Graph-Enchanced-Communication

## Dependecies

Install the required Python Packages by using **requirements.txt**.
 
## Graph Autoencoder

This project includes an implementation of a **Graph AutoEncoder** for encoding and reconstructing agent observations from the `simple_speaker_listener_v4` environment (from [PettingZoo](https://www.pettingzoo.ml/)). It uses a **Gabriel Graph** to build a latent graph structure and includes advanced visualization of the resulting latent space.

### Usage

Train the model and generate visualizations:
```bash
python graph_autoencoder.py
```

### Outputs

- Model Checkpoint:
  - Saved to graph_autoencoder_YYYYMMDD_HHMMSS.pth
- Graph Visualizations:
  - Saved as graphs/epochXX.png, showing:
    - Node positions in 2D latent space 
    - Gabriel Graph edges 
    - Minimum Spanning Tree (MST)
    - Node labels, colors, and semantic grouping 
    - Tabular node and edge summaries

Example of visualisation:
![epoch.png](./resources/epoch-example.png)