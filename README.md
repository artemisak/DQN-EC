# Graph-Enhanced-Communication

## Graph Autoencoder

This project includes an implementation of a **Graph AutoEncoder** for encoding and reconstructing agent observations from the `simple_speaker_listener_v4` environment (from [PettingZoo](https://www.pettingzoo.ml/)). It uses a ** kNN graph with Gabriel pruning** and a bunch of other conventional methods building latent graph representation.

### Usage

Train the models:

```bash
python Metrics.py
```

Visualize the results:

```bash
python visualization.py
```