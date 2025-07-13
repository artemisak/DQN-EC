import os
import torch
from graph_autoencoder import GraphAutoEncoder, generate_sample_data, GRAY_INVERTED_MAPPING, create_graph, param_schema, algorithms

from typing import List

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("model_config", {"input_dim": 5, "output_dim": 3, "hidden_dim": 64})
    model_name = os.path.basename(model_path)
    filename, _ = os.path.splitext(model_name)
    algo_name = filename.split('_')[0]

    model = GraphAutoEncoder(
        input_dim=5,
        output_dim=3,
        hidden_dim=128,
        graph_fn=algorithms[algo_name]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

def evaluate_model(model, batch, model_name):
    with torch.no_grad():
        _, _, logits, values, latent_list, edge_index_list, edge_attr_list = model(batch)

    idx = 0
    obs = batch[idx]
    latent = latent_list[idx]
    edge_index = edge_index_list[idx]
    edge_attr = edge_attr_list[idx]

    variable_order = []
    for row in obs:
        key = tuple(row[:4].int().tolist())
        var_id = GRAY_INVERTED_MAPPING.get(key, None)
        variable_order.append(var_id)

    sorted_schema = [param_schema[i - 1] for i in variable_order]

    G = create_graph(
        latent_points=latent,
        edge_index=edge_index,
        edge_attr=edge_attr,
        parameters={"epoch": 0, "name": model_name},
        param_schema=sorted_schema,
        is_visual=True,
        visual_save_path="results/graphics"
    )

    # Пример: печатаем reconstructed значения
    print(f"\n📊 Модель: {model_name}")
    print("Softmax labels (первые 3 узла):")
    print(torch.softmax(logits[0][:3], dim=-1))
    print("Values (первые 3 узла):")
    print(values[0][:3].squeeze())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = generate_sample_data(num_samples=1, batch_size=1)
    batch = next(iter(loader))[0]
    batch = batch.to(device)
    model_dir = "results/models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

    if not model_files:
        raise FileNotFoundError("❌ Модели не найдены в results/models.")

    for model_file in sorted(model_files):
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path, device)
        evaluate_model(model, batch, model_file)


if __name__ == "__main__":
    main()
