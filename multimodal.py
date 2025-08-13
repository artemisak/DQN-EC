import random
from datetime import datetime
import os
from dataclasses import dataclass

import numpy as np
import tyro
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from graph_autoencoder import GraphAutoEncoder, algorithms, create_graph,NodeDescriptor,save_loss_log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Config:
    # Training hyperparameters
    epochs: int = 100                        # Total number of training epochs
    lr: float = 0.001                      # Learning rate for optimizer
    model_save_path: str = "results/models" # Path where trained model will be saved

    # Data generation
    num_samples: int = 1024                 # Number of synthetic samples to generate from the environment
    batch_size: int = 64                    # Batch size used in training

    # Learning rate scheduler parameters
    factor: float = 0.5                     # Factor by which the learning rate will be reduced
    patience: int = 5                       # Number of epochs with no improvement after which LR will be reduced

    # Other training parameters
    alpha: float = 0.1                      # Skip connection blending coefficient in GAT
    max_norm: float = 1.0                   # Maximum norm for gradient clipping
    gamma: float = 0.1

    # Metrics Parameters
    is_growth_from_central: bool = False    # Enable calculate growth from central user node

    # Parameters for configure results output
    visualise: bool = True                          # Enable create visualisation of epochs
    visual_save_path: str = "results/graphics"  # Path where visualisation will be saved
    save_metrics: bool = False                       # Whether to save metrics to disk
    data_save_path: str = "results/metrics"          # Path where metrics will be saved

classes = ['red', 'green', 'blue', 'other']
epsilon = 0.1
group_marker_map = {
    "word": "o"
}

class ColorTokenVectorExtractor:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the Token Vector Extractor
        Available small models perfect for Colab:
        - distilgpt2 (82M parameters)
        - gpt2 (124M parameters)
        - distilbert-base-uncased (66M parameters)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load appropriate model type
        if "bert" in model_name.lower():
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(device)
        self.model.eval()  # Set to evaluation mode

        print(f"✅ Loaded model: {model_name}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 Hidden size: {self.model.config.hidden_size}")

    def extract_token_vectors(self, text: str, layer_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        Extract token vectors from the model for a given text
        Args:
            text: Input text to analyze
            layer_idx: Which layer to extract vectors from (-1 = last layer, -2 = second to last, etc.)
        Returns:
            Dictionary mapping tokens to their vector representations
        """
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get hidden states (token vectors)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract the specified layer
        hidden_states = outputs.hidden_states[layer_idx]  # Shape: (batch_size, seq_len, hidden_size)
        token_vectors = hidden_states[0].cpu().numpy()  # Remove batch dimension

        # Map tokens to their vectors
        token_vector_map = {}
        for i, token in enumerate(tokens):
            # Clean up token representation
            clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n')
            if clean_token.startswith(' '):
                clean_token = clean_token[1:]  # Remove leading space for display

            token_vector_map[clean_token] = token_vectors[i]

        return token_vector_map

    def create_manual_rgb_tokens(self, r: float, g: float, b: float) -> Dict[str, np.ndarray]:
        """
        Manually create red255, green64, blue16 style tokens by averaging color name and number vectors
        """
        # Get individual component vectors
        red_text = f"red {int(r)}"
        green_text = f"green {int(g)}"
        blue_text = f"blue {int(b)}"
        base_text = "What color is"

        # Extract vectors for each component
        red_vectors = self.extract_token_vectors(red_text)
        green_vectors = self.extract_token_vectors(green_text)
        blue_vectors = self.extract_token_vectors(blue_text)
        base_vectors = self.extract_token_vectors(base_text)

        # Find the color name and number vectors
        def find_vectors(vectors_dict, color_name, number):
            color_vec = None
            number_vec = None

            for token, vec in vectors_dict.items():
                clean_token = token.strip().lower()
                if clean_token == color_name.lower():
                    color_vec = vec
                elif clean_token == str(number):
                    number_vec = vec

            return color_vec, number_vec

        # Get individual component vectors
        red_color_vec, red_num_vec = find_vectors(red_vectors, 'red', int(r))
        green_color_vec, green_num_vec = find_vectors(green_vectors, 'green', int(g))
        blue_color_vec, blue_num_vec = find_vectors(blue_vectors, 'blue', int(b))

        # Create combined tokens
        combined_tokens = {}

        # Add base question tokens (What, color, is)
        for token, vec in base_vectors.items():
            if token.strip().lower() not in ['red', 'green', 'blue']:
                combined_tokens[token] = vec

        # Average the color name and number vectors to create combined tokens
        if red_color_vec is not None and red_num_vec is not None:
            combined_tokens[f'red{int(r)}'] = (red_color_vec + red_num_vec) / 2

        if green_color_vec is not None and green_num_vec is not None:
            combined_tokens[f'green{int(g)}'] = (green_color_vec + green_num_vec) / 2

        if blue_color_vec is not None and blue_num_vec is not None:
            combined_tokens[f'blue{int(b)}'] = (blue_color_vec + blue_num_vec) / 2

        return combined_tokens

    def analyze_color_tokens_average(self, r: float, g: float, b: float) -> Dict:
        """
        Analyze token vectors for a color question using manual RGB token creation with color names
        """
        # Use manual method to create red255, green64, blue16 style tokens
        token_vectors = self.create_manual_rgb_tokens(r, g, b)
        predicted_color = self.predict_color_from_rgb(r, g, b)

        question = f"What color is red{int(r)} green{int(g)} blue{int(b)}"

        return {
            'question': question,
            'rgb_values': (r, g, b),
            'predicted_color': predicted_color,
            'token_vectors': token_vectors,
            'vector_stats': self.get_vector_statistics(token_vectors)
        }

    def analyze_color_tokens_template(self, r: int, g: int, b: int) -> Dict:
        """
        RECOMMENDED: Use template approach for consistent, learnable patterns
        """
        # Use consistent template that creates predictable token patterns
        question = f"What component is dominating: red={r:.2f}, green={g:.2f}, blue={b:.2f}? One word answer."

        token_vectors = self.extract_token_vectors(question)
        predicted_color = self.predict_color_from_rgb(r, g, b)

        return {
            'question': question,
            'rgb_values': (r, g, b),
            'predicted_color': predicted_color,
            'token_vectors': token_vectors,
            'vector_stats': self.get_vector_statistics(token_vectors)
        }

    def predict_color_from_rgb(self, r: int, g: int, b: int) -> str:

        total = r + g + b
        if total == 0:

            return 'red'  # Default choice

        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total

        threshold = 0.4

        if r_ratio >= threshold and r_ratio >= g_ratio and r_ratio >= b_ratio:
            return 'red'
        elif g_ratio >= threshold and g_ratio >= r_ratio and g_ratio >= b_ratio:
            return 'green'
        elif b_ratio >= threshold and b_ratio >= r_ratio and b_ratio >= g_ratio:
            return 'blue'
        else:

            max_component = max(r, g, b)
            if r == max_component:
                return 'red'
            elif g == max_component:
                return 'green'
            else:
                return 'blue'

    def get_vector_statistics(self, token_vectors: Dict[str, np.ndarray]) -> Dict:
        """Get statistics about the token vectors"""
        if not token_vectors:
            return {}

        all_vectors = np.array(list(token_vectors.values()))

        return {
            'num_tokens': len(token_vectors),
            'vector_dimension': all_vectors.shape[1],
            'mean_magnitude': np.mean(np.linalg.norm(all_vectors, axis=1)),
            'std_magnitude': np.std(np.linalg.norm(all_vectors, axis=1)),
            'mean_values': np.mean(all_vectors, axis=0),
            'std_values': np.std(all_vectors, axis=0)
        }

    def visualize_token_vectors(self, token_vectors: Dict[str, np.ndarray], top_dims: int = 15):
        """Visualize token vectors using a heatmap"""
        if not token_vectors:
            print("No token vectors to visualize")
            return

        tokens = list(token_vectors.keys())
        vectors = np.array(list(token_vectors.values()))

        # Show dimensions with highest variance
        variances = np.var(vectors, axis=0)
        top_indices = np.argsort(variances)[-top_dims:]

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            vectors[:, top_indices].T,
            xticklabels=tokens,
            yticklabels=[f"Dim_{i}" for i in top_indices],
            cmap="RdBu_r",
            center=0,
            cbar_kws={'label': 'Vector Value'}
        )
        plt.title(f"Token Vector Heatmap - Top {top_dims} Variance Dimensions\nModel: {self.model_name}")
        plt.xlabel("Tokens")
        plt.ylabel("Vector Dimensions")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def compare_different_layers(self, text: str, layers: List[int] = [-1, -2, -3]):
        """Compare token vectors from different model layers"""
        print(f"🔍 Comparing vectors across layers for: '{text}'")

        layer_vectors = {}
        for layer in layers:
            try:
                vectors = self.extract_token_vectors(text, layer_idx=layer)
                layer_vectors[f"Layer {layer}"] = vectors
                print(f"✅ Extracted vectors from layer {layer}")
            except IndexError:
                print(f"❌ Layer {layer} not available (model has {len(self.model.config.hidden_size)} layers)")

        return layer_vectors

    def print_detailed_vectors(self, token_vectors: Dict[str, np.ndarray], max_dims: int = 10):
        """Print token vectors in the requested format"""
        print(f"\n🎯 Token/Vector pairs (showing first {max_dims} dimensions):")
        print("-" * 60)

        for token, vector in token_vectors.items():
            # Show first max_dims dimensions
            vector_preview = ", ".join([f"{v:.2f}" for v in vector[:max_dims]])
            vector_end = f"{vector[-1]:.2f}"
            print(f"{token:>10} / [{vector_preview}, ..., {vector_end}]")

# Interactive functions
def quick_color_analysis(extractor, r: int, g: int, b: int):
    """Quick analysis function for testing different colors"""
    result = extractor.analyze_color_tokens(r, g, b)
    print(f"\n🎨 RGB({r}, {g}, {b}) Analysis:")
    print(f"Predicted color: {result['predicted_color']}")
    extractor.print_detailed_vectors(result['token_vectors'], max_dims=5)
    pass

def compare_colors(extractor, colors: List[Tuple[int, int, int]]):
    """Compare token vectors for different colors"""
    results = []

    print(f"\n🔄 Comparing {len(colors)} colors:")
    for i, (r, g, b) in enumerate(colors):
        result = extractor.analyze_color_tokens(r, g, b)
        results.append(result)
        print(f"{i+1}. RGB({r}, {g}, {b}) → {result['predicted_color']}")

    return results

def analyze_specific_tokens(extractor, text: str, target_tokens: List[str]):
    """Focus on specific tokens in the text"""
    token_vectors = extractor.extract_token_vectors(text)

    print(f"\n🎯 Analyzing specific tokens in: '{text}'")
    for target in target_tokens:
        found_tokens = [token for token in token_vectors.keys() if target.lower() in token.lower()]
        if found_tokens:
            for token in found_tokens:
                vector = token_vectors[token]
                preview = ", ".join([f"{v:.2f}" for v in vector[:5]])
                print(f"'{token}' → [{preview}, ..., {vector[-1]:.2f}]")
        else:
            print(f"Token containing '{target}' not found")


def create_param_schema_from_tokens(tokens, default_group="word"):
    return [NodeDescriptor(name=tok, group=default_group) for tok in tokens]

def create_soft_label(class_name, classes, epsilon=0.1):
    n = len(classes)
    smooth_value = epsilon / (n - 1)
    soft_label = torch.full((n,), smooth_value)

    if class_name not in classes:
        raise ValueError(f"Class '{class_name}' not in class list {classes}")

    class_idx = classes.index(class_name)
    soft_label[class_idx] = 1.0 - epsilon
    return soft_label


def multimodal_color_data_generator(n_samples, extractor):
    """
    Generate colors with one dominant channel but varied non-dominant values.
    """
    for _ in range(n_samples):
        # Choose which channel will be dominant
        dominant_channel = random.choice([0, 1, 2])

        # Generate varied ranges for non-dominant channels
        # Instead of always 0-80, use different ranges for variety
        low_range_options = [
            (0, 40),  # Very low
            (20, 60),  # Medium-low
            (0, 80),  # Original range
            (10, 50),  # Low-medium
            (30, 70),  # Medium (for subtle dominance)
        ]

        # Pick ranges for the two non-dominant channels
        range1 = random.choice(low_range_options)
        range2 = random.choice(low_range_options)

        # Generate base values for all channels
        r = random.randint(*range1)
        g = random.randint(*range2)
        b = random.randint(*range1)  # Reuse range1 or could use another range

        # Set the dominant channel to high value (75-100 for clear dominance)
        dominant_value = random.randint(75, 100)

        if dominant_channel == 0:
            r = dominant_value
        elif dominant_channel == 1:
            g = dominant_value
        else:
            b = dominant_value

        result = extractor.analyze_color_tokens_average(r, g, b)
        yield {
            'token_vectors': result['token_vectors'],
            'gpt_color_label': result['predicted_color'],
            'rgb': (r, g, b),
            'tokens': list(result['token_vectors'].keys())
        }


class TokenizedDataset(torch.utils.data.Dataset):
    """Custom dataset that preserves token information with data"""

    def __init__(self, data_tensor, label_tensor, tokens_list, rgb_list):
        self.data = data_tensor
        self.labels = label_tensor
        self.tokens = tokens_list
        self.rgb = rgb_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.tokens[idx], self.rgb[idx]

def generate_color_token_dataloader(
        extractor,
        num_samples=1024,
        batch_size=64,
        classes=None,
        epsilon=0.1,
        device='cpu',
        validation_split=0.2
):
    if classes is None:
        raise ValueError("Argument `classes` must be provided.")

    observations, labels, tokens_all, rgb_values = [], [], [], []

    for data in multimodal_color_data_generator(num_samples, extractor):
        token_vecs = list(data["token_vectors"].values())
        label = data["gpt_color_label"]
        tokens = data["tokens"]
        rgb = data["rgb"]

        observations.append(torch.tensor(token_vecs, dtype=torch.float32))
        labels.append(create_soft_label(label, classes, epsilon))
        tokens_all.append(tokens)
        rgb_values.append(rgb)

    max_len = max(obs.shape[0] for obs in observations)
    vector_dim = observations[0].shape[1]

    padded_observations = []
    for obs in observations:
        pad_len = max_len - obs.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, vector_dim))
            obs = torch.cat([obs, padding], dim=0)
        padded_observations.append(obs)

    data_tensor = torch.stack(padded_observations).to(device)
    label_tensor = torch.stack(labels).to(device)

    # Split into train and validation sets
    total_samples = len(data_tensor)
    val_size = int(total_samples * validation_split)
    train_size = total_samples - val_size

    # Create random indices for splitting
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Split data
    train_data = data_tensor[train_indices]
    train_labels = label_tensor[train_indices]
    train_tokens = [tokens_all[i] for i in train_indices]
    train_rgb = [rgb_values[i] for i in train_indices]

    val_data = data_tensor[val_indices]
    val_labels = label_tensor[val_indices]
    val_tokens = [tokens_all[i] for i in val_indices]
    val_rgb = [rgb_values[i] for i in val_indices]

    # Create custom datasets that preserve token information
    train_dataset = TokenizedDataset(train_data, train_labels, train_tokens, train_rgb)
    val_dataset = TokenizedDataset(val_data, val_labels, val_tokens, val_rgb)

    # Custom collate function to handle the extra token and RGB data
    def custom_collate_fn(batch):
        data_batch = torch.stack([item[0] for item in batch])
        labels_batch = torch.stack([item[1] for item in batch])
        tokens_batch = [item[2] for item in batch]
        rgb_batch = [item[3] for item in batch]
        return data_batch, labels_batch, tokens_batch, rgb_batch

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader


def train_model(
        model,
        train_dataloader,
        val_dataloader,
        name: str,
        epochs: int,
        lr: float,
        factor: float,
        patience: int,
        model_save_path: str = "results/models",
        model_filename: str = "model.pth",
        is_visual: bool = False,
        visual_save_path: str = "results/graphics",
        is_save: bool = False,
        data_save_path: str = "results/metrics"
):
    os.makedirs(model_save_path, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience
    )

    print(f"🎯 Starting training for {epochs} epochs...")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Optimizer: Adam (lr={lr})")
    print(f"   Training samples: {len(train_dataloader.dataset)}")
    print(f"   Validation samples: {len(val_dataloader.dataset)}")

    # Create validation iterator once to avoid same batch issue
    val_iterator = iter(val_dataloader)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_kl_loss = 0.0
        total_reg_loss = 0.0

        for batch_idx, (batch_data, soft_labels, batch_tokens, batch_rgb) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            soft_labels = soft_labels.to(device)

            _, _, logits_batch, values_batch, latent_list, edge_index_list, edge_attr_list = model(batch_data)

            batch_size = logits_batch.shape[0]

            batch_kl_loss = 0
            batch_reg_loss = 0

            for i in range(batch_size):
                sample_logits = logits_batch[i]  # [seq_len, num_classes]
                sample_soft_labels = soft_labels[i]  # [num_classes]
                sample_original = batch_data[i]  # [seq_len, embed_dim]

                averaged_logits = torch.mean(sample_logits, dim=0)  # [num_classes]
                gae_probs = F.softmax(averaged_logits, dim=0)

                kl_loss = F.kl_div(gae_probs.log(), sample_soft_labels, reduction='batchmean')
                batch_kl_loss += kl_loss

                if i < len(latent_list):
                    latent_sample = latent_list[i]  # [seq_len, latent_dim]
                    latent_reg = torch.norm(latent_sample, p=1) * 0.001
                    batch_reg_loss += latent_reg

            avg_kl_loss = batch_kl_loss / batch_size
            avg_recon_loss = batch_reg_loss / batch_size

            loss = avg_kl_loss + 0.1 * avg_recon_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_kl_loss += avg_kl_loss.item()
            total_reg_loss += avg_recon_loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = total_kl_loss / len(train_dataloader)
        avg_recon = total_reg_loss / len(train_dataloader)

        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'  Total Loss: {avg_loss:.6f} | KL Loss: {avg_kl:.6f} | Reg Loss: {avg_recon:.6f}')

        if is_save:
            save_loss_log(
                epoch=epoch + 1,
                avg_loss=avg_loss,
                data_save_path=data_save_path
            )

        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Learning Rate: {current_lr:.2e}')

        if (epoch % 10 == 0) or (epoch == epochs - 1):
            print(f'\n🔍 Detailed Analysis - Epoch {epoch + 1}:')

            model.eval()

            with torch.no_grad():
                # Get a fresh batch from validation set with tokens and RGB
                try:
                    test_batch, test_labels, test_tokens, test_rgb = next(val_iterator)
                except StopIteration:
                    # If we've exhausted the validation iterator, create a new one
                    val_iterator = iter(val_dataloader)
                    test_batch, test_labels, test_tokens, test_rgb = next(val_iterator)

                test_batch = test_batch.to(device)
                test_labels = test_labels.to(device)

                _, _, test_logits, _, test_latent, test_edges, _ = model(test_batch)

                correct_predictions = 0
                total_predictions = 0

                for i in range(min(3, test_logits.shape[0])):
                    sample_logits = test_logits[i]
                    averaged_logits = torch.mean(sample_logits, dim=0)
                    gae_probs = F.softmax(averaged_logits, dim=0)

                    gae_pred_idx = gae_probs.argmax().item()
                    gae_pred = classes[gae_pred_idx]

                    gpt_pred_idx = test_labels[i].argmax().item()
                    gpt_pred = classes[gpt_pred_idx]

                    confidence = gae_probs.max().item()
                    is_correct = gae_pred_idx == gpt_pred_idx

                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1

                    status = "✅" if is_correct else "❌"

                    # Display actual RGB values and tokens for this sample
                    r, g, b = test_rgb[i]
                    current_tokens = test_tokens[i]
                    print(
                        f'    Sample {i + 1}: {status} RGB({r},{g},{b}) GPT="{gpt_pred}" | GAE="{gae_pred}" (conf={confidence:.3f})')
                    print(f'      Tokens: {current_tokens[:5]}...' if len(
                        current_tokens) > 5 else f'      Tokens: {current_tokens}')

                    # Show GPT soft labels for comparison
                    gpt_soft_probs = test_labels[i].cpu().numpy()
                    gpt_prob_str = " | ".join([f"{cls}:{prob:.2f}" for cls, prob in zip(classes, gpt_soft_probs)])
                    print(f'      Probabilities GPT: {gpt_prob_str}')

                    # Show GAE probabilities
                    prob_str = " | ".join([f"{cls}:{prob:.2f}" for cls, prob in zip(classes, gae_probs.cpu().numpy())])
                    print(f'      Probabilities GAE: {prob_str}')

                    latent_sample = test_latent[i]
                    edge_index_sample = test_edges[i]

                    if edge_index_sample is not None:
                        num_edges = edge_index_sample.shape[1] if edge_index_sample.numel() > 0 else 0
                        edge_attr_sample = torch.ones(num_edges) if num_edges > 0 else torch.tensor([])

                        try:
                            # Use the actual tokens from this specific sample
                            param_schema = create_param_schema_from_tokens(current_tokens, "word")

                            # Ensure schema matches latent dimensions
                            if len(param_schema) > latent_sample.shape[0]:
                                param_schema = param_schema[:latent_sample.shape[0]]
                            elif len(param_schema) < latent_sample.shape[0]:
                                for j in range(len(param_schema), latent_sample.shape[0]):
                                    param_schema.append(NodeDescriptor(f"pad_{j}", "word"))

                            graph_title = f"RGB({r},{g},{b}) GPT={gpt_pred} | GAE={gae_pred}"

                            graph = create_graph(
                                latent_points=latent_sample,
                                edge_index=edge_index_sample,
                                edge_attr=edge_attr_sample,
                                parameters={
                                    "epoch": epoch + 1,
                                    "name": f"{name}-{'success' if is_correct else 'failed'}",
                                    "title": graph_title
                                },
                                param_schema=param_schema,
                                group_marker_map=group_marker_map,
                                is_visual=is_visual,
                                visual_save_path=visual_save_path
                            )
                            print(
                                f'      Graph created with {len(param_schema)} nodes: {[node.name for node in param_schema]}')
                        except Exception as e:
                            print(f"⚠️ Graph visualization failed: {e}")

                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f'    Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})')

            model.train()

        if current_lr < 1e-6:
            print(f"ℹ️ Early stopping: learning rate too small ({current_lr:.2e})")
            break

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_loss': avg_loss,
        'epochs_trained': epochs,
        'model_config': {
            'input_dim': model.encoder[0].in_features,
            'output_dim': 3,
            'hidden_dim': 64
        },
        'classes': classes,
        'final_accuracy': accuracy if 'accuracy' in locals() else 0.0
    }, f"{model_save_path}/{model_filename}")

    print(f"\n✅ Training completed!")
    print(f"💾 Model saved to {model_save_path}/{model_filename}")
    if 'accuracy' in locals():
        print(f"🎯 Final accuracy: {accuracy:.1%}")

def main():
    config = tyro.cli(Config)
    # Initialize the extractor with different model options
    print("\n🎯 Available Models:")
    print("1. distilgpt2 (82M params) - Fast, good for experimentation")
    print("2. gpt2 (124M params) - Slightly larger, more capable")
    print("3. distilbert-base-uncased (66M params) - BERT-based, different architecture")

    # You can change this to experiment with different models
    MODEL_NAME = "distilgpt2"  # Change to "gpt2" or "distilbert-base-uncased" to try others

    print(f"\n🚀 Initializing with {MODEL_NAME}...")
    extractor = ColorTokenVectorExtractor(MODEL_NAME)

    model = GraphAutoEncoder(
        input_dim=extractor.model.config.hidden_size,
        output_dim=3,
        hidden_dim=64,
        graph_fn=algorithms["delaunay"]
    )

    train_dataloader, val_dataloader = generate_color_token_dataloader(
        extractor=extractor,
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        classes=classes,
        epsilon=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        validation_split=0.2
    )

    # Create a timestamped model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"gpt-to-gae_{timestamp}.pth"

    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        name="gae-kl-train",
        epochs=config.epochs,                         # config.epochs
        lr=config.lr,                         # config.lr
        factor=config.factor,                        # config.factor
        patience=config.patience,                        # config.patience
        model_save_path=config.model_save_path,
        model_filename=model_filename,
        is_visual=config.visualise,
        visual_save_path=config.visual_save_path,
    )

if __name__ == "__main__":
    main()