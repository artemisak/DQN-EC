import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from graph_autoencoder import GraphAutoEncoder, algorithms, create_graph,NodeDescriptor

print("🎨 Color Token Vector Extractor")
print("=" * 40)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

    def analyze_color_tokens(self, r: float, g: float, b: float) -> Dict:
        """
        Analyze token vectors for a color question
        """
        question = f"What color is this {r} R {g} G {b} B"
        token_vectors = self.extract_token_vectors(question)

        # Simple color prediction based on RGB values
        predicted_color = self.predict_color_from_rgb(r, g, b)

        return {
            'question': question,
            'rgb_values': (r, g, b),
            'predicted_color': predicted_color,
            'token_vectors': token_vectors,
            'vector_stats': self.get_vector_statistics(token_vectors)
        }

    def predict_color_from_rgb(self, r: int, g: int, b: int) -> str:
        """Simple rule-based color prediction"""
        colors = {
            'red': (r > 127 and r > g and r > b),
            'green': (g > 127 and g > r and g > b),
            'blue': (b > 127 and b > r and b > g),
            'yellow': (r > 127 and g > 127 and b < 76),
            'cyan': (g > 127 and b > 127 and r < 76),
            'magenta': (r > 127 and b > 127 and g < 76),
            'orange': (r > 153 and 76 < g < 153 and b < 76),
            'purple': (r > 76 and b > 127 and g < 102),
            'white': (r > 204 and g > 204 and b > 204),
            'black': (r < 51 and g < 51 and b < 51),
            'gray': (abs(r - g) < 25 and abs(g - b) < 25 and 51 < r < 204),
        }

        for color, condition in colors.items():
            if condition:
                return color

        # Default to dominant channel
        if r >= g and r >= b:
            return 'red'
        elif g >= r and g >= b:
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

# Initialize the extractor with different model options
print("\n🎯 Available Models:")
print("1. distilgpt2 (82M params) - Fast, good for experimentation")
print("2. gpt2 (124M params) - Slightly larger, more capable")
print("3. distilbert-base-uncased (66M params) - BERT-based, different architecture")

# You can change this to experiment with different models
MODEL_NAME = "distilgpt2"  # Change to "gpt2" or "distilbert-base-uncased" to try others

print(f"\n🚀 Initializing with {MODEL_NAME}...")
extractor = ColorTokenVectorExtractor(MODEL_NAME)

# Main example - exactly as requested
print("\n" + "="*60)
print("🧪 MAIN EXAMPLE - COLOR ANALYSIS")
print("="*60)

# Your exact example
test_question = "What color is this 15 R 15 G 65 B"
print(f"📝 Question: {test_question}")

# Extract token vectors
result = extractor.analyze_color_tokens(15, 15, 65)

# Print in your requested format
extractor.print_detailed_vectors(result['token_vectors'])


print(f"\n🎨 Answer: {result['predicted_color']}")

# Additional analysis
print(f"\n📊 Vector Statistics:")
stats = result['vector_stats']
print(f"   • Number of tokens: {stats['num_tokens']}")
print(f"   • Vector dimension: {stats['vector_dimension']}")
print(f"   • Average vector magnitude: {stats['mean_magnitude']:.2f}")

# Interactive functions
def quick_color_analysis(r: float, g: float, b: float):
    """Quick analysis function for testing different colors"""
    result = extractor.analyze_color_tokens(r, g, b)
    print(f"\n🎨 RGB({r}, {g}, {b}) Analysis:")
    print(f"Predicted color: {result['predicted_color']}")
    extractor.print_detailed_vectors(result['token_vectors'], max_dims=5)
    pass

def compare_colors(colors: List[Tuple[float, float, float]]):
    """Compare token vectors for different colors"""
    results = []

    print(f"\n🔄 Comparing {len(colors)} colors:")
    for i, (r, g, b) in enumerate(colors):
        result = extractor.analyze_color_tokens(r, g, b)
        results.append(result)
        print(f"{i+1}. RGB({r}, {g}, {b}) → {result['predicted_color']}")

    return results

def analyze_specific_tokens(text: str, target_tokens: List[str]):
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

model = GraphAutoEncoder(
    input_dim=stats['vector_dimension'],
    output_dim=3,
    hidden_dim=64,
    graph_fn=algorithms["delaunay"]
)

group_marker_map = {
    "word": "o"
}

def create_param_schema_from_tokens(tokens, default_group="word"):
    """
    Создает param_schema для списка токенов.
    Каждому токену назначается группа (по умолчанию 'token', можно изменить логику группировки).
    """
    return [NodeDescriptor(name=tok, group=default_group) for tok in tokens]


def evaluate_model(model, batch, model_name,batch_idx=0):
    with torch.no_grad():
        _, _, logits, values, latent_list, edge_index_list, edge_attr_list = model(batch)
    print("Batch:")
    print(batch)
    idx = 0
    obs = batch[idx]

    tokens = list(result['token_vectors'].keys())
    param_schema = create_param_schema_from_tokens(tokens)

    latent = latent_list[idx]
    edge_index = edge_index_list[idx]
    edge_attr = edge_attr_list[idx]

    G = create_graph(
        latent_points=latent,
        edge_index=edge_index,
        edge_attr=edge_attr,
        parameters={"epoch": 0, "name": f"{model_name}-{batch_idx}"},
        param_schema=param_schema,
        group_marker_map=group_marker_map,
        is_visual=True,
        visual_save_path="results/graphics"
    )

data = result['token_vectors'].values()
batch = torch.tensor(list(data)).unsqueeze(0).to(device)
evaluate_model(model, batch, "test", 1)
