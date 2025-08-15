import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColorTokenVectorExtractor:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the Token Vector Extractor for multimodal preprocessing
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

        print(f"Loaded model: {model_name}")

    def extract_token_vectors(self, text: str, layer_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        Extract token vectors from the model for a given text
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
            clean_token = token.replace('Ġ ', ' ').replace('ĠĊ', '\n')
            if clean_token.startswith(' '):
                clean_token = clean_token[1:]  # Remove leading space for display

            token_vector_map[clean_token] = token_vectors[i]

        return token_vector_map

    def create_manual_rgb_tokens(self, r: float, g: float, b: float) -> Dict[str, np.ndarray]:
        """
        Create averaged embeddings for color+value combinations
        Following the multimodal pipeline: embedding(color) + embedding(value) / 2
        """
        # Format values to 2 decimal places as in the multimodal code
        red_text = f"red {r:.2f}"
        green_text = f"green {g:.2f}"
        blue_text = f"blue {b:.2f}"
        base_text = "What color is"

        # Extract vectors for each component
        red_vectors = self.extract_token_vectors(red_text)
        green_vectors = self.extract_token_vectors(green_text)
        blue_vectors = self.extract_token_vectors(blue_text)
        base_vectors = self.extract_token_vectors(base_text)

        # Find the color name and number vectors
        def find_vectors(vectors_dict, color_name, number_str):
            color_vec = None
            number_tokens = []

            for token, vec in vectors_dict.items():
                clean_token = token.strip().lower()
                if clean_token == color_name.lower():
                    color_vec = vec
                elif any(char.isdigit() or char == '.' for char in clean_token):
                    # Collect all tokens that are part of the number
                    number_tokens.append(vec)

            # Average all number-related tokens
            if number_tokens:
                number_vec = np.mean(number_tokens, axis=0)
            else:
                number_vec = None

            return color_vec, number_vec

        # Get individual component vectors
        red_color_vec, red_num_vec = find_vectors(red_vectors, 'red', f"{r:.2f}")
        green_color_vec, green_num_vec = find_vectors(green_vectors, 'green', f"{g:.2f}")
        blue_color_vec, blue_num_vec = find_vectors(blue_vectors, 'blue', f"{b:.2f}")

        # Create combined tokens
        combined_tokens = {}

        # Add base question tokens (What, color, is)
        for token, vec in base_vectors.items():
            if token.strip().lower() not in ['red', 'green', 'blue']:
                combined_tokens[token] = vec

        # Average the color name and number vectors to create combined tokens
        if red_color_vec is not None and red_num_vec is not None:
            combined_tokens[f'red{r:.2f}'] = (red_color_vec + red_num_vec) / 2

        if green_color_vec is not None and green_num_vec is not None:
            combined_tokens[f'green{g:.2f}'] = (green_color_vec + green_num_vec) / 2

        if blue_color_vec is not None and blue_num_vec is not None:
            combined_tokens[f'blue{b:.2f}'] = (blue_color_vec + blue_num_vec) / 2

        return combined_tokens

    def process_rgb_to_embeddings(self, r: float, g: float, b: float) -> torch.Tensor:
        """
        Main function to convert RGB values to token embeddings following multimodal pipeline
        """
        # Create the manual RGB tokens with averaged embeddings
        token_vectors = self.create_manual_rgb_tokens(r, g, b)

        # Convert to tensor and return
        if token_vectors:
            embeddings = list(token_vectors.values())
            embeddings_array = np.array(embeddings)
            return torch.tensor(embeddings_array, dtype=torch.float32)
        else:
            # Fallback: return empty tensor if no tokens found
            return torch.zeros((1, self.model.config.hidden_size), dtype=torch.float32)