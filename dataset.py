import os

import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4
import torch

from suppelemtary import ColorTokenVectorExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LISTENER_FORWARD_MAPPING = {
    1: [0, 0, 0, 0],
    2: [0, 0, 0, 1],
    3: [0, 0, 1, 1],
    4: [0, 0, 1, 0],
    5: [0, 1, 1, 0],
    6: [0, 1, 1, 1],
    7: [0, 1, 0, 1],
    8: [0, 1, 0, 0],
}

LISTENER_BACKWARD_MAPPING = {
    (0, 0, 0, 0): 1,
    (0, 0, 0, 1): 2,
    (0, 0, 1, 1): 3,
    (0, 0, 1, 0): 4,
    (0, 1, 1, 0): 5,
    (0, 1, 1, 1): 6,
    (0, 1, 0, 1): 7,
    (0, 1, 0, 0): 8,
}

SPEAKER_FORWARD_MAPPING = {
    1: [0, 0, 0, 0],
    2: [0, 0, 0, 1],
    3: [0, 0, 1, 1],
}

SPEAKER_BACKWARD_MAPPING = {
    (0, 0, 0, 0): 1,
    (0, 0, 0, 1): 2,
    (0, 0, 1, 1): 3,
}

class SyntheticData:

    def __init__(self, num_samples=1024, batch_size=64, model_name="distilgpt2", data_path="data/synthetic_dataset.pt"):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.data_path = data_path
        self.speaker_filter = ['red', 'green', 'blue']
        self.listener = []
        self.speaker = []
        self.messages = []

        if os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            self.load()
        else:
            print("Data not found. Generating new data...")
            self.extractor = ColorTokenVectorExtractor(model_name)

            env = simple_speaker_listener_v4.parallel_env(max_cycles=self.num_samples)
            env.reset()

            while env.agents:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                msg, *_ = env.step(actions)
                self.messages.append(torch.tensor([*msg['listener_0'][:8], *msg['speaker_0'][:3]]))
                self.listener.append(self.prepare_listener(msg['listener_0'][:8]))
                self.speaker.append(self.prepare_speaker(msg['speaker_0'][:3]))
            env.close()

            self.messages = torch.stack(self.messages)
            self.listener = torch.stack(self.listener)
            self.speaker = torch.stack(self.speaker)

            self.save()

        self.loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(range(num_samples)).to(device)),
            batch_size=self.batch_size,
            shuffle=True
        )

    def prepare_listener(self, listener_msg):
        listener_obs = torch.zeros(len(listener_msg), 5)
        listener_obs[:, :4] = torch.tensor(list(LISTENER_FORWARD_MAPPING.values())[:len(listener_msg)])
        listener_obs[0:len(listener_msg), 4] = torch.tensor(listener_msg)
        return listener_obs

    def prepare_speaker(self, speaker_msg):
        r, g, b = speaker_msg[0], speaker_msg[1], speaker_msg[2]

        # Use the extractor to get the structured data, including token_vectors
        result = self.extractor.process_rgb_to_embeddings(r, g, b)

        # Get the token_vectors dictionary
        token_vectors = result['token_vectors']

        # Filter the tokens based on the provided list
        filtered_vectors = {}
        for token, vector in token_vectors.items():
            if any(key_part in token for key_part in self.speaker_filter):
                filtered_vectors[token] = vector

        if not filtered_vectors:
            print("Warning: No matching tokens found. Returning empty tensor.")
            embed_dim = self.extractor.model.config.hidden_size
            return torch.zeros((0, 4 + embed_dim), dtype=torch.float32)

        num_tokens = len(filtered_vectors)
        embed_dim = self.extractor.model.config.hidden_size

        speaker_obs = torch.zeros(num_tokens, 4 + embed_dim)

        vecs = torch.tensor(np.stack(list(filtered_vectors.values())), dtype=torch.float32)

        speaker_obs[:, :4] = torch.tensor(list(SPEAKER_FORWARD_MAPPING.values())[:num_tokens])
        speaker_obs[:, 4:] = vecs

        return speaker_obs

    def save(self):
        """Saves the generated data to a file."""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        torch.save({
            'listener': self.listener,
            'speaker': self.speaker,
            'messages': self.messages
        }, self.data_path)
        print(f"Dataset saved to {self.data_path}")

    def load(self):
        """Loads the generated data from a file."""
        data = torch.load(self.data_path)
        self.listener = data['listener']
        self.speaker = data['speaker']
        self.messages = data['messages']