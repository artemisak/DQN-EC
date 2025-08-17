import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4
import torch
from scipy.stats import spearmanr

from suppelemtary import ColorTokenVectorExtractor
from multimodal import ColorTokenVectorExtractor

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
    4: [0, 0, 1, 0],
    5: [0, 1, 1, 0],
    6: [0, 1, 1, 1],
}

SPEAKER_BACKWARD_MAPPING = {
    (0, 0, 0, 0): 1,
    (0, 0, 0, 1): 2,
    (0, 0, 1, 1): 3,
    (0, 0, 1, 0): 4,
    (0, 1, 1, 0): 5,
    (0, 1, 1, 1): 6,
}

class SyntheticData:

    def __init__(self, num_samples=1024, batch_size=64, model_name="distilgpt2"):

        self.num_samples = num_samples
        self.batch_size = batch_size

        self.extractor = ColorTokenVectorExtractor(model_name)

        self.listener = []
        self.speaker = []
        self.messages = []

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
        self.speaker = self._pad_speaker_embeddings(self.speaker)

        self.loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(range(len(self.messages))).to(device)),
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
        result = self.extractor.analyze_color_tokens_average(r, g, b)

        token_vectors = result['token_vectors']  # A dict: token -> numpy array
        num_tokens = len(token_vectors)
        embed_dim = self.extractor.model.config.hidden_size
        speaker_obs = torch.zeros(num_tokens, 4 + embed_dim)
        # Convert list of token vectors to a tensor
        vecs = torch.tensor(np.stack(list(token_vectors.values())), dtype=torch.float32)
        # Assign forward mapping for first 4 dims
        speaker_obs[:, :4] = torch.tensor(list(SPEAKER_FORWARD_MAPPING.values())[:num_tokens])
        # Assign token vectors to remaining dims
        speaker_obs[:, 4:] = vecs
        return speaker_obs

    def _pad_speaker_embeddings(self, speaker_list):
        if not speaker_list:
            return torch.empty(0)

        max_len = max(emb.shape[0] for emb in speaker_list)
        embed_dim = speaker_list[0].shape[1]

        padded_speakers = []
        for embeddings in speaker_list:
            pad_len = max_len - embeddings.shape[0]
            if pad_len > 0:
                padding = torch.zeros((pad_len, embed_dim))
                padded_embeddings = torch.cat([embeddings, padding], dim=0)
            else:
                padded_embeddings = embeddings
            padded_speakers.append(padded_embeddings)

        return torch.stack(padded_speakers)
