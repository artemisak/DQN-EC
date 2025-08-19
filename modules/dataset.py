import os

from pettingzoo.mpe import simple_speaker_listener_v4
import torch

from modules.utils import prepare_listener, prepare_speaker
from modules.vectorizer import TokenVectorizer


class SyntheticData:

    def __init__(self, num_samples=1024, batch_size=64, model_name="distilgpt2", data_path="checkpoints/dataset.pt"):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.data_path = data_path
        self.filter = ['red', 'green', 'blue']
        self.listener = []
        self.speaker = []
        self.messages = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            self.load()
        else:
            print("Data not found. Generating new data...")
            self.vectorizer = TokenVectorizer(model_name)

            env = simple_speaker_listener_v4.parallel_env(max_cycles=self.num_samples)
            env.reset()

            while env.agents:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                obs, *_ = env.step(actions)
                self.messages.append(torch.tensor([*obs['listener_0'], *obs['speaker_0']]))
                self.listener.append(prepare_listener(obs['listener_0'], shuffle=True))
                self.speaker.append(prepare_speaker(obs['speaker_0'], vectorizer=self.vectorizer,
                                                    filter=self.filter, shuffle=True))
            env.close()

            self.messages = torch.stack(self.messages)
            self.listener = torch.stack(self.listener)
            self.speaker = torch.stack(self.speaker)

            self.save()

        self.loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(range(num_samples)).to(self.device)),
            batch_size=self.batch_size,
            shuffle=True
        )

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