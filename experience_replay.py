
import torch
import collections
import numpy as np

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, size, device):
        """

        :param size:
        :param device: when sampling we put tensors on this device
        """
        self.buffer = collections.deque(maxlen=size)
        self._device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, dones, new_states = zip(*[tuple(self.buffer[i]) for i in indices])

        return {'states': torch.stack(states).to(device=self._device),
                'actions': torch.stack(actions).to(device=self._device),
                'rewards': torch.tensor(rewards, device=self._device),
                'dones': torch.tensor(dones, device=self._device),
                'new_states': torch.stack(new_states).to(device=self._device)}

