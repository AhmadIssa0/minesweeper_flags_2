import torch
import torch.nn as nn
import torch.nn.functional as F
from experience_replay import *
from policy import *
from gamestate import GameState
from value_network import ValueNetwork
from typing import Tuple
from evaluation import score_match
import copy


class Trainer:

    def __init__(self, value_network: ValueNetwork, board_size, replay_buffer_size, learning_rate,
                 device, save_path=None, alpha=0.1, gamma=0.9, comparison_policy=None):
        """

        :param value_network:
        :param save_path: filepath to save value network checkpoints during training. if None doesn't save.
        :param alpha: Q-learning learning rate.
        :param gamma: discount factor.
        """
        self._value_network = value_network.to(device=device)
        self._target_network = copy.deepcopy(self._value_network)
        self._board_size = board_size
        self._optimizer = torch.optim.Adam(value_network.parameters(), lr=learning_rate)
        self._policy = EpGreedyPolicy(value_network=self._value_network, eps=0.02)
        self._save_path = save_path
        self._alpha = alpha
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._er_buffer = ExperienceReplay(size=replay_buffer_size, device=device)
        self._opponent_policy = comparison_policy if comparison_policy else RandomPolicy

    def _move_to_action(self, move: Tuple[int, int]):
        action = torch.full((self._board_size, self._board_size), fill_value=False, dtype=torch.bool)
        action[move] = True
        return action

    def train(self, max_steps, batch_size, network_update_freq=5000):
        step = 0
        gs = GameState.create_new_game(self._board_size)
        loss = collections.deque(maxlen=1000)

        self._target_network = copy.deepcopy(self._value_network)
        while step < max_steps:
            if step % 500 == 0:
                print(f'Step: {step}')
                if len(loss) == loss.maxlen:
                    print(f'Loss: {sum(loss) / len(loss)}')
            if step % 500 == 0:
                score = score_match(self._board_size,
                                    OptimalPolicy(self._value_network),
                                    self._opponent_policy, num_games=100)
                print(f'Score in 100 game match: {score}')
            if step % 2000 == 0:
                if self._save_path:
                    torch.save(self._value_network, self._save_path)
                    print(f'Saved value network to path: {self._save_path}')
            if step < 2000:
                for g in self._optimizer.param_groups:
                    g['lr'] = self._learning_rate * step / 2000
            if step % network_update_freq == 0:
                self._target_network = copy.deepcopy(self._value_network)

            for _ in range(20):
                if gs.is_game_over():
                    gs = GameState.create_new_game(self._board_size)

                move = self._policy.get_move(gs, eps=0.1 * (1. - step / max_steps))  # set eps here
                state = self._value_network.game_state_to_input(gs).squeeze(0)
                action = self._move_to_action(move)
                reward = 1 if gs.board[move] == GameState.FLAG else 0
                gs = gs.make_move(*move)
                done = True if gs.is_game_over() else False
                new_state = self._value_network.game_state_to_input(gs).squeeze(0)
                self._er_buffer.append(Experience(state, action, reward, done, new_state))

            if len(self._er_buffer) > batch_size * 10:
                loss.append(self._train_one_step(**self._er_buffer.sample(batch_size)))

            step += 1

    def _train_one_step(self, states, actions, rewards, dones, new_states):
        """

        :param states: tensor to be input into value network (blue to move)
        :param actions: bool tensor shape [batch size, board size, board size], true in location move was made
        :param rewards: int tensor shape [batch size]
        :param dones: bool tensor shape [batch size]
        :param new_states: tensor to be input into value network (blue to move)
        :return:
        """
        batch_size = states.shape[0]
        self._optimizer.zero_grad()
        with torch.no_grad():
            target_next_q_values = self._target_network(new_states)
            value_next_q_values = self._value_network(new_states)
            # Use value network to select the best action. Then use target network's estimate of the q-value of that
            # action.
            best_action_indices = value_next_q_values.view(batch_size, -1).argmax(dim=1, keepdim=False)  # (N)
            best_action_mask = F.one_hot(best_action_indices, num_classes=self._board_size ** 2)  # (N, board ** 2)
            max_next_q = (target_next_q_values.view(batch_size, -1) * best_action_mask).sum(dim=1)

        # Q_new = (1 - alpha) Q_old + alpha * (r - gamma * max_a Q_next)
        # Q_new - Q_old = alpha * [r - gamma * max_a Q_next - Q_old]
        move_mask = actions.to(torch.int)
        q_values = self._value_network(states)
        q_value_of_move = (q_values * move_mask).sum(dim=(-1, -2))
        # if rewards == 1 we get another turn, that determines if we add reward of next mover, or subtract it
        turn_sign = rewards * 2 - 1
        temporal_difference = rewards - q_value_of_move + turn_sign * (~dones).to(torch.int) * self._gamma * max_next_q
        mse_loss = (self._alpha ** 2) * torch.mean(temporal_difference * temporal_difference, dim=0)
        mse_loss.backward()
        self._optimizer.step()
        return mse_loss.item()
