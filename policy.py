
import torch
from value_network import ValueNetwork
from gamestate import GameState
import random
import numpy as np


class EpGreedyPolicy:
    def __init__(self, value_network: ValueNetwork, eps=0.0):
        self._value_network = value_network
        self.eps = eps

    def get_move(self, gs: GameState, eps=None):
        """ Gets a move according to the epsilon-greedy policy. """
        assert not gs.is_game_over(), "Can't make move since game is over."
        if not gs.is_blues_turn:
            return self.get_move(gs.reverse_red_and_blue())

        if eps is None:
            eps = self.eps

        if random.random() < eps:
            return RandomPolicy.get_move(gs)
        else:
            with torch.no_grad():
                q_values = self._value_network(gs)
                # Make valid move with maximum Q-value.
                # Set already revealed squares to have -inf Q-value.
                q_values[gs.visible] = float('-inf')
                flattened_argmax = q_values.view(-1).argmax().item()
                move = (flattened_argmax // gs.board_size, flattened_argmax % gs.board_size)
                #assert not gs.visible[move].item(), \
                #    f'Making move on revealed square:\n {q_values} \n {gs}'
        return move


class OptimalPolicy(EpGreedyPolicy):
    def __init__(self, value_network: ValueNetwork):
        super().__init__(value_network=value_network, eps=0.0)


class RandomPolicy:
    @staticmethod
    def get_move(gs: GameState):
        assert not gs.is_game_over(), "Can't make move since game is over."
        # Indices of all valid moves.
        valid_moves = (~gs.visible).view(-1).nonzero().view(-1)
        # Randomly choose a valid move.
        flattened_index = valid_moves[np.random.randint(0, valid_moves.shape[0])].item()
        move = (flattened_index // gs.board_size, flattened_index % gs.board_size)
        #assert not gs.visible[move].item(), 'Making move on revealed square'
        return move

