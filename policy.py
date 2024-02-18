
import torch
from value_network import ValueNetwork
from gamestate import GameState
import random
import numpy as np
from mcmc_vectorized import sample_hidden_boards


class Policy:
    def get_move(self, gs: GameState):
        return NotImplementedError()


class EpGreedyPolicy(Policy):
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


class NetworkWithSearchPolicy(Policy):
    def __init__(self, value_network: ValueNetwork, eps=0.0, device='cuda'):
        self._value_network = value_network
        self.eps = eps
        self.device = device

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
                # q_values = self._value_network(gs)
                # Make valid move with maximum Q-value.
                # Set already revealed squares to have -inf Q-value.
                # q_values[gs.visible] = float('-inf')

                # We'll only search over the top k moves by q-values.
                # top_k_moves = torch.sort(q_values.view(-1), descending=True, dim=-1).indices[:20]
                # row_coords = (top_k_moves // gs.board_size).tolist()
                # col_coords = (top_k_moves % gs.board_size).tolist()
                # moves = list(zip(row_coords, col_coords))
                moves = [(r, c) for c in range(gs.board_size) for r in range(gs.board_size) if not gs.visible[r, c]]

                hidden_boards = sample_hidden_boards(gs, self.device)[:50]
                avg_q_values = [0.0] * len(moves)
                prob_its_a_flag = [0.0] * len(moves)

                for j, new_board in enumerate(hidden_boards):
                    if j < 5:
                        print('Possible board:')
                        print(GameState.board_as_string(new_board))
                    new_gs = gs.with_new_board_state(new_board.to(device='cpu'))
                    for i, (r, c) in enumerate(moves):
                        next_gs = new_gs.make_move(r, c)
                        q_values = self._value_network(next_gs)
                        q_values[gs.visible] = float('-inf')
                        if new_gs.board[r, c] == GameState.FLAG:
                            avg_q_values[i] += (1.0 + q_values.view(-1).max().item()) / len(hidden_boards)
                            prob_its_a_flag[i] += 1.0 / len(hidden_boards)
                        else:
                            avg_q_values[i] -= q_values.view(-1).max().item() / len(hidden_boards)

                print(list(zip(moves, [round(x, 2) for x in prob_its_a_flag], [round(x, 2) for x in avg_q_values])))
                q_value, move, prob_of_flag = max(list(zip(avg_q_values, moves, prob_its_a_flag)), key=lambda x: x[0])
                print('Chosen move:', move, 'prob its a flag:', prob_of_flag, 'q_value:', q_value)
                print('Number of hidden boards considered:', len(hidden_boards))
                #assert not gs.visible[move].item(), \
                #    f'Making move on revealed square:\n {q_values} \n {gs}'
        return move


class RandomPolicy(Policy):
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

