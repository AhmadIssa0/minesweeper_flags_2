
import torch
import torch.nn as nn
from gamestate import GameState
from typing import Union


class ValueNetwork(nn.Module):
    """ Represents Q-values of state-action pairs assuming it is blue player to move. """
    n_channels = 11

    def __init__(self, network, board_size, device):
        super().__init__()
        self._network = network
        self.board_size = board_size
        self._device = device

    def forward(self, inputs: Union[GameState, torch.Tensor]) -> torch.Tensor:
        """

        :param inputs: if tensor, must have shape [batch size, state channels, board size, board size]
        :return:
        """
        if isinstance(inputs, GameState):
            nn_inputs = self.game_state_to_input(inputs)
        else:
            nn_inputs = inputs

        nn_inputs = nn_inputs.to(device=self._device)
        # squeeze the channel dimension
        outputs = self._network(nn_inputs).squeeze(1)

        if isinstance(inputs, GameState) or inputs.shape[0] == 1:
            # We didn't start with a batch, so squeeze the batch dimension.
            outputs = outputs.squeeze(0)
        return outputs

    def game_state_to_input(self, gs: GameState):
        """
        Converts a game state to a tensor that can be input into the value network. Our network
        assumes blue's turn to move, if it's red's turn to move we switch captured flags and treat
        it as blue's turn.

        :param gs:
        :return: tensor of shape (n_channels, board_size, board_size). The different channels are:
            channel 0: 1 if visible, else 0.
            channel 1-8: 1 if square is visible and contains this many adjacent mines, else 0.
            channel 9: 1 if blue captured flag, else 0.
            channel 10: 1 if red captured flag, else 0.
        """
        if not gs.is_blues_turn:
            return self.game_state_to_input(gs.reverse_red_and_blue())

        network_input = torch.zeros((self.n_channels, self.board_size, self.board_size), dtype=torch.float)
        visible_mask = gs.visible.to(torch.float)
        network_input[0] = visible_mask

        for i in range(1, 9):
            adj_mines_mask = (gs.board == i).to(torch.float)
            network_input[i] = visible_mask * adj_mines_mask * gs.board

        for r, c in gs.blue_flags_captured:
            network_input[9, r, c] = 1.0
        for r, c in gs.red_flags_captured:
            network_input[10, r, c] = 1.0

        return network_input.unsqueeze(0)


class VanillaConv(nn.Module):

    def __init__(self, n_channels):
        """

        :param n_channels: Number of computational channels kept throughout the convolutions.
        """
        super().__init__()
        self.input_channels = ValueNetwork.n_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding='same', groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=1, kernel_size=(1, 1), stride=(1, 1),
                      padding='same', groups=1),
        )

    def forward(self, inputs: torch.Tensor):
        """

        :param inputs: tensor returned by ValueNetwork.game_state_to_input()
        :return:
        """
        return self.model(inputs)
