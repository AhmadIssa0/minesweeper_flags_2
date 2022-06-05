import torch.cuda

import policy
from gamestate import GameState
from value_network import *
from trainer import Trainer
import torch
from neural_networks import *
from evaluation import score_match
from two_player import GameManager


def test():
    board_size = 5
    value_network = ValueNetwork(VanillaConv(), board_size=board_size)
    gs = GameState.create_new_game(board_size=board_size)
    print(gs)
    move = policy.RandomPolicy.get_move(gs)
    print(f'Move: {move}')
    gs = gs.make_move(*move)
    print(gs)
    print(value_network(gs))
    print(value_network(gs).shape)

def eval_trained_model(board_size, save_path):
    value_network = torch.load(save_path)
    print(f'Number of params: {sum(p.numel() for p in value_network.parameters() if p.requires_grad)}')
    print(score_match(board_size, policy.OptimalPolicy(value_network), policy.RandomPolicy, num_games=100))

def human_vs_ai(value_network, board_size, human_first):
    ai_policy = policy.OptimalPolicy(value_network)
    GameManager(human_first, ai_policy, board_size).play_loop()

def train_ai(value_network, board_size, **kwargs):
    trainer = Trainer(value_network, board_size, 20000, device=device, save_path=save_path, learning_rate=1e-4, **kwargs)
    trainer.train(max_steps=200000, batch_size=300)
    print(score_match(board_size, policy.RandomPolicy, policy.RandomPolicy, num_games=100))
    print(score_match(board_size, policy.OptimalPolicy(value_network), policy.RandomPolicy, num_games=100))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    board_size = 10
    save_path = 'convnext2_b10.p'
    load_path = 'convnext2_b10.p'
    value_network = torch.load(load_path)

    human_vs_ai(value_network, board_size, human_first=True)
    #value_network = ValueNetwork(VanillaConv(n_channels=20), board_size=board_size, device=device)
    #value_network = ValueNetwork(ConvNeXt(input_channels=ValueNetwork.n_channels, output_channels=1,
    #                             intermediary_channels=50, n_blocks=10), board_size=board_size, device=device)
    print(f'Number of params: {sum(p.numel() for p in value_network.parameters() if p.requires_grad)}')

    value_network2 = torch.load('convnext2_b10.p')
    train_ai(value_network, board_size, comparison_policy=policy.OptimalPolicy(value_network2))
    #print(score_match(board_size, policy.OptimalPolicy(value_network2), policy.OptimalPolicy(value_network), 1000))

    #print(score_match(board_size, policy.OptimalPolicy(value_network), policy.RandomPolicy, num_games=1000))
