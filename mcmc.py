

from gamestate import GameState
import torch
import random
from policy import RandomPolicy


def randomly_place_flags(board: torch.IntTensor, num_flags_to_place):
    num_flags_placed = 0
    board_size = board.shape[0]
    while num_flags_placed < num_flags_to_place:
        row = random.randint(0, board_size - 1)  # includes endpoints
        col = random.randint(0, board_size - 1)
        if board[row, col] != GameState.FLAG:
            num_flags_placed += 1
        board[row, col] = GameState.FLAG


def generate_random_board(gs: GameState):
    """Generates a random board state satisfying all visible constraints from the current game state."""
    # Keep all visible flags fixed
    fixed_flags = (gs.board == GameState.FLAG) & gs.visible
    board = torch.full_like(gs.board, GameState.EMPTY)
    board[fixed_flags] = GameState.FLAG

    # Place remaining flags randomly
    board_size = gs.board_size
    total_flags = gs.num_total_flags(board_size)
    num_variable_flags = total_flags - fixed_flags.sum().item()
    randomly_place_flags(board, num_variable_flags)

    return board


if __name__ == '__main__':
    gs = GameState.create_new_game(12)

    for i in range(10):
        gs = gs.make_move(*RandomPolicy.get_move(gs))

    print(gs)
    print(gs.board_as_string())
    gs.board = generate_random_board(gs)
    print(gs.board_as_string())
