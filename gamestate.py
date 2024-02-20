import torch
from collections import deque
from typing import List, Set, Tuple
import utils
import random
from colorama import Fore, Back, Style


class GameState:
    """Immutable class representing the minesweeper gamestate."""
    FLAG = -1  # negative since positive numbers on board indicate number of adjacent mines
    EMPTY = 0

    def __init__(self, board: torch.IntTensor, visible: torch.BoolTensor, is_blues_turn, blue_flags_captured: Set,
                 red_flags_captured: Set, last_move: Tuple = None):
        self.board = board
        self.visible = visible
        self.is_blues_turn = is_blues_turn
        self.blue_flags_captured = blue_flags_captured
        self.red_flags_captured = red_flags_captured
        self.num_flags_on_board = torch.sum(board == GameState.FLAG).item()
        self.winning_score = self.num_flags_on_board // 2 + 1
        self.board_size = board.size(0)
        self.last_move = last_move

    def with_new_board_state(self, board: torch.IntTensor):
        return GameState(board=board, visible=self.visible.to(device=board.device), is_blues_turn=self.is_blues_turn,
                         blue_flags_captured=self.blue_flags_captured, red_flags_captured=self.red_flags_captured)

    def num_blue_flags(self):
        return len(self.blue_flags_captured)

    def num_red_flags(self):
        return len(self.red_flags_captured)

    def winner(self):
        if not self.is_game_over():
            return None
        elif self.num_red_flags() > self.num_blue_flags():
            return 'Red'
        else:
            return 'Blue'

    def is_visible(self, row, col):
        return self.visible[row, col].item()

    def is_game_over(self):
        return len(self.blue_flags_captured) >= self.winning_score or len(self.red_flags_captured) >= self.winning_score

    def reverse_red_and_blue(self):
        return GameState(board=self.board, visible=self.visible, is_blues_turn=not self.is_blues_turn,
                         blue_flags_captured=self.red_flags_captured, red_flags_captured=self.blue_flags_captured)

    def _create_visible(self, fn):
        return utils.create_2d_tensor_from_func(rows=self.board_size, cols=self.board_size, fn=fn, dtype=torch.bool)

    def is_move_valid(self, row, col):
        if (not (0 <= row < self.board_size and 0 <= col < self.board_size) or
                self.is_game_over() or self.visible[row, col]):
            return False
        else:
            return True

    def make_move(self, row, col):
        assert 0 <= row < self.board_size and 0 <= col < self.board_size, \
            f"Can't make move: ({row}, {col}), since this is not a square on a board of size {self.board_size}."
        assert not self.is_game_over(), "Can't make move since game is already over."
        assert not self.visible[row, col], "Can't make move on square that is already revealed."

        move = (row, col)
        if self.board[row, col] == GameState.FLAG:
            new_visible = self.visible.detach().clone()
            new_visible[row, col] = True
            if self.is_blues_turn:
                new_blue_flags_captured = self.blue_flags_captured.union({(row, col)})
                new_red_flags_captured = self.red_flags_captured.copy()
            else:
                new_blue_flags_captured = self.blue_flags_captured.copy()
                new_red_flags_captured = self.red_flags_captured.union({(row, col)})
            return GameState(board=self.board, visible=new_visible, is_blues_turn=self.is_blues_turn,
                             blue_flags_captured=new_blue_flags_captured, red_flags_captured=new_red_flags_captured,
                             last_move=move)
        elif self.board[row, col] == GameState.EMPTY:
            revealed_squares = self._compute_revealed_squares((row, col))
            new_visible = self._create_visible(lambda r, c: (r, c) in revealed_squares or self.visible[r, c])
            return GameState(board=self.board, visible=new_visible, is_blues_turn=not self.is_blues_turn,
                             blue_flags_captured=self.blue_flags_captured, red_flags_captured=self.red_flags_captured,
                             last_move=move)
        else:  # made a move on a square adjacent to a mine
            new_visible = self.visible.detach().clone()
            new_visible[row, col] = True
            return GameState(board=self.board, visible=new_visible, is_blues_turn=not self.is_blues_turn,
                             blue_flags_captured=self.blue_flags_captured, red_flags_captured=self.red_flags_captured,
                             last_move=move)

    def _compute_revealed_squares(self, init_square: (int, int)) -> Set[Tuple[int, int]]:
        """Given an initial squares, returns a list of squares which will be revealed by opening the initial square."""
        to_visit = deque([init_square])
        connected_component = set()
        while to_visit:
            curr_sq = to_visit.popleft()
            r0, c0 = curr_sq
            connected_component.add((r0, c0))
            # Get the subset of 8 surrounding squares which are either empty or numbers (not flags).
            neighbours = [(r, c) for r in [r0 - 1, r0, r0 + 1] for c in [c0 - 1, c0, c0 + 1]
                          if 0 <= r < self.board_size and 0 <= c < self.board_size and
                          self.board[r, c] != GameState.FLAG and (r, c) != (r0, c0)]
            # Enqueue those neighbours which are empty and we haven't already visited.
            to_visit += [sq for sq in neighbours
                         if self.board[sq] == GameState.EMPTY and sq not in connected_component]
            connected_component = connected_component.union(set(neighbours))
        return connected_component

    @staticmethod
    def num_total_flags(board_size: int):
        half_size = board_size // 2
        total_flags = half_size ** 2 if half_size % 2 == 1 else (half_size - 1) ** 2
        return total_flags

    @staticmethod
    def create_new_game(board_size: int) -> "GameState":
        total_flags = GameState.num_total_flags(board_size)

        flags = set()
        while len(flags) < total_flags:
            row = random.randint(0, board_size - 1)  # includes endpoints
            col = random.randint(0, board_size - 1)
            flags.add((row, col))

        def board_value_at_square(r, c):
            if (r, c) in flags:
                return GameState.FLAG
            # Return number of adjacent squares which are flags
            return sum([(i, j) in flags for i in [r - 1, r, r + 1] for j in [c - 1, c, c + 1] if (i, j) != (r, c)])

        board = utils.create_2d_tensor_from_func(rows=board_size, cols=board_size, fn=board_value_at_square,
                                                 dtype=torch.int)
        visible = torch.full((board_size, board_size), False, dtype=torch.bool)
        return GameState(board=board, visible=visible, is_blues_turn=True,
                         blue_flags_captured=set(), red_flags_captured=set())

    @staticmethod
    def board_as_string(board: torch.IntTensor):
        board_size = board.shape[0]
        s = ''
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == GameState.FLAG:
                    c = Fore.YELLOW
                    s += f'{c}*{Style.RESET_ALL}'
                elif board[row, col] == GameState.EMPTY:
                    s += '-'
                else:
                    s += str(board[row, col].item())
                s += ' '
            s += '\r\n'
        return s

    def __str__(self):
        s = ''
        for row in range(self.board_size):
            for col in range(self.board_size):
                if (row, col) == self.last_move:
                    s += Back.YELLOW
                if not self.visible[row, col]:
                    s += '.'
                elif self.board[row, col] == GameState.FLAG:
                    c = Fore.BLUE if (row, col) in self.blue_flags_captured else Fore.RED
                    # if (row, col) == self.last_move:
                    #     c = Fore.LIGHTBLUE_EX if (row, col) in self.blue_flags_captured else Fore.LIGHTRED_EX
                    s += f'{c}*{Style.RESET_ALL}'
                elif self.board[row, col] == GameState.EMPTY:
                    s += '-'
                else:
                    s += str(self.board[row, col].item())
                if (row, col) == self.last_move:
                    s += f'{Style.RESET_ALL}'
                s += ' '
            s += '\r\n'
        return s


if __name__ == '__main__':
    gs = GameState.create_new_game(5)
    print(gs)
    print(gs.make_move(0, 0))