
from gamestate import GameState


class GameManager:
    """
    Handles games of human vs player agent.
    """
    def __init__(self, human_first: bool, policy, board_size):
        self.human_is_blue = human_first
        self.policy = policy
        self.gamestate = GameState.create_new_game(board_size)

    def play_loop(self):
        while not self.gamestate.is_game_over():
            print(self.gamestate)
            humans_turn = (self.gamestate.is_blues_turn and self.human_is_blue) or \
                          (not self.gamestate.is_blues_turn and not self.human_is_blue)
            if humans_turn:
                move = self.get_move_from_console()
            else:
                move = self.policy.get_move(self.gamestate)
            self.gamestate.make_move(*move)
        print('Game over.')

    def get_move_from_console(self) -> "Move":
        move = None
        while not move or not self.gamestate.is_move_valid(move):
            move = self._parse_move_string(input('Please enter a valid move:'))
        return move

    def _parse_move_string(self, move_str):
        """

        :param move_str: two character string, characters specify row then column in hexadecimal.
        :return: (row, col) representing the move.
        """
        if len(move_str) != 2:
            return None
        try:
            row = int(move_str[0], self.gamestate.board_size)
            col = int(move_str[1], self.gamestate.board_size)
            return row, col
        except ValueError:
            return None

    def play_policy_move(self):
        policy_move = self.policy.get_move(self.gamestate)
        self.gamestate.make_move(*policy_move)


