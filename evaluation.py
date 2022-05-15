
from gamestate import GameState


def score_match(board_size, policy1, policy2, num_games):
    """
    Returns number of wins by `policy1` in a `num_games` game match.
    :param board_size:
    :param policy1:
    :param policy2:
    :param num_games:
    :return:
    """
    score = 0
    avg_flag_lead = 0
    policy1_is_blue = True
    for games_played in range(num_games):
        gs = GameState.create_new_game(board_size)
        while not gs.is_game_over():
            if gs.is_blues_turn == policy1_is_blue:
                gs = gs.make_move(*policy1.get_move(gs))
            else:
                gs = gs.make_move(*policy2.get_move(gs))
        blue_won = len(gs.blue_flags_captured) > len(gs.red_flags_captured)
        if policy1_is_blue:
            avg_flag_lead += len(gs.blue_flags_captured) - len(gs.red_flags_captured)
        else:
            avg_flag_lead += len(gs.red_flags_captured) - len(gs.blue_flags_captured)

        if (blue_won and policy1_is_blue) or (not blue_won and not policy1_is_blue):
            score += 1
        policy1_is_blue = not policy1_is_blue
    avg_flag_lead = avg_flag_lead / num_games
    return score, avg_flag_lead
