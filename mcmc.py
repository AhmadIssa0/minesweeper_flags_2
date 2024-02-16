

from gamestate import GameState
import torch
import random
from policy import RandomPolicy
import utils
import math

def randomly_place_flags(board: torch.IntTensor, visible: torch.BoolTensor, num_flags_to_place):
    """Randomly places flags in invisible locations."""
    num_flags_placed = 0
    board_size = board.shape[0]
    assert (~visible).sum() >= num_flags_to_place
    while num_flags_placed < num_flags_to_place:
        row = random.randint(0, board_size - 1)  # includes endpoints
        col = random.randint(0, board_size - 1)
        if visible[row, col]:
            continue
        if board[row, col] != GameState.FLAG:
            num_flags_placed += 1
        board[row, col] = GameState.FLAG


def energy(flag_board: torch.IntTensor, original_board: torch.IntTensor, visible: torch.BoolTensor) -> int:
    """

    :param flag_board: the board which agrees with the original board in visible locations, and has flags placed elsewhere,
        but numbers induced by flags may be wrong in non-visible areas.
    :param visible:
    :return:
    """
    board_size = flag_board.shape[0]
    has_flag = (flag_board == GameState.FLAG)
    flags = has_flag.nonzero().tolist()

    def board_value_at_square(r, c):
        if [r, c] in flags:
            return GameState.FLAG
        # Return number of adjacent squares which are flags
        return sum([[i, j] in flags for i in [r - 1, r, r + 1] for j in [c - 1, c, c + 1] if (i, j) != (r, c)])

    induced_board = utils.create_2d_tensor_from_func(rows=board_size, cols=board_size, fn=board_value_at_square,
                                                     dtype=torch.int)

    # print('Cost:\n', (induced_board - original_board).abs() * visible.float())
    return (induced_board - original_board).abs()[visible].sum().item()


def proposal_transition(flag_board: torch.IntTensor, visible: torch.BoolTensor):
    available_mask = ~((flag_board == GameState.FLAG) | visible)
    assert available_mask.sum() > 0

    available = available_mask.nonzero().tolist()
    new_flag_coords = available[random.randint(0, len(available) - 1)]
    curr_flags = ((flag_board == GameState.FLAG) & ~visible).nonzero().tolist()
    old_flag_coords = curr_flags[random.randint(0, len(curr_flags) - 1)]

    new_flag_board = flag_board.detach().clone()
    new_flag_board[old_flag_coords[0], old_flag_coords[1]] = GameState.EMPTY
    new_flag_board[new_flag_coords[0], new_flag_coords[1]] = GameState.FLAG
    return new_flag_board


def accept_proposal(old_flags, new_flags, orig_board, visible, temperature):
    old_energy = energy(old_flags, orig_board, visible)
    new_energy = energy(new_flags, orig_board, visible)
    if temperature > 0:
        return random.random() <= math.exp(-1.0 / temperature * (new_energy - old_energy))
    else:
        return new_energy <= old_energy


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
    randomly_place_flags(board, gs.visible, num_variable_flags)

    return board


def mcmc(gs):
    traj = generate_random_board(gs)

    for temperature_scaled in range(10, -1, -1):
        temperature = temperature_scaled / 50.0
        print('Temperature:', temperature)
        acceptances = 0
        for i in range(1000):
            prop_traj = proposal_transition(traj, gs.visible)
            if accept_proposal(traj, prop_traj, gs.board, gs.visible, temperature=temperature):
                traj = prop_traj
                acceptances += 1
                if energy(traj, gs.board, gs.visible) == 0:
                    print('Found energy 0')
                    print(GameState.board_as_string(traj))
                    return True
        # print(GameState.board_as_string(traj))
        # print('Energy:', energy(traj, original_board, gs.visible))
        # print('Acceptances:', acceptances)
    return False


def mcmc_update(traj, gs, temperature):
    prop_traj = proposal_transition(traj, gs.visible)
    if accept_proposal(traj, prop_traj, gs.board, gs.visible, temperature=temperature):
        return prop_traj
    else:
        return traj


def parallel_tempering(gs, temperatures, num_steps):
    trajectories = [generate_random_board(gs) for _ in temperatures]

    for _ in range(num_steps):
        trajectories = [mcmc_update(traj, gs, temp) for temp, traj in zip(temperatures, trajectories)]
        min_energy = min([energy(traj, gs.board, gs.visible) for traj in trajectories])
        if min_energy == 0:
            print('Energies:', [energy(traj, gs.board, gs.visible) for traj in trajectories])
            return True
        i = random.randint(0, len(trajectories) - 2)
        e1 = energy(trajectories[i], gs.board, gs.visible)
        e2 = energy(trajectories[i+1], gs.board, gs.visible)
        acceptance_thres = -1.0 * (e2 / temperatures[i] + e1 / temperatures[i+1] - e1 / temperatures[i] - e2 / temperatures[i+1])
        if math.log(random.random() + 1e-15) <= acceptance_thres:
            # print('swapping:', i, i+1)
            trajectories[i], trajectories[i+1] = trajectories[i+1], trajectories[i]
    print('Energies:', [energy(traj, gs.board, gs.visible) for traj in trajectories])
    for traj in trajectories:
        print(GameState.board_as_string(traj))
    return False


if __name__ == '__main__':

    for k in range(100):
        print('Board:', k)
        gs = GameState.create_new_game(10)

        for i in range(10):
            gs = gs.make_move(*RandomPolicy.get_move(gs))

        print(gs)
        # print(gs.board_as_string())
        original_board = gs.board

        if not parallel_tempering(gs, temperatures=[0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 100.0, 1000.0], num_steps=5000):
            print('Couldnt solve!')
            # fixed_flags = (gs.board == GameState.FLAG) & gs.visible
            # print(fixed_flags)
            exit(0)
