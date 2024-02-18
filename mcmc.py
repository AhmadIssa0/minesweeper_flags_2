

from gamestate import GameState
import torch
import random
from policy import RandomPolicy
import utils
import math
import torch.nn.functional as F

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
    # board_size = flag_board.shape[0]
    # has_flag = (flag_board == GameState.FLAG)
    # flags = has_flag.nonzero().tolist()
    #
    # def board_value_at_square(r, c):
    #     if [r, c] in flags:
    #         return GameState.FLAG
    #     # Return number of adjacent squares which are flags
    #     return sum([[i, j] in flags for i in [r - 1, r, r + 1] for j in [c - 1, c, c + 1] if (i, j) != (r, c)])
    #
    # induced_board = utils.create_2d_tensor_from_func(rows=board_size, cols=board_size, fn=board_value_at_square,
    #                                                  dtype=torch.int, device=flag_board.device)

    induced_board = number_of_adjacent_flags(flag_board == GameState.FLAG)
    # print(flag_board)
    # print(induced_board)
    # exit(0)
    # print('Cost:\n', (induced_board - original_board).abs() * visible.float())
    return (induced_board - original_board).abs()[visible & (original_board != GameState.FLAG)].sum().item()


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
    return new_flag_board, 1.0, 1.0


def number_of_adjacent_flags(flag_mask):
    """

    :param flag_mask: [B, H, W]
    :return:
    """
    # Assuming mask is a 2D boolean tensor (mask.shape = [height, width])

    # Add batch and channel dimensions to mask
    mask_unsqueezed = flag_mask.unsqueeze(0).unsqueeze(0).float()  # Convert to float for conv2d

    # Define a 3x3 kernel with all ones. The kernel size needs to match the number of neighbors including the cell itself
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float, device=flag_mask.device)

    # Apply convolution. Padding is set to 1 to include edges, stride is 1 to move one step at a time
    # The convolution result will indicate the count of True values in the 3x3 area centered around each cell
    conv_result = F.conv2d(mask_unsqueezed, kernel, padding=1)

    return conv_result.squeeze()


def adjacent_squares_mask(mask):
    # Assuming mask is a 2D boolean tensor (mask.shape = [height, width])

    # Add batch and channel dimensions to mask
    mask_unsqueezed = mask.unsqueeze(0).unsqueeze(0).float()  # Convert to float for conv2d

    # Define a 3x3 kernel with all ones. The kernel size needs to match the number of neighbors including the cell itself
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float)

    # Apply convolution. Padding is set to 1 to include edges, stride is 1 to move one step at a time
    # The convolution result will indicate the count of True values in the 3x3 area centered around each cell
    conv_result = F.conv2d(mask_unsqueezed, kernel, padding=1)

    # Check if any of the adjacent entries is True (conv_result > 0) and convert back to boolean tensor
    expanded_mask = conv_result > 0

    # Remove added batch and channel dimensions to match the original mask's shape
    expanded_mask = expanded_mask.squeeze(0).squeeze(0)

    return expanded_mask.bool()


def probability_of_transition(
        flag_board: torch.IntTensor,
        visible: torch.BoolTensor,
        old_flag,  # coords
        new_flag,  # coords
        weight_adj,
        weight_not_adj,
        weight_adj_empty,
        weight_not_adj_empty,
):
    available_mask = ~((flag_board == GameState.FLAG) | visible)
    assert available_mask.sum() > 0

    adj_visible = adjacent_squares_mask(visible & (flag_board != GameState.FLAG))

    empty_adj_to_visible = available_mask & adj_visible
    empty_not_adj_to_visible = available_mask & ~adj_visible

    movable_flags_mask = ((flag_board == GameState.FLAG) & ~visible)
    flags_adj_vis_mask = movable_flags_mask & adj_visible
    flags_not_adj_mask = movable_flags_mask & ~adj_visible

    # Pick a flag to move
    if flags_not_adj_mask.sum() == 0:
        weight_not_adj = 0.0
    if flags_adj_vis_mask.sum() == 0:
        weight_adj = 0.0
    prob_move_adj = weight_adj / (weight_adj + weight_not_adj)

    if flags_adj_vis_mask[old_flag[0], old_flag[1]]:
        prob_of_selection = prob_move_adj / flags_adj_vis_mask.sum().item()
    else:
        prob_of_selection = (1 - prob_move_adj) / flags_not_adj_mask.sum().item()

    # Pick a square to move to
    if empty_not_adj_to_visible.sum() == 0:
        weight_not_adj_empty = 0.0
    if empty_adj_to_visible.sum() == 0:
        weight_adj_empty = 0.0
    prob_empty_adj = weight_adj_empty / (weight_adj_empty + weight_not_adj_empty)

    if empty_adj_to_visible[new_flag[0], new_flag[1]]:
        prob_of_landing = prob_empty_adj / empty_adj_to_visible.sum().item()
    else:
        prob_of_landing = (1.0 - prob_empty_adj) / empty_not_adj_to_visible.sum().item()

    return prob_of_selection * prob_of_landing


def proposal_transition_(
        flag_board: torch.IntTensor,
        visible: torch.BoolTensor,
        weight_from_adj=2.0,
        weight_from_not_adj=1.0,
        weight_to_adj=1.0,
        weight_to_not_adj=1.0,
):
    available_mask = ~((flag_board == GameState.FLAG) | visible)
    assert available_mask.sum() > 0

    adj_visible = adjacent_squares_mask(visible & (flag_board != GameState.FLAG))

    empty_adj_to_visible = available_mask & adj_visible
    empty_not_adj_to_visible = available_mask & ~adj_visible

    movable_flags_mask = ((flag_board == GameState.FLAG) & ~visible)
    flags_adj_vis_mask = movable_flags_mask & adj_visible
    flags_not_adj_mask = movable_flags_mask & ~adj_visible

    def random_coord_from_mask(mask: torch.BoolTensor):
        # index in flattened
        index_of_random = (torch.rand_like(mask.float()) * mask.float()).argmax().item()
        width = mask.shape[1]
        return [index_of_random // width, index_of_random % width]

    weight_adj, weight_not_adj = weight_from_adj, weight_from_not_adj
    # Pick a flag to move
    if flags_not_adj_mask.sum() == 0:
        weight_not_adj = 0.0
    if flags_adj_vis_mask.sum() == 0:
        weight_adj = 0.0
    prob_move_adj = weight_adj / (weight_adj + weight_not_adj)

    if random.random() <= prob_move_adj:
        old_flag = random_coord_from_mask(flags_adj_vis_mask)
        prob_of_selection = prob_move_adj / flags_adj_vis_mask.sum().item()
    else:
        old_flag = random_coord_from_mask(flags_not_adj_mask)
        prob_of_selection = (1 - prob_move_adj) / flags_not_adj_mask.sum().item()

    # Pick a square to move to
    weight_adj_empty, weight_not_adj_empty = weight_to_adj, weight_to_not_adj
    if empty_not_adj_to_visible.sum() == 0:
        weight_not_adj_empty = 0.0
    if empty_adj_to_visible.sum() == 0:
        weight_adj_empty = 0.0
    prob_empty_adj = weight_adj_empty / (weight_adj_empty + weight_not_adj_empty)

    if random.random() <= prob_empty_adj:
        new_flag = random_coord_from_mask(empty_adj_to_visible)
        prob_of_landing = prob_empty_adj / empty_adj_to_visible.sum().item()
    else:
        new_flag = random_coord_from_mask(empty_not_adj_to_visible)
        prob_of_landing = (1.0 - prob_empty_adj) / empty_not_adj_to_visible.sum().item()

    prob_of_transition = prob_of_selection * prob_of_landing

    new_flag_board = flag_board.detach().clone()
    new_flag_board[old_flag[0], old_flag[1]] = GameState.EMPTY
    new_flag_board[new_flag[0], new_flag[1]] = GameState.FLAG

    prob_of_backward_transition = probability_of_transition(
        new_flag_board,
        visible,
        new_flag,
        old_flag,
        weight_from_adj,
        weight_from_not_adj,
        weight_to_adj,
        weight_to_not_adj,
    )
    # print('Num flags adj:', flags_adj_vis_mask.sum().item())
    # print('Num flags not adj:', flags_not_adj_mask.sum().item())
    # print('Prob of transition:', prob_of_transition, old_flag, new_flag)
    # print('Prob of selection:', prob_of_selection)
    # print('Prob of landing:', prob_of_landing)
    # print('Prob move adj:', prob_move_adj, 'prob move non-adj:', 1-prob_move_adj)
    # print('Before:')
    # print(GameState.board_as_string(flag_board))
    # print('After:')
    # print(GameState.board_as_string(new_flag_board))
    # exit(0)
    return new_flag_board, prob_of_transition, prob_of_backward_transition


def accept_proposal(old_flags, new_flags, orig_board, visible, temperature, flow_bias_correction=1.0) -> bool:
    old_energy = energy(old_flags, orig_board, visible)
    new_energy = energy(new_flags, orig_board, visible)
    if temperature > 0:
        # print(flow_bias_correction)
        return random.random() <= math.exp(-1.0 / temperature * (new_energy - old_energy)) * flow_bias_correction
    else:
        return new_energy <= old_energy


def generate_random_board(board, visible):
    """Generates a random board state satisfying all visible constraints from the current game state."""
    # Keep all visible flags fixed
    fixed_flags = (board == GameState.FLAG) & visible
    board = torch.full_like(board, GameState.EMPTY)
    board[fixed_flags] = GameState.FLAG

    # Place remaining flags randomly
    board_size = board.shape[0]
    total_flags = GameState.num_total_flags(board_size)
    num_variable_flags = total_flags - fixed_flags.sum().item()
    randomly_place_flags(board, visible, num_variable_flags)

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


def mcmc_update(traj, board, visible, temperature):
    prop_traj, prob_forward, prob_backward = proposal_transition(traj, visible)
    if accept_proposal(traj, prop_traj, board, visible, temperature=temperature, flow_bias_correction=prob_backward / prob_forward):
        return prop_traj
    else:
        return traj


def parallel_tempering(board, visible, temperatures, num_steps, device):
    trajectories = [generate_random_board(board, visible).to(device=device) for _ in temperatures]

    for _ in range(num_steps):
        trajectories = [mcmc_update(traj, board, visible, temp) for temp, traj in zip(temperatures, trajectories)]
        min_energy = min([energy(traj, board, visible) for traj in trajectories])
        if min_energy == 0:
            print('Energies:', [energy(traj, board, visible) for traj in trajectories])
            for traj in trajectories:
                if energy(traj, board, visible) == 0:
                    print(GameState.board_as_string(traj))
            return True
        i = random.randint(0, len(trajectories) - 2)
        e1 = energy(trajectories[i], board, visible)
        e2 = energy(trajectories[i+1], board, visible)
        acceptance_thres = -1.0 * (e2 / temperatures[i] + e1 / temperatures[i+1] - e1 / temperatures[i] - e2 / temperatures[i+1])
        if math.log(random.random() + 1e-15) <= acceptance_thres:
            # print('swapping:', i, i+1)
            trajectories[i], trajectories[i+1] = trajectories[i+1], trajectories[i]
    print('Energies:', [energy(traj, board, visible) for traj in trajectories])
    for traj in trajectories:
        print(GameState.board_as_string(traj))
    return False


def run():
    random.seed(10)
    device = 'cpu'
    for k in range(5):
        print('Board:', k)
        gs = GameState.create_new_game(10)

        for i in range(10):
            gs = gs.make_move(*RandomPolicy.get_move(gs))

        print(gs)
        # print(gs.board_as_string())
        original_board = gs.board

        if not parallel_tempering(
                gs.board.to(device=device), gs.visible.to(device=device),
                temperatures=[0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 100.0, 1000.0, 10000.0],
                num_steps=5000, device=device):
            print('Couldnt solve!')
            # fixed_flags = (gs.board == GameState.FLAG) & gs.visible
            # print(fixed_flags)
            exit(0)


if __name__ == '__main__':
    import cProfile
    cProfile.run('run()')
