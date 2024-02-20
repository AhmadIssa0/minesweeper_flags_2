

from gamestate import GameState
import torch
import random
import torch.nn.functional as F


def randomly_place_flags(board: torch.IntTensor, visible: torch.BoolTensor, num_flags_to_place: torch.Tensor) -> None:
    """Randomly places flags in invisible locations.

    :param board: [B, H, W]
    :param num_flags_to_place: [B]
    """
    board_size = board.shape[-1]
    assert ((~visible).sum(dim=(-2, -1)) >= num_flags_to_place).all(), 'Not enough empty squares to place flags.'

    # Random select appropriate number of squares to place flags from the available squares
    available = (~visible & (board != GameState.FLAG)).flatten(-2, -1).float()  # [B, board_size**2]
    probs = torch.rand_like(available) * available
    sort_values = torch.sort(probs, descending=True, dim=-1).values
    kth_values = sort_values[torch.arange(0, len(num_flags_to_place)), num_flags_to_place - 1]
    new_flags_mask = (probs >= kth_values.unsqueeze(1)).view(-1, board_size, board_size)
    board[new_flags_mask] = GameState.FLAG


def energy(flag_board: torch.IntTensor, original_board: torch.IntTensor, visible: torch.BoolTensor) -> torch.Tensor:
    """

    :param flag_board: the board which agrees with the original board in visible locations, and has flags placed elsewhere,
        but numbers induced by flags may be wrong in non-visible areas.
    :param visible:
    :return:
    """
    induced_board = number_of_adjacent_flags(flag_board == GameState.FLAG)
    difference = (induced_board - original_board).abs()
    filter = (visible & (original_board != GameState.FLAG)).float()
    return (difference * filter).sum(dim=(-1, -2))


def random_true_from_mask(mask):
    """Returns a new mask with a single true in a location where mask has a true. Note mask must have a true!"""
    board_size = mask.shape[-1]
    indices = torch.argmax((torch.rand_like(mask.float()) * mask.float()).flatten(-2, -1), dim=-1)
    return F.one_hot(indices, num_classes=board_size**2).view(-1, board_size, board_size).bool()


def proposal_transition(flag_board: torch.IntTensor, visible: torch.BoolTensor):
    # vectorized!
    available_mask = ~((flag_board == GameState.FLAG) | visible)  # [B, H, W]
    if not (available_mask.sum(dim=(-1, -2)) > 0).all():
        # All squares are occupied, no empty squares!
        return flag_board.clone()

    occupied_mask = ((flag_board == GameState.FLAG) & ~visible.unsqueeze(0))
    old_flag_mask = random_true_from_mask(occupied_mask)
    new_flag_mask = random_true_from_mask(available_mask)

    new_flag_board = flag_board.detach().clone()
    new_flag_board[old_flag_mask] = GameState.EMPTY
    new_flag_board[new_flag_mask] = GameState.FLAG

    return new_flag_board


def number_of_adjacent_flags(flag_mask):
    """

    :param flag_mask: [B, H, W]
    :return:
    """
    # Assuming mask is a 2D boolean tensor (mask.shape = [height, width])

    # Add batch and channel dimensions to mask
    mask_unsqueezed = flag_mask.unsqueeze(1).float()  # Convert to float for conv2d

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


def accept_proposal(old_flags, new_flags, orig_board, visible, temperature) -> torch.Tensor:
    old_energy = energy(old_flags, orig_board, visible)  # shape [B]
    new_energy = energy(new_flags, orig_board, visible)

    return torch.rand_like(old_energy) <= torch.exp(-1.0 / temperature * (new_energy - old_energy))


def construct_starting_points(board, visible):
    """Generates a random board state satisfying all visible constraints from the current game state.

    :param board: [B, H, W]
    """
    # Keep all visible flags fixed
    fixed_flags = (board == GameState.FLAG) & visible
    board = torch.full_like(board, GameState.EMPTY)
    board[fixed_flags] = GameState.FLAG

    # Place remaining flags randomly
    board_size = board.shape[-1]
    total_flags = GameState.num_total_flags(board_size)
    num_variable_flags = total_flags - fixed_flags.sum(dim=(-2, -1))
    randomly_place_flags(board, visible, num_variable_flags)

    return board


def mcmc_update(traj, board, visible, temperature):
    prop_traj = proposal_transition(traj, visible)
    return torch.where(accept_proposal(traj, prop_traj, board, visible, temperature=temperature).view(-1, 1, 1),
                       prop_traj, traj)


def parallel_tempering(board, visible, temperatures, num_steps, device):
    """Does B*T parallel MCMC chains.

    :param board: [B, H, W]
    :param visible: [B, H, W]
    :param temperatures: [T]
    :param num_steps:
    :param device:
    :return:
    """
    num_boards = len(board)
    num_temps = len(temperatures)
    board_size = board.shape[-1]
    board = board.repeat_interleave(len(temperatures), dim=0)
    visible = visible.repeat_interleave(len(temperatures), dim=0)
    trajectories = construct_starting_points(board, visible).to(device=device)
    temperatures = torch.as_tensor(temperatures, device=device)

    for step_num in range(num_steps):
        trajectories = mcmc_update(trajectories, board, visible, temperatures.repeat_interleave(num_boards))
        energies = energy(trajectories, board, visible).view(num_boards, num_temps)

        i = torch.randint(0, num_temps - 2, size=(num_boards,))
        index1 = F.one_hot(i, num_classes=num_temps).bool()
        index2 = F.one_hot(i + 1, num_classes=num_temps).bool()
        e1 = energies[index1]
        e2 = energies[index2]

        temps1 = temperatures.unsqueeze(0).expand(num_boards, -1)[index1]
        temps2 = temperatures.unsqueeze(0).expand(num_boards, -1)[index2]
        acceptance_thres = -1.0 * (e2 / temps1 + e1 / temps2 - e1 / temps1 - e2 / temps2)
        swapped_trajs = trajectories.clone().view(num_boards, num_temps, board_size, board_size)  # (num_boards, num_temps, H, W)
        swapped_trajs[index1], swapped_trajs[index2] = swapped_trajs[index2].clone(), swapped_trajs[index1].clone()
        cond = (torch.rand_like(acceptance_thres).log() <= acceptance_thres)
        cond = cond.view(-1, 1, 1, 1).repeat(1, num_temps, board_size, board_size)

        trajectories = torch.where(cond,
                                   swapped_trajs,
                                   trajectories.view(num_boards, num_temps, board_size, board_size)
                                   )
        trajectories = trajectories.view(-1, board_size, board_size)
    energies = energy(trajectories, board, visible).view(num_boards, num_temps)
    return trajectories.view(num_boards, num_temps, board_size, board_size), energies


def sample_hidden_boards(gs, device):
    temps = [0.001, 0.01, 0.01, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 100.0,
             1000.0, 10000.0]
    trajectories, energies = parallel_tempering(
        gs.board.to(device=device).unsqueeze(0).expand(500, -1, -1),
        gs.visible.to(device=device).unsqueeze(0).expand(500, -1, -1),
        temperatures=temps,
        num_steps=500,
        device=device
    )
    # energies has shape [500, len(temps)]
    # trajectories has shape [500, len(temps), board_size, board_size]
    solution_flags = trajectories.flatten(0, 1)[energies.flatten() == 0]
    solution_flag_masks = solution_flags == GameState.FLAG
    boards = number_of_adjacent_flags(solution_flag_masks)
    boards[solution_flag_masks] = GameState.FLAG
    boards = boards.int()
    if not ((gs.board & gs.visible).unsqueeze(0).to(device=device) == (boards & gs.visible.to(device=device))).all():
        # Ensure everything is on the same device and in the correct shape
        gs_board_visible = (gs.board & gs.visible).unsqueeze(0).to(
            device=device)  # Shape [1, 16, 16] to match batch dimension
        boards_visible = boards & gs.visible.to(device=device)  # Shape [batch_size, 16, 16]

        # Find where they differ
        differences = gs_board_visible != boards_visible  # This will be a boolean tensor of shape [batch_size, 16, 16]

        # Get the indices where they differ
        diff_indices = differences.nonzero()

        # Extract the unique batch indices where differences were found
        unique_batch_index = diff_indices[:, 0].unique()[0].item()
        print(GameState.board_as_string(boards[unique_batch_index]))
        print(energy(solution_flags[unique_batch_index].unsqueeze(0),
                     gs.board.to(device=device).unsqueeze(0), gs.visible.to(device=device).unsqueeze(0)))
        exit(0)
    return boards

def run():
    from policy import RandomPolicy
    device = 'cuda'

    for k in range(100):
        print('Board:', k)
        gs = GameState.create_new_game(16)

        for i in range(15):
            gs = gs.make_move(*RandomPolicy.get_move(gs))

        print(gs)

        states = sample_hidden_boards(gs, device)
        print('Number of sampled states:', len(states))
        print(GameState.board_as_string(states[0]))


        # temps = [0.001, 0.01, 0.01, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 100.0, 1000.0, 10000.0]
        # # temps = [0.001, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 100.0, 1000.0, 10000.0]
        # trajectories, energies = parallel_tempering(gs.board.to(device=device).unsqueeze(0).expand(500, -1, -1),
        #                                             gs.visible.to(device=device).unsqueeze(0).expand(500, -1, -1),
        #                                             # temperatures=[0.01, 0.3, 0.4, 0.7, 1.2, 2.0, 4.0, 1000.0],
        #                                             temperatures=temps,
        #                                             num_steps=2000, device=device)
        # print('Number of solutions:', (energies == 0).sum())


if __name__ == '__main__':
    random.seed(10)
    torch.manual_seed(0)

    with torch.no_grad():
        run()
    # import cProfile
    # cProfile.run('run()')
