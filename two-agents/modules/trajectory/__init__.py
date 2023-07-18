import numpy as np
import math

# Last position remains the inital local position
# One point is randomly dropped
# First position is the current local position
def drop_random_points(
    current_trajectory: np.ndarray,
    agent_local_poistion: np.ndarray,
    n_steps: int,
    n_trajectory_points: int,
    initial_position: np.ndarray,
) -> np.ndarray:

    updated_trajectory = current_trajectory.copy()

    if n_steps >= n_trajectory_points:
        trajectory_full = True
    else:
        trajectory_full = False

    if not trajectory_full:
        updated_trajectory[0] = agent_local_poistion
        for i in range(n_trajectory_points - 1):
            updated_trajectory[i + 1] = current_trajectory[i]

    elif trajectory_full:
        dropped_position = np.random.random_integers(
            low=1, high=n_trajectory_points - 2
        )
        aux = np.delete(updated_trajectory, dropped_position, axis=0)
        updated_trajectory[0] = agent_local_poistion
        updated_trajectory[-1] = initial_position
        for i in range(n_trajectory_points - 2):
            updated_trajectory[i + 1] = aux[i]

    return updated_trajectory.astype(int)


def fifo(
    current_trajectory: np.ndarray,
    agent_local_poistion: np.ndarray,
    n_steps: int,
    n_trajectory_points: int,
) -> np.ndarray:

    updated_trajectory = current_trajectory.copy()

    updated_trajectory[0] = agent_local_poistion
    for i in range(n_trajectory_points - 1):
        updated_trajectory[i + 1] = current_trajectory[i]

    return updated_trajectory.astype(int)


def merge_last_positions(
    current_trajectory: np.ndarray,
    agent_local_poistion: np.ndarray,
    n_steps: int,
    n_trajectory_points: int,
    initial_position: np.ndarray,
    merge_iteration: int,
) -> np.ndarray:

    updated_trajectory = current_trajectory.copy()

    num_mergeble_points = int((n_trajectory_points - 2) * 0.8)
    first_twenty_percent = (n_trajectory_points - 2) - num_mergeble_points
    # print(num_mergeble_points)
    if n_steps >= n_trajectory_points:
        trajectory_full = True
    else:
        trajectory_full = False

    if not trajectory_full:
        updated_trajectory[0] = agent_local_poistion
        for i in range(n_trajectory_points - 1):
            updated_trajectory[i + 1] = current_trajectory[i]

    elif trajectory_full:

        merge_iteration = math.floor(int(n_steps % n_trajectory_points))
        # print(f"Merge iteration {n_steps}%{n_trajectory_points}={merge_iteration}")

        # Last 80% - merge
        aux = current_trajectory[
            first_twenty_percent + 1 : n_trajectory_points - 1
        ].copy()

        # print(current_trajectory)

        # print(first_twenty_percent)
        # print(n_trajectory_points - 2)
        # print(aux)
        # print(f"num_mergeble_points {num_mergeble_points}")
        # print(f"merge_iteration {merge_iteration}")
        # # print(aux)
        # print(f"ranges N-1 = {num_mergeble_points - merge_iteration - 2}")
        # print(f"ranges N = {num_mergeble_points - merge_iteration -1}")
        # mean = (
        #     current_trajectory[n_trajectory_points - 2]
        #     + current_trajectory[n_trajectory_points - 3]
        # ) / 2

        mean = (
            aux[num_mergeble_points - merge_iteration - 2]
            + aux[num_mergeble_points - merge_iteration - 1]
        ) / 2

        # print(f"mean = {mean}")
        # print(f"mean pos = {num_mergeble_points - merge_iteration - 1}")

        aux[num_mergeble_points - merge_iteration - 1] = mean

        for i in range(num_mergeble_points - merge_iteration - 3, 0, -1):
            # print(f"Shifting from {i+3} to {i+1+3}")
            aux[i + 1] = aux[i]

        # print(aux)

        # print(first_twenty_percent)
        # print(n_trajectory_points - 2)
        updated_trajectory[
            first_twenty_percent + 1 : n_trajectory_points - 1
        ] = aux.copy()

        # First 20% - keep all points
        for i in range(first_twenty_percent + 2):
            updated_trajectory[i + 1] = current_trajectory[i]

        updated_trajectory[0] = agent_local_poistion
        updated_trajectory[-1] = initial_position
        # print(updated_trajectory)

        # if merge_iteration == 5:
        #     exit()

    return updated_trajectory.astype(float)
