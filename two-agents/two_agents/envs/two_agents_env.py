import gym
from gym import spaces
import pygame
import numpy as np
import random
from modules import agent_trajectory
import json


class TwoAgentsEnv(gym.Env):

    with open(f"parameters.json", "r") as parameters_file:
        parameters = json.load(parameters_file)

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": parameters["visualization"]["FPS"],
    }

    def __init__(self, render_mode=None, idx=None, num_robots=2):

        with open(f"parameters.json", "r") as parameters_file:
            parameters = json.load(parameters_file)

        # Shared variables
        self.initial_free_cells = 0
        self.robots = []
        self.global_steps = 0.0
        self.shared_current_position_map = None
        self.grid_matrix = None

        self.paint_lasers = True

        # Map variables
        self.map_file_path = parameters["map"]["map_file_path"]
        self.pix_square_size = parameters["map"][
            "pix_square_size"
        ]  # The size of a single grid square in pixels
        self.canvas = pygame.image.load(self.map_file_path)
        self.clean_canvas = pygame.image.load(self.map_file_path)
        self.window_width = self.canvas.get_width()
        self.window_height = self.canvas.get_height()
        self.n_cells_x = int(self.window_width / self.pix_square_size)
        self.n_cells_y = int(self.window_height / self.pix_square_size)
        self.idx = idx

        # Rewards
        self.max_penalty = parameters["reward"]["max_penalty"]
        self.max_reward = parameters["reward"]["max_reward"]
        self.penalty_per_old_region = parameters["reward"]["penalty_per_old_region"]
        self.reward_per_new_region = parameters["reward"]["reward_per_new_region"]

        # Done criteria
        self.max_exploration_rate = parameters["done_criteria"]["max_exploration_rate"]
        self.max_steps_to_done = parameters["done_criteria"]["max_steps"]

        # Trajectory info
        self.n_trajectory_points = parameters["agent"]["n_trajectory_points"]
        self.n_proximity_regions = parameters["agent"]["n_proximity_regions"]
        self.explored_radius = parameters["agent"]["explored_radius"]
        self.n_others_trajectory_points = parameters["agent"][
            "n_others_trajectory_points"
        ]

        # Visualization window
        self.draw_lasers = parameters["visualization"]["draw_lasers"]
        self.grid_lines = parameters["visualization"]["grid_lines"]
        self.draw_agent = parameters["visualization"]["draw_agent"]
        self.draw_trajectory = parameters["visualization"]["draw_trajectory"]

        self.explored_rate = 0.0  # goes to robot
        self.global_steps = 0.0  # goes to robot
        self.episode = 0
        self.num_robots = num_robots

        max_bound = self.n_cells_x
        if self.n_cells_y > self.n_cells_x:
            max_bound = self.n_cells_y

        robot_name = "robot0"
        robot_name1 = "robot1"
        self.observation_space = spaces.Dict(
            {
                robot_name: spaces.Dict(
                    {
                        "proximity_regions": spaces.Box(
                            low=1.0,
                            high=4.0,
                            shape=(self.n_proximity_regions,),
                            dtype=int,
                        ),
                        "trajectory": spaces.Box(
                            low=-max_bound - 1,
                            high=max_bound - 1,
                            shape=(self.n_trajectory_points, 2),
                            dtype=int,
                        ),
                        "other_robots_trajectories": spaces.Box(
                            low=-max_bound - 1,
                            high=max_bound - 1,
                            shape=(self.n_others_trajectory_points, 2),
                            dtype=int,
                        ),
                    }
                ),
                robot_name1: spaces.Dict(
                    {
                        "proximity_regions": spaces.Box(
                            low=1.0,
                            high=4.0,
                            shape=(self.n_proximity_regions,),
                            dtype=int,
                        ),
                        "trajectory": spaces.Box(
                            low=-max_bound - 1,
                            high=max_bound - 1,
                            shape=(self.n_trajectory_points, 2),
                            dtype=int,
                        ),
                        "other_robots_trajectories": spaces.Box(
                            low=-max_bound - 1,
                            high=max_bound - 1,
                            shape=(self.n_others_trajectory_points, 2),
                            dtype=int,
                        ),
                    }
                ),
            }
        )
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([-1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

        self.window = None
        self.clock = None

        self._init_map()

        # Here will create each robot and assign their exploration grid etc
        for i in range(self.num_robots):
            self.robots.append(self.Robots(i))

        self.exploration_grid_matrix = self.grid_matrix.copy()

        self.initial_free_cells = self.grid_matrix.size - np.count_nonzero(
            self.grid_matrix
        )

    class Robots:

        # Individual
        _initial_global_position = None
        _agent_global_position = None
        _agent_local_position = None
        _visited_x = None
        _visited_y = None
        _agent_trajectory = None
        _other_agents_trajectory = None
        exploration_delta = 0
        _lasers = None
        _initial_local_position = 0
        number_of_free_cells = 0

        def __init__(self, robotID) -> None:
            self.explored_rate = 0.0
            self.n_steps = 0.0
            self.n_collisions = False
            self.robotID = robotID

        def _get_agent_local_position(self):
            return self._agent_global_position - self._initial_global_position

        def reset(self):
            self._initial_global_position = None
            self._agent_global_position = None
            self._agent_local_position = None
            self._visited_x = None
            self._visited_y = None
            self._agent_trajectory = None
            self.exploration_delta = 0
            self._lasers = None
            self.number_of_free_cells = 0

        # Function to translates the environment’s state into an observation.
        def _get_obs(self):
            return {
                "proximity_regions": self._lasers,
                "trajectory": self._agent_trajectory,
                "other_robots_trajectories": self._other_agents_trajectory,
            }

        def _get_info(
            self,
        ):
            return {
                "explored_rate": self.explored_rate,
                "path_lenght": self.n_steps,
                "collision": self.n_collisions,
            }

        def _get_laser_measurements(self, shared_map_grid, n_cells_x, n_cells_y):
            current_x = self._agent_global_position[0]
            current_y = self._agent_global_position[1]

            north_laser = 1  # Safe
            east_laser = 1  # Safe
            west_laser = 1  # Safe
            south_laser = 1  # Safe

            # North
            # Check edges
            if current_y == 0:
                north_laser = 4  # Collision
            elif current_y == 1:
                north_laser = 3  # Risk - it is on the top edge
            elif current_y == 2:
                north_laser = 2  # Caution - it one cell away from the top edge
            # Check for sourrounding objects
            else:
                if shared_map_grid[current_x][current_y]:
                    north_laser = 4
                elif shared_map_grid[current_x][current_y - 1]:
                    north_laser = 3
                elif shared_map_grid[current_x][current_y - 2]:
                    north_laser = 2

            # South
            # Check edges
            if current_y == n_cells_y - 1:
                south_laser = 4  # Collision
            elif current_y == n_cells_y - 2:
                south_laser = 3  # Risk - it is on the bottom edge
            elif current_y == n_cells_y - 3:
                south_laser = 2  # Caution - it one cell away from the bottom edge
            # Check for sourrounding objects
            else:
                if shared_map_grid[current_x][current_y]:
                    south_laser = 4
                elif shared_map_grid[current_x][current_y + 1]:
                    south_laser = 3
                elif shared_map_grid[current_x][current_y + 2]:
                    south_laser = 2

            # East
            # Check edges
            if current_x == n_cells_x - 1:
                east_laser = 4  # Collision
            elif current_x == n_cells_x - 2:
                east_laser = 3  # Risk - it is on the east edge
            elif current_x == n_cells_x - 3:
                east_laser = 2  # Caution - it one cell away from the east edge
            # Check for sourrounding objects
            else:
                if shared_map_grid[current_x][current_y]:
                    east_laser = 4
                elif shared_map_grid[current_x + 1][current_y]:
                    east_laser = 3
                elif shared_map_grid[current_x + 2][current_y]:
                    east_laser = 2

            # West
            # Check edges
            if current_x == 0:
                west_laser = 4  # Collision
            elif current_x == 1:
                west_laser = 3  # Risk - it is on the west edge
            elif current_x == 2:
                west_laser = 2  # Caution - it one cell away from the west edge
            # Check for sourrounding objects
            else:
                if shared_map_grid[current_x][current_y]:
                    west_laser = 4
                elif shared_map_grid[current_x - 1][current_y]:
                    west_laser = 3
                elif shared_map_grid[current_x - 2][current_y]:
                    west_laser = 2

            self._lasers = np.array(
                [int(north_laser), int(south_laser), int(east_laser), int(west_laser)]
            )

            return self._lasers

        def _set_agent_local_init_position(self, _agent_global_position):
            self._initial_global_position = _agent_global_position

        def _get_agent_local_position(self):
            return self._agent_global_position - self._initial_global_position

        def get_line(self, start, end):
            # Setup initial conditions
            x1, y1 = start
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1

            # Determine how steep the line is
            is_steep = abs(dy) > abs(dx)

            # Rotate line
            if is_steep:
                x1, y1 = y1, x1
                x2, y2 = y2, x2

            # Swap start and end points if necessary and store swap state
            swapped = False
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = True

            # Recalculate differentials
            dx = x2 - x1
            dy = y2 - y1

            # Calculate error
            error = int(dx / 2.0)
            ystep = 1 if y1 < y2 else -1

            # Iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in range(x1, x2 + 1):
                coord = (y, x) if is_steep else (x, y)
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += ystep
                    error += dx

            # Reverse the list if the coordinates were swapped
            if swapped:
                points.reverse()
            return points

        def _define_explored_area(
            self, explored_radius, n_cells_x, n_cells_y, shared_exploration_grid_matrix
        ):

            # internal_exploration_grid_matrix = shared_exploration_grid_matrix.copy()

            list_of_points = []
            x_curr = self._agent_global_position[0]
            y_curr = self._agent_global_position[1]

            for i in range(2):
                if i == 0:
                    x = x_curr - explored_radius
                else:
                    x = x_curr + explored_radius + 1

                for y in range(y_curr - explored_radius, y_curr + explored_radius + 1):
                    list_of_points.append((x, y))

                if i == 0:
                    y = y_curr - explored_radius
                else:
                    y = y_curr + explored_radius + 1

                for x in range(x_curr - explored_radius, x_curr + explored_radius + 1):
                    list_of_points.append((x, y))

            # count = 0
            for point in list_of_points:
                line_is_get = self.get_line(self._agent_global_position, point)
                for cell in line_is_get:
                    if cell[0] >= 0 and cell[0] < n_cells_x:
                        if cell[1] >= 0 and cell[1] < n_cells_y:
                            # if any on the list is a wall
                            if shared_exploration_grid_matrix[cell[0]][cell[1]] == 2.0:
                                break
                            else:
                                shared_exploration_grid_matrix[cell[0]][cell[1]] = 1.0

        def init_robot_map(
            self,
            explored_radius,
            n_cells_x,
            n_cells_y,
            n_trajectory_points,
            n_others_trajectory_points,
            shared_exploration_grid_matrix,
            current_position,
        ):

            self._agent_global_position = current_position.copy()

            self._agent_local_position = self._get_agent_local_position()

            self._agent_trajectory = np.zeros(shape=(n_trajectory_points, 2), dtype=int)
            self._other_agents_trajectory = np.zeros(
                shape=(n_others_trajectory_points, 2), dtype=int
            )
            self._visited_x = [self._agent_local_position[0]]
            self._visited_y = [self._agent_local_position[1]]

            self._agent_trajectory[0] = self._agent_local_position

            self._define_explored_area(
                explored_radius, n_cells_x, n_cells_y, shared_exploration_grid_matrix
            )

        def move(self, direction, n_cells_x, n_cells_y):

            old_pos = self._agent_global_position

            self._agent_global_position[0] = np.clip(
                self._agent_global_position[0] + direction[0], 0, n_cells_x - 1
            )
            self._agent_global_position[1] = np.clip(
                self._agent_global_position[1] + direction[1], 0, n_cells_y - 1
            )

            self._agent_local_position = self._get_agent_local_position()

            return self._agent_global_position, old_pos

        def move_safe(self, action, direction, n_cells_x, n_cells_y):
            # Moves avoiding collision
            old_pos = self._agent_global_position

            if self._lasers[int(action)] < 3:
                self._agent_global_position[0] = np.clip(
                    self._agent_global_position[0] + direction[0], 0, n_cells_x - 1
                )
                self._agent_global_position[1] = np.clip(
                    self._agent_global_position[1] + direction[1], 0, n_cells_y - 1
                )
            else:
                # will not move if going to colide
                pass

            self._agent_local_position = self._get_agent_local_position()

            return self._agent_global_position, old_pos

        def update_trajectory(
            self,
            n_trajectory_points,
            explored_radius,
            n_cells_x,
            n_cells_y,
            initial_free_cells,
            shared_exploration_grid_matrix,
            other_robot_visited_x,
            other_robot_visited_y,
        ):

            _initial_local_position = self._get_agent_local_position()

            # LOGICS ENVOLVING THE AGENT'S TRAJECTORY------------------------------
            self._agent_trajectory = agent_trajectory.drop_random_points(
                self._agent_trajectory.copy(),
                self._agent_local_position,
                self.n_steps,
                n_trajectory_points,
                _initial_local_position,
            )

            # IS VISITED REGION NEW?-----------------------------------
            all_visited_x = np.concatenate((self._visited_x, other_robot_visited_x))
            all_visited_y = np.concatenate((self._visited_y, other_robot_visited_y))
            new_region = True
            for i in range(len(all_visited_x)):
                if (all_visited_x[i] == self._agent_local_position[0]) and (
                    all_visited_y[i] == self._agent_local_position[1]
                ):
                    new_region = False
                    break
            if new_region:
                self._visited_x.append(self._agent_local_position[0])
                self._visited_y.append(self._agent_local_position[1])

            # LOGICA EXPLORED RATE--------------------------------------
            old_number_of_free_cells = (
                shared_exploration_grid_matrix.size
                - np.count_nonzero(shared_exploration_grid_matrix)
            )

            self._define_explored_area(
                explored_radius, n_cells_x, n_cells_y, shared_exploration_grid_matrix
            )

            self.number_of_free_cells = (
                shared_exploration_grid_matrix.size
                - np.count_nonzero(shared_exploration_grid_matrix)
            )

            number_of_visited_cells = initial_free_cells - self.number_of_free_cells
            self.explored_rate = number_of_visited_cells / initial_free_cells

            # exploration delta in number of cells
            new_explored_cells = abs(
                old_number_of_free_cells - self.number_of_free_cells
            )

            self.exploration_delta = new_explored_cells

            return new_region
            # ------------------------------------------------------------------------

    def _init_map(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window_width = self.canvas.get_width()
            self.window_height = self.canvas.get_height()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )

        else:
            self.window_width = self.canvas.get_width()
            self.window_height = self.canvas.get_height()

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.n_cells_x = int(
            self.window_width / self.pix_square_size
        )  # The size of the square grid
        self.n_cells_y = int(
            self.window_height / self.pix_square_size
        )  # The size of the square grid

        # Determine obstacle in the grid matrix
        self.grid_matrix = np.zeros(shape=(self.n_cells_x, self.n_cells_y))
        for x in range(self.n_cells_x):
            for y in range(self.n_cells_y):
                found_obstacle = 0
                for i in range(int(self.pix_square_size)):
                    if found_obstacle:
                        break
                    for j in range(int(self.pix_square_size)):
                        pixel = pygame.Surface.get_at(
                            self.canvas,
                            (
                                int(i + x * self.pix_square_size),
                                int(j + y * self.pix_square_size),
                            ),
                        )
                        if (pixel[0] != 255) or (pixel[1] != 255) or (pixel[2] != 255):
                            self.grid_matrix[x][y] = 2.0
                            found_obstacle = 1
                            break

    def generate_random_position(self, robotID):
        is_wall = True

        random_pos = None
        if robotID == 0:
            while is_wall:
                random_pos = self.np_random.integers(
                    np.array([0, 0]),
                    np.array([self.n_cells_x / 2, self.n_cells_y]),
                    size=(2,),
                    dtype=int,
                )
                if self.grid_matrix[random_pos[0]][random_pos[1]] == 0:
                    is_wall = False

        elif robotID == 1:
            while is_wall:
                random_pos = self.np_random.integers(
                    np.array([self.n_cells_x / 2, 0]),
                    np.array([self.n_cells_x, self.n_cells_y]),
                    size=(2,),
                    dtype=int,
                )
                if self.grid_matrix[random_pos[0]][random_pos[1]] == 0:
                    is_wall = False

        return random_pos

    def reset(self, seed=None, options=None):
        self.episode += 1
        super().reset(seed=seed)

        taken_positions = []
        observations = {"robot0": []}

        self.first_steps = np.ones(self.num_robots)

        self.exploration_grid_matrix = self.grid_matrix.copy()

        for i in range(self.num_robots):

            # reset robot vars
            self.robots[i].reset()

            repeating = True

            curr_position = (0, 0)

            # Choose the agent's initial position uniformly at random, making sure it does not coincide with a wall
            while repeating:
                repeating = False
                curr_position = self.generate_random_position(i)

                for pos in taken_positions:
                    if pos[0] == curr_position[0]:
                        if pos[1] == curr_position[1]:
                            repeating = True
                            break

            taken_positions.append(curr_position)

        for i in range(self.num_robots):

            self.robots[i]._set_agent_local_init_position(taken_positions[i])

            self.robots[i].init_robot_map(
                self.explored_radius,
                self.n_cells_x,
                self.n_cells_y,
                self.n_trajectory_points,
                self.n_others_trajectory_points,
                self.exploration_grid_matrix,
                taken_positions[i],
            )  # owner is robot

        # ROBOTS SHARING THEIR TRAJECTORY
        self.robots[1]._other_agents_trajectory = self.robots[0]._agent_trajectory
        self.robots[0]._other_agents_trajectory = self.robots[1]._agent_trajectory

        for i in range(self.num_robots):
            self.robots[i]._get_laser_measurements(
                self.grid_matrix, self.n_cells_x, self.n_cells_y
            )  # owner is robot

            observation = self.robots[i]._get_obs()  # owner is robot
            robot_name = "robot" + str(i)
            observations[robot_name] = observation

        info = {}
        info_robot_1 = self.robots[0]._get_info()  # owner is robot
        info_robot_2 = self.robots[1]._get_info()
        info["robot1"] = info_robot_1
        info["robot2"] = info_robot_2

        self.robots[0].n_steps = 0
        self.robots[1].n_steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return observations, info

    def step(self, action):
        # MOVE ROBOT--------------------------------------------------------------
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        self.global_steps += 1

        i = 0

        add_pos = []
        observations = {"robot0": []}
        new_regions = np.zeros(2)

        # first move all the robots
        for i in range(self.num_robots):
            if isinstance(action, (list, tuple, np.ndarray)):
                action_item = action[i]
            else:
                action_item = action
            direction = self._action_to_direction[action_item]
            self.robots[i].n_steps += 1

            if i == 0:
                curr_pos, old_pos = self.robots[i].move(
                    direction, self.n_cells_x, self.n_cells_y
                )
            else:  # does not hit walls
                curr_pos, old_pos = self.robots[i].move_safe(
                    action_item, direction, self.n_cells_x, self.n_cells_y
                )

            add_pos.append(curr_pos)

            if i == 0:
                # print("ROBOT 0")
                new_regions[i] = self.robots[i].update_trajectory(
                    self.n_trajectory_points,
                    self.explored_radius,
                    self.n_cells_x,
                    self.n_cells_y,
                    self.initial_free_cells,
                    self.exploration_grid_matrix,
                    self.robots[1]._visited_x,
                    self.robots[1]._visited_y,
                )
            elif i == 1:
                # print("ROBOT 1")
                new_regions[i] = self.robots[i].update_trajectory(
                    self.n_trajectory_points,
                    self.explored_radius,
                    self.n_cells_x,
                    self.n_cells_y,
                    self.initial_free_cells,
                    self.exploration_grid_matrix,
                    self.robots[0]._visited_x,
                    self.robots[0]._visited_y,
                )

        # SHARE TRAJECTORY BETWEEN ROBOTS
        self.robots[1]._other_agents_trajectory = self.robots[0]._agent_trajectory
        self.robots[0]._other_agents_trajectory = self.robots[1]._agent_trajectory

        # then measure all the lasers
        for i in range(self.num_robots):

            colider_matrix = self.grid_matrix.copy()

            # update colider map here
            for j in range(self.num_robots):
                if i != j:
                    other_robot_pos = add_pos[j]
                    colider_matrix[other_robot_pos[0]][other_robot_pos[1]] = 1

            robot_lasers = self.robots[i]._get_laser_measurements(
                colider_matrix, self.n_cells_x, self.n_cells_y
            )
            if i == 0:
                if any(robot_lasers == 4):
                    collision = True
                    self.robots[0].n_collisions = True
                else:
                    collision = False
                    self.robots[0].n_collisions = False

            observation = self.robots[i]._get_obs()
            # print(f"observation single robot = {observation}")
            robot_name = "robot" + str(i)
            observations[robot_name] = observation

        info = {}
        info_robot_1 = self.robots[0]._get_info()  # owner is robot
        info_robot_2 = self.robots[1]._get_info()
        info["robot1"] = info_robot_1
        info["robot2"] = info_robot_2
        # info = self.robots[0]._get_info()
        exploration_delta = self.robots[1].exploration_delta

        # this is done for the only robot that is training:
        # REWARDS AND DONE------------------------------------------------------
        terminated = False
        if collision:
            reward = self.max_penalty
            terminated = True
            # print("colision")
        elif self.first_steps[0]:
            reward = 0.0
            self.first_steps = np.zeros(2)
            # print("first step")
        elif self.robots[0].explored_rate >= self.max_exploration_rate:
            reward = self.max_reward
            terminated = True
            # print("explored more than 93%")
        elif self.robots[0].exploration_delta > 0.0:
            if (
                self.robots[0].exploration_delta + self.robots[1].exploration_delta
            ) / 2.0 > 20:
                reward = 20.0
            else:
                reward = (
                    self.robots[0].exploration_delta + self.robots[1].exploration_delta
                ) / 2.0
        elif new_regions[0]:
            reward = self.reward_per_new_region
        else:
            reward = self.penalty_per_old_region
        if self.render_mode == "human":
            self._render_frame()

        return observations, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        self.canvas = self.clean_canvas.copy()

        if self.render_mode == "human":
            for x in range(np.shape(self.exploration_grid_matrix)[0]):
                for y in range(np.shape(self.exploration_grid_matrix)[1]):
                    if self.exploration_grid_matrix[x, y] == 1.0:
                        pygame.draw.rect(
                            self.canvas,
                            (0, 200, 0),
                            pygame.Rect(
                                self.pix_square_size * np.array([x, y]),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )
            for x in range(np.shape(self.exploration_grid_matrix)[0]):
                for y in range(np.shape(self.exploration_grid_matrix)[1]):
                    if self.exploration_grid_matrix[x, y] == 2.0:
                        pygame.draw.rect(
                            self.canvas,
                            (0, 0, 0),
                            pygame.Rect(
                                self.pix_square_size * np.array([x, y]),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

            # for each robot:
            if self.draw_lasers == "True":
                for i in range(self.num_robots):
                    robot_position = self.robots[i]._agent_global_position
                    robot_lasers = self.robots[i]._lasers

                    # Draw lasers
                    # North lasers
                    for i in range(3):
                        if robot_lasers[0] == 1:
                            color = (0, 255, 0)
                        elif robot_lasers[0] == 2:
                            color = (255, 165, 0)
                        elif robot_lasers[0] == 3:
                            color = (255, 0, 0)
                        elif robot_lasers[0] == 4:
                            color = (255, 0, 0)
                        pygame.draw.rect(
                            self.canvas,
                            color,
                            pygame.Rect(
                                self.pix_square_size
                                * np.array(
                                    [
                                        robot_position[0],
                                        robot_position[1] - i - 1,
                                    ]
                                ),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

                    # South
                    for i in range(3):
                        if robot_lasers[1] == 1:
                            color = (0, 255, 0)
                        elif robot_lasers[1] == 2:
                            color = (255, 165, 0)
                        elif robot_lasers[1] == 3:
                            color = (255, 0, 0)
                        elif robot_lasers[1] == 4:
                            color = (255, 0, 0)
                        pygame.draw.rect(
                            self.canvas,
                            color,
                            pygame.Rect(
                                self.pix_square_size
                                * np.array(
                                    [
                                        robot_position[0],
                                        robot_position[1] + i + 1,
                                    ]
                                ),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

                    # East
                    for i in range(3):
                        if robot_lasers[2] == 1:
                            color = (0, 255, 0)
                        elif robot_lasers[2] == 2:
                            color = (255, 165, 0)
                        elif robot_lasers[2] == 3:
                            color = (255, 0, 0)
                        elif robot_lasers[2] == 4:
                            color = (255, 0, 0)
                        pygame.draw.rect(
                            self.canvas,
                            color,
                            pygame.Rect(
                                self.pix_square_size
                                * np.array(
                                    [
                                        robot_position[0] + i + 1,
                                        robot_position[1],
                                    ]
                                ),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

                    # West
                    for i in range(3):
                        if robot_lasers[3] == 1:
                            color = (0, 255, 0)
                        elif robot_lasers[3] == 2:
                            color = (255, 165, 0)
                        elif robot_lasers[3] == 3:
                            color = (255, 0, 0)
                        elif robot_lasers[3] == 4:
                            color = (255, 0, 0)
                        pygame.draw.rect(
                            self.canvas,
                            color,
                            pygame.Rect(
                                self.pix_square_size
                                * np.array(
                                    [
                                        robot_position[0] - i - 1,
                                        robot_position[1],
                                    ]
                                ),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

            if self.draw_agent == "True":
                for i in range(self.num_robots):
                    robot_position = self.robots[i]._agent_global_position
                    # Draw the agent
                    if i == 0:
                        color = (255, 0, 0)
                    else:
                        color = (0, 100, 255)
                    pygame.draw.circle(
                        self.canvas,
                        color,
                        (robot_position + 0.5) * self.pix_square_size,
                        self.pix_square_size / 3,
                    )

            if self.grid_lines == "True":
                # Add some gridlines
                for x in range(self.n_cells_x + 1):
                    pygame.draw.line(
                        self.canvas,
                        0,
                        (0, self.pix_square_size * x),
                        (self.window_width, self.pix_square_size * x),
                        width=3,
                    )
                    pygame.draw.line(
                        self.canvas,
                        0,
                        (self.pix_square_size * x, 0),
                        (self.pix_square_size * x, self.window_height),
                        width=3,
                    )
                    self.pix_square_size * x

        if self.window is not None and self.render_mode == "human":
            # The following line copies our drawings from `self.canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed
