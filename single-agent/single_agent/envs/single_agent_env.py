import gym
from gym import spaces
import pygame
import numpy as np
import random
from modules import trajectory
import json


class SingleAgentEnv(gym.Env):
    with open(f"parameters.json", "r") as parameters_file:
        parameters = json.load(parameters_file)

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": parameters["visualization"]["FPS"],
    }

    def __init__(
        self,
        render_mode=None,
    ):
        with open(f"parameters.json", "r") as parameters_file:
            parameters = json.load(parameters_file)

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
        self.number_of_free_cells = 0

        # Rewards
        self.max_penalty = parameters["reward"]["max_penalty"]
        self.max_reward = parameters["reward"]["max_reward"]
        self.penalty_per_old_region = parameters["reward"]["penalty_per_old_region"]
        self.reward_per_new_region = parameters["reward"]["reward_per_new_region"]
        self.divide_delta = parameters["reward"]["divide_delta"]

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

        self.explored_rate = 0.0
        self.n_steps = 0.0
        self.global_steps = 0.0
        self.episode = 0
        self.merge_iteration = 0
        self.collision = False

        self.last_dropped_position = 0

        max_bound = self.n_cells_x
        if self.n_cells_y > self.n_cells_x:
            max_bound = self.n_cells_y

        self.observation_space = spaces.Dict(
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
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "up", 1 to "down", 2 to right, 3 to left.
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

        # EXPLORATION GRID
        self.exploration_grid_matrix = self.grid_matrix.copy()
        self.initial_free_cells = self.grid_matrix.size - np.count_nonzero(
            self.grid_matrix
        )  # value use to determine how explortion rate is increasing

    # Function to translates the environment’s state into an observation.
    def _get_obs(self):
        return {
            "proximity_regions": self._lasers,
            "trajectory": self._agent_trajectory,
            "other_robots_trajectories": self._other_agents_trajectory,
        }

    def _get_info(self):
        return {
            "explored_rate": self.explored_rate,
            "path_lenght": self.n_steps,
            "collision": self.collision,
        }

    def _get_laser_measurements(self):
        current_x = self._agent_global_position[0]
        current_y = self._agent_global_position[1]

        north_laser = 1  # Safe
        east_laser = 1  # Safe
        west_laser = 1  # Safe
        south_laser = 1  # Safe

        # North
        # Check edges
        if current_y == 0:
            north_laser = 4  # self.collision
        elif current_y == 1:
            north_laser = 3  # Risk - it is on the top edge
        elif current_y == 2:
            north_laser = 2  # Caution - it one cell away from the top edge
        # Check for sourrounding objects
        else:
            if self.grid_matrix[current_x][current_y]:
                north_laser = 4
            elif self.grid_matrix[current_x][current_y - 1]:
                north_laser = 3
            elif self.grid_matrix[current_x][current_y - 2]:
                north_laser = 2

        # South
        # Check edges
        if current_y == self.n_cells_y - 1:
            south_laser = 4  # self.collision
        elif current_y == self.n_cells_y - 2:
            south_laser = 3  # Risk - it is on the bottom edge
        elif current_y == self.n_cells_y - 3:
            south_laser = 2  # Caution - it one cell away from the bottom edge
        # Check for sourrounding objects
        else:
            if self.grid_matrix[current_x][current_y]:
                south_laser = 4
            elif self.grid_matrix[current_x][current_y + 1]:
                south_laser = 3
            elif self.grid_matrix[current_x][current_y + 2]:
                south_laser = 2

        # East
        # Check edges
        if current_x == self.n_cells_x - 1:
            east_laser = 4  # self.collision
        elif current_x == self.n_cells_x - 2:
            east_laser = 3  # Risk - it is on the east edge
        elif current_x == self.n_cells_x - 3:
            east_laser = 2  # Caution - it one cell away from the east edge
        # Check for sourrounding objects
        else:
            if self.grid_matrix[current_x][current_y]:
                east_laser = 4
            elif self.grid_matrix[current_x + 1][current_y]:
                east_laser = 3
            elif self.grid_matrix[current_x + 2][current_y]:
                east_laser = 2

        # West
        # Check edges
        if current_x == 0:
            west_laser = 4  # self.collision
        elif current_x == 1:
            west_laser = 3  # Risk - it is on the west edge
        elif current_x == 2:
            west_laser = 2  # Caution - it one cell away from the west edge
        # Check for sourrounding objects
        else:
            if self.grid_matrix[current_x][current_y]:
                west_laser = 4
            elif self.grid_matrix[current_x - 1][current_y]:
                west_laser = 3
            elif self.grid_matrix[current_x - 2][current_y]:
                west_laser = 2

        self._lasers = np.array(
            [int(north_laser), int(south_laser), int(east_laser), int(west_laser)]
        )

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

    def _define_explored_area(self):

        list_of_points = []
        self.exploration_grid_matrix
        x_curr = self._agent_global_position[0]
        y_curr = self._agent_global_position[1]

        for i in range(2):
            if i == 0:
                x = x_curr - self.explored_radius
            else:
                x = x_curr + self.explored_radius + 1

            for y in range(
                y_curr - self.explored_radius, y_curr + self.explored_radius + 1
            ):
                list_of_points.append((x, y))

            if i == 0:
                y = y_curr - self.explored_radius
            else:
                y = y_curr + self.explored_radius + 1

            for x in range(
                x_curr - self.explored_radius, x_curr + self.explored_radius + 1
            ):
                list_of_points.append((x, y))

        for point in list_of_points:
            line_is_get = self.get_line(self._agent_global_position, point)

            for cell in line_is_get:
                if cell[0] >= 0 and cell[0] < self.n_cells_x:
                    if cell[1] >= 0 and cell[1] < self.n_cells_y:
                        # if any on the list is a wall
                        if self.exploration_grid_matrix[cell[0]][cell[1]] == 2.0:
                            break
                        else:
                            self.exploration_grid_matrix[cell[0]][cell[1]] = 1

    def reset(self, seed=None, options=None):
        self.episode += 1
        self.first_step = 1

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's initial position uniformly at random, making sure it does not coincide with a wall
        is_wall = True
        while is_wall:
            self._initial_global_position = self.np_random.integers(
                np.array([0, 0]),
                np.array([self.n_cells_x, self.n_cells_y]),
                size=(2,),
                dtype=int,
            )
            if (
                self.grid_matrix[self._initial_global_position[0]][
                    self._initial_global_position[1]
                ]
                == 0.0
            ):
                is_wall = False
        self._agent_global_position = self._initial_global_position.copy()

        # Mark cells around the robot's initial poistion as explored
        self.exploration_grid_matrix = self.grid_matrix.copy()

        self._define_explored_area()

        self._initial_local_position = self._get_agent_local_position()
        self._agent_local_position = self._get_agent_local_position()
        self.merge_iteration = 0
        self._agent_trajectory = np.zeros(
            shape=(self.n_trajectory_points, 2), dtype=int
        )
        self._other_agents_trajectory = np.zeros(
            shape=(self.n_others_trajectory_points, 2), dtype=int
        )

        self._visited_x = [self._agent_local_position[0]]
        self._visited_y = [self._agent_local_position[1]]

        self._visited_x_global = [self._agent_global_position[0]]
        self._visited_y_global = [self._agent_global_position[1]]

        self._get_laser_measurements()

        observation = self._get_obs()
        info = self._get_info()

        self.n_steps = 0
        self.explored_rate = 0.0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # MOVE ROBOT--------------------------------------------------------------
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_global_position[0] = np.clip(
            self._agent_global_position[0] + direction[0], 0, self.n_cells_x - 1
        )
        self._agent_global_position[1] = np.clip(
            self._agent_global_position[1] + direction[1], 0, self.n_cells_y - 1
        )

        self._agent_local_position = self._get_agent_local_position()

        self.n_steps += 1
        self.global_steps += 1
        # ---------------------------------------------------------------------------

        # LOGICS ENVOLVING THE AGENT'S TRAJECTORY------------------------------
        self._agent_trajectory = trajectory.drop_random_points(
            self._agent_trajectory.copy(),
            self._agent_local_position,
            self.n_steps,
            self.n_trajectory_points,
            self._initial_local_position,
        )  # Other trajectory logics available in te agent_trajectory module

        self.merge_iteration += 1

        # IS VISITED REGION NEW?-----------------------------------
        new_region = True
        for i in range(len(self._visited_x)):
            if (self._visited_x[i] == self._agent_local_position[0]) and (
                self._visited_y[i] == self._agent_local_position[1]
            ):
                new_region = False
                break
        if new_region:
            self._visited_x.append(self._agent_local_position[0])
            self._visited_y.append(self._agent_local_position[1])
            self._visited_x_global.append(self._agent_global_position[0])
            self._visited_y_global.append(self._agent_global_position[1])

        # EXPLORED RATE LOGIC--------------------------------------
        self._define_explored_area()

        old_number_of_free_cells = self.number_of_free_cells
        self.number_of_free_cells = (
            self.exploration_grid_matrix.size
            - np.count_nonzero(self.exploration_grid_matrix)
        )

        number_of_visited_cells = self.initial_free_cells - self.number_of_free_cells
        self.explored_rate = number_of_visited_cells / self.initial_free_cells

        # exploration delta in number of cells
        new_explored_cells = abs(old_number_of_free_cells - self.number_of_free_cells)
        if new_explored_cells > (2 * self.explored_radius + 1):
            self.exploration_delta = 2 * self.explored_radius + 1
        else:
            self.exploration_delta = new_explored_cells

        # LASER MEASUREMENTS----------------------------------------------------
        self._get_laser_measurements()

        # Get new observation
        observation = self._get_obs()

        # REWARDS AND DONE------------------------------------------------------
        # Determine if a collision happend
        if any(self._lasers == 4):
            self.collision = True
        else:
            self.collision = False

        info = self._get_info()

        terminated = False
        if self.n_steps > self.max_steps_to_done:
            terminated = True

        if self.collision:
            reward = self.max_penalty
            terminated = True
            # print("colision")
        elif self.first_step:
            reward = 0.0
            self.first_step = 0
        elif self.explored_rate >= self.max_exploration_rate:
            reward = self.max_reward
            terminated = True
            # print(f"explored more than {self.max_exploration_rate}")
        elif self.exploration_delta > 0.0:
            if self.exploration_delta < 10.0:
                reward = self.exploration_delta  # / self.divide_delta
            else:
                reward = 10.0
        elif new_region:
            reward = (
                self.reward_per_new_region
            )  # Did not increased explored area, but position is new
        else:
            reward = self.penalty_per_old_region
        # --------------------------------------------------------------------

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        self.canvas = self.clean_canvas.copy()

        # Paint obstacles in rendered map
        if self.render_mode == "human":
            for x in range(np.shape(self.exploration_grid_matrix)[0]):
                for y in range(np.shape(self.exploration_grid_matrix)[1]):
                    if self.exploration_grid_matrix[x, y]:
                        pygame.draw.rect(
                            self.canvas,
                            (0, 200, 0),
                            pygame.Rect(
                                self.pix_square_size * np.array([x, y]),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

            for x in range(np.shape(self.grid_matrix)[0]):
                for y in range(np.shape(self.grid_matrix)[1]):
                    if self.grid_matrix[x, y]:
                        pygame.draw.rect(
                            self.canvas,
                            (0, 0, 0),
                            pygame.Rect(
                                self.pix_square_size * np.array([x, y]),
                                (self.pix_square_size, self.pix_square_size),
                            ),
                        )

        if self.draw_trajectory == "True":
            for i in range(len(self._visited_x)):
                pygame.draw.rect(
                    self.canvas,
                    (255, 20, 147),
                    pygame.Rect(
                        self.pix_square_size
                        * np.array(
                            [self._visited_x_global[i], self._visited_y_global[i]]
                        ),
                        (self.pix_square_size, self.pix_square_size),
                    ),
                )

        if self.draw_agent == "True":
            # Draw the agent
            pygame.draw.circle(
                self.canvas,
                (0, 0, 255),
                (self._agent_global_position + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )

        # Draw lasers
        if self.draw_lasers == "True":
            # North lasers
            for i in range(3):
                if self._lasers[0] == 1:
                    color = (0, 100, 0)
                elif self._lasers[0] == 2:
                    color = (255, 165, 0)
                elif self._lasers[0] == 3:
                    color = (255, 0, 0)
                elif self._lasers[0] == 4:
                    color = (255, 0, 0)
                pygame.draw.rect(
                    self.canvas,
                    color,
                    pygame.Rect(
                        self.pix_square_size
                        * np.array(
                            [
                                self._agent_global_position[0],
                                self._agent_global_position[1] - i - 1,
                            ]
                        ),
                        (self.pix_square_size, self.pix_square_size),
                    ),
                )

            # Soute
            for i in range(3):
                if self._lasers[1] == 1:
                    color = (0, 100, 0)
                elif self._lasers[1] == 2:
                    color = (255, 165, 0)
                elif self._lasers[1] == 3:
                    color = (255, 0, 0)
                elif self._lasers[1] == 4:
                    color = (255, 0, 0)
                pygame.draw.rect(
                    self.canvas,
                    color,
                    pygame.Rect(
                        self.pix_square_size
                        * np.array(
                            [
                                self._agent_global_position[0],
                                self._agent_global_position[1] + i + 1,
                            ]
                        ),
                        (self.pix_square_size, self.pix_square_size),
                    ),
                )

            # East
            for i in range(3):
                if self._lasers[2] == 1:
                    color = (0, 100, 0)
                elif self._lasers[2] == 2:
                    color = (255, 165, 0)
                elif self._lasers[2] == 3:
                    color = (255, 0, 0)
                elif self._lasers[2] == 4:
                    color = (255, 0, 0)
                pygame.draw.rect(
                    self.canvas,
                    color,
                    pygame.Rect(
                        self.pix_square_size
                        * np.array(
                            [
                                self._agent_global_position[0] + i + 1,
                                self._agent_global_position[1],
                            ]
                        ),
                        (self.pix_square_size, self.pix_square_size),
                    ),
                )

            # West
            for i in range(3):
                if self._lasers[3] == 1:
                    color = (0, 100, 0)
                elif self._lasers[3] == 2:
                    color = (255, 165, 0)
                elif self._lasers[3] == 3:
                    color = (255, 0, 0)
                elif self._lasers[3] == 4:
                    color = (255, 0, 0)
                pygame.draw.rect(
                    self.canvas,
                    color,
                    pygame.Rect(
                        self.pix_square_size
                        * np.array(
                            [
                                self._agent_global_position[0] - i - 1,
                                self._agent_global_position[1],
                            ]
                        ),
                        (self.pix_square_size, self.pix_square_size),
                    ),
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
