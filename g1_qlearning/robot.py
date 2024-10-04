# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.




import random
class Robot:

    x_pos = 0
    y_pos = 0
    alpha = 0.8 # Learning rate
    gamma = 0.8
    reward_matrix = []
    q_matrix = []
    running = False

    def __init__(self):
        # Define R- and Q-matrices here.

        # Lager en større reward matrix, for at robot skal lære å ikke gå utenfor satelittområdet.
        self.reward_matrix = [[-100, -100, -100, -100, -100, -100, -100, -100],
                     [-100, -50, -25, -25, 0, 0, -50, -100],
                     [-100, -50, -50, 0, -25, 0, 0, -100],
                     [-100, -25, 0, 0, -25, 0, -25, -100],
                     [-100, -25, 0, 0, 0, 0, 0, -100],
                     [-100, -25, 0, -25, 0, -25, 0, -100],
                     [-100, 100, -50, -50, -50, -50, -50, -100],
                     [-100, -100, -100, -100, -100, -100, -100, -100]]

        self.wall_value = -100
        self.hill_value = -4
        self.water_value = -10
        self.plain_value = 0
        self.goal_value = 100
        self.reward_matrix2 = { #   Up,               Down              Left            Right
                                1: [self.wall_value, self.water_value, self.wall_value, self.hill_value],
                                2: [self.wall_value, self.water_value, self.water_value, self.hill_value],
                                3: [self.wall_value, self.plain_value, self.hill_value, self.plain_value],
                                4: [self.wall_value, self.hill_value, self.hill_value, self.plain_value],
                                5: [self.wall_value, self.plain_value, self.plain_value, self.water_value],
                                6: [self.wall_value, self.plain_value, self.plain_value, self.wall_value],

                                7: [self.water_value, self.hill_value, self.wall_value, self.water_value],
                                8: [self.hill_value, self.plain_value, self.water_value, self.plain_value],
                                9: [self.hill_value, self.plain_value, self.water_value, self.hill_value],
                                10: [self.plain_value, self.hill_value, self.plain_value, self.plain_value],
                                11: [self.plain_value, self.plain_value, self.hill_value, self.plain_value],
                                12: [self.water_value, self.hill_value, self.plain_value, self.wall_value],

                                13: [self.water_value, self.hill_value, self.wall_value, self.plain_value],
                                14: [self.water_value, self.plain_value, self.hill_value, self.plain_value],
                                15: [self.plain_value, self.plain_value, self.plain_value, self.hill_value],
                                16: [self.hill_value, self.plain_value, self.plain_value, self.plain_value],
                                17: [self.plain_value, self.plain_value, self.hill_value, self.hill_value],
                                18: [self.plain_value, self.plain_value, self.plain_value, self.wall_value],

                                19: [self.hill_value, self.hill_value, self.wall_value, self.plain_value],
                                20: [self.plain_value, self.plain_value, self.hill_value, self.plain_value],
                                21: [self.plain_value, self.hill_value, self.plain_value, self.plain_value],
                                22: [self.hill_value, self.plain_value, self.plain_value, self.plain_value],
                                23: [self.plain_value, self.hill_value, self.plain_value, self.plain_value],
                                24: [self.hill_value, self.plain_value, self.plain_value, self.wall_value],

                                25: [self.hill_value, self.goal_value, self.wall_value, self.plain_value],
                                26: [self.plain_value, self.water_value, self.hill_value, self.hill_value],
                                27: [self.plain_value, self.water_value, self.plain_value, self.plain_value],
                                28: [self.plain_value, self.water_value, self.hill_value, self.hill_value],
                                29: [self.plain_value, self.water_value, self.plain_value, self.plain_value],
                                30: [self.plain_value, self.water_value, self.hill_value, self.wall_value],

                                31: [self.hill_value, self.wall_value, self.wall_value, self.water_value],
                                32: [self.plain_value, self.wall_value, self.goal_value, self.water_value],
                                33: [self.hill_value, self.wall_value, self.water_value, self.water_value],
                                34: [self.plain_value, self.wall_value, self.water_value, self.water_value],
                                35: [self.hill_value, self.wall_value, self.water_value, self.water_value],
                                36: [self.plain_value, self.wall_value, self.water_value, self.wall_value]
                            }
    
        # Tilstander * handlinger: Tenker oss at det er 6*6 = 36 tilstander 4 handlinger opp/ned/høyre/venstre.
        #self.q_matrix = [ [0] * 4 for _ in range(1, 37)]
        self.q_matrix = {i: [0] * 4 for i in range(1, 37)}


    def get_x(self):
        # Return the current column of the robot, should be in the range 0-5.
        return self.x_pos-1


    def get_y(self):
        # Return the current row of the robot, should be in the range 0-5.
        return self.y_pos-1


    def get_next_state_mc(self):
        # Return the next state based on Monte Carlo.
        action = random.randint(0,3)
        if action == 0: # Opp
            current_state = self.get_state()
            # self.y_pos -= 1
            # reward = self.reward_matrix[self.y_pos - 1][self.x_pos]
            # next_state = self.get_state(self.x_pos, self.y_pos-1)
            # self.reward_update(current_state, action, reward, next_state)
            # self.q_matrix[current_state][random_num] = \
            #     self.reward_matrix[self.x_pos][self.y_pos] + self.gamma*max(self.q_matrix[self.get_state()])

            if self.y_pos != 1:
                self.y_pos -= 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)
            
            
        if action == 1: # Ned
            current_state = self.get_state()

            if self.y_pos != 6:
                self.y_pos += 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)
            
        if action == 2: # Venstre
            current_state = self.get_state()
            #reward = self.reward_matrix[self.y_pos][self.x_pos - 1]

            #self.reward_update(current_state, action, reward)
            if self.x_pos != 1:
                self.x_pos -= 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)
            
        if action == 3: # Høyre
            current_state = self.get_state()
            # reward = self.reward_matrix[self.y_pos][self.x_pos + 1]
            # self.reward_update(current_state, action, reward)

            if self.x_pos != 6:
                self.x_pos += 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)
            

    def get_next_state_eg(self):
        # Return the next state based on Epsilon-greedy.
        pass


    def monte_carlo_exploration(self):
        pass


    def q_learning(self):
        pass


    def one_step_q_learning(self, policy):
        # Get action based on policy
        # Get the next state based on the action
        # Get the reward for going to this state
        # Update the Q-matrix
        # Go to the next state
        if policy == "MC":
            self.running = True
            self.get_next_state_mc()
        if policy == "Greedy":
            self.running = False
        if policy == "Epilson":
            self.running = False
    

    def has_reached_goal(self, goal):
        # Return 'True' if the robot is in the goal state.
        if (self.x_pos == goal['X']) and (self.y_pos == goal['Y']):
            #self.running = False
            print(self.q_matrix)
            for x in range(1, 37):
                print(f"State {x}: {self.q_matrix[x]}:")
            return True
        else:
            return False

        
    def reset_random(self):
        # Place the robot in a new random state.
        import random
        self.x_pos = random.randint(1, 6)
        self.y_pos = random.randint(1, 6)

    def greedy_path(self):
        pass

    def get_state(self):
        print(f"State: {(self.y_pos-1)*6 + self.x_pos} X: {self.x_pos} Y: {self.y_pos}")
        return (self.y_pos-1)*6 + self.x_pos
        # print(f"State: {(self.y_pos-1)*6 + self.x_pos} X: {self.x_pos} Y: {self.y_pos}")
        #     return (self.y_pos-1)*6 + self.x_pos
    
    def get_state_pos(self, state):
        #print(f"State: {state}")
        y = state // 6
        x = state % 6
        # print(f"X: {x} Y: {y}")
        return x,y

    def reward_update(self, current_state, action, next_state):
        #current_state = self.get_state()
            #next_state = self.get_state()
        #x, y = self.get_state_pos(new_state)
        # if x == 0 and y == 5:
        #     print(f"Reward: {self.reward_matrix[x][y]}")


        # self.q_matrix[current_state][action] = \
        #     self.reward_matrix[x+1][y+1] + self.gamma*max(self.q_matrix[self.get_state()])
        
        # print(f"State: {current_state} Action: {action} New: X: {x} Y: {y}")
        print(current_state)
        #self.q_matrix[current_state][action] = reward + self.gamma*max(self.q_matrix[self.get_state()])


        self.q_matrix[current_state][action] = (1 - self.alpha) * self.q_matrix[current_state][action]\
                                                + self.alpha * (self.reward_matrix2[current_state][action]\
                                                                + self.gamma * max(self.q_matrix[next_state]))
                                                
        
        # self.q_matrix[current_state][next_state] = \
        #     self.reward_matrix[self.x_pos][self.y_pos] + self.gamma*max(self.q_matrix[self.get_state()])

    def get_next_state(state, action):
        if action == 0: #Opp
            pass
        if action == 1: #Ned
            pass
        if action == 2: #Venstre
            pass
        if action == 3: #Høyre
            pass

# Feel free to add additional classes / methods / functions to solve the assignment...