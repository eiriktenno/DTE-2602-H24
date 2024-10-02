# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.




import random
class Robot:

    x_pos = 0
    y_pos = 0
    alpha = 1 # Learning rate
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
    
        # Tilstander * handlinger: Tenker oss at det er 6*6 = 36 tilstander 4 handlinger opp/ned/høyre/venstre.
        self.q_matrix = [ [0] * 4 for _ in range(36)]


    def get_x(self):
        # Return the current column of the robot, should be in the range 0-5.
        return self.x_pos


    def get_y(self):
        # Return the current row of the robot, should be in the range 0-5.
        return self.y_pos


    def get_next_state_mc(self):
        # Return the next state based on Monte Carlo.
        action = random.randint(0,3)
        if action == 0: # Opp
            current_state = self.get_state()

            self.y_pos -= 1

            new_state = self.get_state()

            self.reward_update(current_state, action, new_state)
            # self.q_matrix[current_state][random_num] = \
            #     self.reward_matrix[self.x_pos][self.y_pos] + self.gamma*max(self.q_matrix[self.get_state()])
            if self.y_pos < 0:
                self.y_pos = 0
            
            
        if action == 1: # Ned
            current_state = self.get_state()

            self.y_pos += 1

            new_state = self.get_state()

            self.reward_update(current_state, action, new_state)
            if self.y_pos > 5:
                self.y_pos = 5
            
        if action == 2: # Venstre
            current_state = self.get_state()

            self.x_pos -= 1

            new_state = self.get_state()

            self.reward_update(current_state, action, new_state)
            if self.x_pos < 0:
                self.x_pos = 0
            
        if action == 3: # Høyre
            current_state = self.get_state()

            self.x_pos += 1

            new_state = self.get_state()

            self.reward_update(current_state, action, new_state)
            if self.x_pos > 5:
                self.x_pos = 5
            

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
            self.running = False
            for x in range(len(self.q_matrix)):
                print(f"State {x}: {self.q_matrix[x]}:")
            return True
        else:
            return False

        
    def reset_random(self):
        # Place the robot in a new random state.
        import random
        self.x_pos = random.randint(0, 5)
        self.y_pos = random.randint(0, 5)

    def greedy_path(self):
        pass

    def get_state(self):
        return (self.y_pos-1)*6 + self.x_pos
    
    def get_state_pos(self, state):
        print(f"State: {state}")
        y = state // 6
        x = state % 6
        print(f"X: {x} Y: {y}")
        return x,y

    def reward_update(self, current_state, action, new_state):
        #current_state = self.get_state()
            #next_state = self.get_state()
        x, y = self.get_state_pos(new_state)
        # if x == 0 and y == 5:
        #     print(f"Reward: {self.reward_matrix[x][y]}")
        self.q_matrix[current_state][action] = \
            self.reward_matrix[x+1][y+1] + self.gamma*max(self.q_matrix[self.get_state()])
        
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