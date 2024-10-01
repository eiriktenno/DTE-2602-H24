# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.




import random
class Robot:

    x_pos = 0
    y_pos = 0
    alpha = 1
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
        self.q_matrix = [ [0] * 36 for _ in range(4)]


    def get_x(self):
        # Return the current column of the robot, should be in the range 0-5.
        return self.x_pos


    def get_y(self):
        # Return the current row of the robot, should be in the range 0-5.
        return self.y_pos


    def get_next_state_mc(self):
        # Return the next state based on Monte Carlo.
        pass


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
            random_num = random.randint(0,3)
            if random_num == 0:
                if self.y_pos != 0:
                    self.y_pos -= 1
            if random_num == 1:
                if self.y_pos != 5:
                    self.y_pos += 1
            if random_num == 2:
                if self.x_pos != 0:
                    self.x_pos -= 1
            if random_num == 3:
                if self.x_pos != 5:
                    self.x_pos +=1
        if policy == "Greedy":
            self.running = False
        if policy == "Epilson":
            self.running = False
    

    def has_reached_goal(self, goal):
        # Return 'True' if the robot is in the goal state.
        if (self.x_pos == goal['X']) and (self.y_pos == goal['Y']):
            self.running = False
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

# Feel free to add additional classes / methods / functions to solve the assignment...