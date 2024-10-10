# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.

import math
import time
import random

class Robot:
    x_pos = 4
    y_pos = 1
    alpha = 0.6 # I et statisk miljø har det liten hensikt å bruke en stor alpha/læringsrate.
                # Det er ikke noen særlige endringer
                # dynamiske miljø = Høy alpha
                # statiske miljø = lav alpha
    gamma = 0.8 # Discount factor/gamma - Hvor stor tro har agenten på at fremtiden vil bringe noe godt.
                # Stor gamma vil si at vi tror den får en stor belønning i fremtiden.
                # Dersom vi ikke er sikker på om man får en stor belønning, kan vi bruke lav gamma.
                # Tro på høy belønning = Høy gamma
                # Usikker på belønning = Lav gamma
    reward_matrix = []
    q_matrix = []
    running = False


    # Konvergens-sjekk
    q_diffs = []
    diff_num = 0.01 # Hvor høy differansen skal være før man sier det er konvergens.
    diff_loops = 10 # Hvor mange ganger på rad man skal få "konvergens" før man slutter av.

    # Epsilon settings
    epsilon = 0.1 # Verdien som blir satt her skal representere hvor høy "random" Eks: 0.1 = 10% random.

    # Greedy
    greedy_max_steps = 1000 # Hvor mange steps man skal kjøre før man bestemmer seg for at greedy ikke finner veien.

    # Monte Carlo
    mc_total_reward = -math.inf
    mc_steps = []

    mc_episode_reward = 0
    mc_episode_steps = []

    def __init__(self):
        # Define R- and Q-matrices here.

        # Lager en større reward matrix, for at robot skal lære å ikke gå utenfor satelittområdet.
        # self.reward_matrix = [[-100, -100, -100, -100, -100, -100, -100, -100],
        #              [-100, -50, -25, -25, 0, 0, -50, -100],
        #              [-100, -50, -50, 0, -25, 0, 0, -100],
        #              [-100, -25, 0, 0, -25, 0, -25, -100],
        #              [-100, -25, 0, 0, 0, 0, 0, -100],
        #              [-100, -25, 0, -25, 0, -25, 0, -100],
        #              [-100, 100, -50, -50, -50, -50, -50, -100],
        #              [-100, -100, -100, -100, -100, -100, -100, -100]]

        self.wall_value = -1000
        self.hill_value = -50
        self.water_value = -100
        self.plain_value = 0
        self.goal_value = 100
        self.step_visited_punishment = 2
        self.reward_matrix = { #   Up,               Down              Left            Right
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

        self.q_matrix = {i: [0] * 4 for i in range(1, 37)}
        self.visited = {}
        self.visited_matrix_reset()

    def get_x(self):
        # Return the current column of the robot, should be in the range 0-5.
        return self.x_pos-1

    def get_y(self):
        # Return the current row of the robot, should be in the range 0-5.
        return self.y_pos-1

    def get_next_state_mc(self, current_state: int) -> tuple:
        # Return the next state based on Monte Carlo.
        x,y = self.get_state_cord(current_state)
        action = random.randint(0,3)
        x,y = self.pos_move(current_state, action)
        self.x_pos = x
        self.y_pos = y
        return self.get_state(y_pos = y, x_pos=x), action

    def get_next_state_eg(self, state:int) -> tuple:
        # Return the next state based on Epsilon-greedy.
        if random.uniform(0, 1) > self.epsilon:
            return self.get_next_state_mc(state)
        else:
            max_reward = max(self.q_matrix[state])
            action = self.q_matrix[state].index(max_reward)
            print(action)
            x,y = self.pos_move(state, action)
            print(f"X: {x}, Y: {y}")
            next_state = self.get_state(x, y)
            print(f"Next State: {next_state}")
            return next_state, action

    def monte_carlo_exploration(self, epochs: int, start_pos: dict, goal_pos: dict) -> list:
        """Monte Carlo exploration
        
        Keyword arguments:
        epochs -- Antall episoder
        start_pos - Start posisjon i dict ('X' og 'Y')
        goal_pos - Mål posisjon i dict ('X' og 'Y')
        Return: Route en liste med de posisjonene som blir tatt.
        """
        self.x_pos = start_pos['X']
        self.y_pos = start_pos['Y']

        best_route = []
        best_reward = -math.inf

        while epochs > 0:
            current_route = []
            total_reward = 0

            state = self.get_state(self.x_pos, self.y_pos)
            current_route.append((self.x_pos, self.y_pos))

            while not self.has_reached_goal(goal_pos):
                # Legger til tilstand i ruten
                #current_route.append((self.x_pos, self.y_pos))

                # Finner neste tilstand basert på random handling
                next_state, action = self.get_next_state_mc(state)

                # Vi skal finne beste rute. Dersom man treffer en vegg vil ikke tilstanden endres.
                # Derfor blir det tatt en sjekk om tilstand er endret. Oppdaterer kun dersom endret.
                if not (state == next_state):
                    reward = self.reward_matrix[state][action]
                    total_reward += reward
                    print(f"State: {state} Next State: {next_state} Reward: {reward} Total Reward: {total_reward}")
                    state = next_state
                    current_route.append((self.x_pos, self.y_pos))

                #state = next_state
            
            
            self.reset_pos(start_pos)

            if total_reward > best_reward:
                best_reward = total_reward
                best_route = current_route

            epochs -= 1

        self.running = False
        #self.reset_pos(start_pos)
        return best_route, best_reward

    def q_learning(self, epochs, start_pos, goal_pos, policy):

        # Reset q og visited matrix
        self.reset_q_matrix()
        self.visited_matrix_reset()

        route = []
        route.append([start_pos['X'], start_pos['Y']])
        # Get route
        goal_state = self.get_state(x_pos=goal_pos['X'], y_pos=goal_pos['Y'])
        state = self.get_state(x_pos=start_pos['X'], y_pos=start_pos['Y'])

        counter = 0
        while epochs > 0:
            self.one_step_q_learning(policy)
            if self.has_reached_goal(goal_pos):

                epochs -= 1

        for x in range(1, 37):
            print(f"State {x}: {self.q_matrix[x]}:")

        done = False
        steps = 0
        while not done:
            x,y = self.get_next_state_greedy(state)
            route.append([x, y])
            state = self.get_state(x_pos=x, y_pos=y)
            if state == goal_state:
                done = True
            steps += 1
            if steps > 100:
                done = True
                route = []
        self.running = False
        return route

        
    def one_step_q_learning(self, policy):
        # Get action based on policy
        # Get the next state based on the action
        # Get the reward for going to this state
        # Update the Q-matrix
        # Go to the next state
        if policy == 'Greedy':
            action = random.randint(0,3)
            current_state = self.get_state()
            if action == 0: # Opp
                if self.y_pos != 1:
                    self.y_pos -= 1
                    next_state = self.get_state()
                else:
                    next_state = current_state
                self.reward_update(current_state, action, next_state)
                
                
            if action == 1: # Ned
                if self.y_pos != 6:
                    self.y_pos += 1
                    next_state = self.get_state()
                else:
                    next_state = current_state
                self.reward_update(current_state, action, next_state)
                
            if action == 2: # Venstre
                if self.x_pos != 1:
                    self.x_pos -= 1
                    next_state = self.get_state()
                else:
                    next_state = current_state
                self.reward_update(current_state, action, next_state)
                
            if action == 3: # Høyre
                if self.x_pos != 6:
                    self.x_pos += 1
                    next_state = self.get_state()
                else:
                    next_state = current_state
                self.reward_update(current_state, action, next_state)
        elif policy == 'Epsilon':
            current_state = self.get_state()
            next_state, action = self.get_next_state_eg(current_state)
            self.reward_update(current_state, action, next_state)


    
    def has_reached_goal(self, goal: dict) -> bool:
        """Returnerer true/false basert på input fra goal dict (Inneholder 'X' og 'Y')
        
        Keyword arguments:
        goal -- Mål dictionary. Skal inneholde 'X' og 'Y'
        Return: Bool
        """
        if (self.x_pos == goal['X']) and (self.y_pos == goal['Y']):
            return True
        else:
            return False
        
    def reset_random(self):
        """
            Setter roboten i en tilfeldig posisjon i kordinatsystemet.
        """
        self.x_pos = random.randint(1, 6)
        self.y_pos = random.randint(1, 6)


    def reset_q_matrix(self):
        """
            Reset quality-matrisen.
        """
        self.q_matrix = {i: [0] * 4 for i in range(1, 37)}

    def visited_matrix_reset(self):
        """
            Reset av visited matrisen.

            Denne er bygd opp med +1 i alle retninger.
            Dette er for å legge vegger til som "visited"
        """
        self.visited = {i: {u: False for u in range(0, 8)} for i in range(0, 8)}

        # Legger til visited i alle "vegger"
        # TOPP/BUNN
        for x in range(6):
            self.visited[x][0] = True
            self.visited[x][7] = True
        # HØYRE/VENSTRE
        for y in range(6):
            self.visited[0][y] = True
            self.visited[7][y] = True

    def get_state_cord(self, state):
        """Henter kordinater basert på state.
        
        Keyword arguments:
        state - Hvilken tilstand agenten er i.
        Return: Returnerer x og y kordinater
        """
        x = (state - 1)%6 + 1
        y = (state - 1)//6 + 1
        return x, y
    
    def get_state(self, x_pos=None, y_pos=None):
        """ Henter ut hvilken state man er i basert på
            kordinater
        
        Keyword arguments:
        y_pos - X posisjon i kordinatsystemet. Dersom None, bruker x_pos til agent.
        x_pos - Y posisjon i kordinatsystemet. Dersom None, bruker y_pos til agent.
        Return: return_description
        """
        y_pos = y_pos if y_pos is not None else self.y_pos
        x_pos = x_pos if x_pos is not None else self.x_pos
        return (y_pos-1)*6 + x_pos
    
    def reset_pos(self, start_pos):
        self.x_pos = start_pos['X']
        self.y_pos = start_pos['Y']

    def pos_move(self, state, action):
        x,y = self.get_state_cord(state)
        if action == 0: #Opp
            if y != 1:
                y -= 1 
        if action == 1: #Ned
            if y != 6:
                y +=1
        if action == 2: #Venstre
            if x != 1:
                x -= 1
        if action == 3: #Høyre
            if x != 6:
                x+=1
        return x, y
    

    def get_next_state_greedy(self, state):
        x,y = self.get_state_cord(state)

        # Legger de som er besøkt til en visited liste.
        # Disse skal ikke besøkes igjen.
        self.visited[x][y] = True
        # print(self.visited)
        
        # Henter neste state fra q-matrix
        max_reward = max(self.q_matrix[state])
        action = self.q_matrix[state].index(max_reward)

        next_step_found = False
        action_locked = [False for _ in range(4)]
        while next_step_found == False:
            max_reward = max(self.q_matrix[state])
            action = self.q_matrix[state].index(max_reward)
            # visited check
            x, y = self.pos_move(state, action)
            if self.visited[x][y]:
                # Dersom man prøver å besøke samme step flere ganger, legger til en straff.
                #print("VISITED")
                self.q_matrix[state][action] -= self.step_visited_punishment
                action_locked[action] = True
                #time.sleep(1)
            else:
                next_step_found = True
                    
            if action_locked[0] and action_locked[1] and action_locked[2] and action_locked[3]:
                r_num = random.randint(0,3)
                x, y = self.pos_move(state, r_num)
                next_step_found = True
        return x,y
    
    def reward_update(self, current_state, action, next_state):
        self.q_matrix[current_state][action] = (1 - self.alpha) * self.q_matrix[current_state][action]\
                                                + self.alpha * (self.reward_matrix[current_state][action]\
                                                + self.gamma * max(self.q_matrix[next_state]))


# Feel free to add additional classes / methods / functions to solve the assignment...