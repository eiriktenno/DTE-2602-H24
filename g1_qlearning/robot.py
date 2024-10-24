# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.

import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

class Robot:
    x_pos = 4
    y_pos = 1
    alpha = 0.4 # I et statisk miljø har det liten hensikt å bruke en stor alpha/læringsrate.
                # Det er ikke noen særlige endringer
                # dynamiske miljø = Høy alpha
                # statiske miljø = lav alpha
    gamma = 0.9 # Discount factor/gamma - Hvor stor tro har agenten på at fremtiden vil bringe noe godt.
                # Stor gamma vil si at vi tror den får en stor belønning i fremtiden.
                # Dersom vi ikke er sikker på om man får en stor belønning, kan vi bruke lav gamma.
                # Tro på høy belønning = Høy gamma
                # Usikker på belønning = Lav gamma
    reward_matrix = []
    q_matrix = []
    running = False

    # Konvergens-sjekk
    # NB!: Dette har ikke blitt implementert
    q_diffs = []
    diff_num = 0.01 # Hvor høy differansen skal være før man sier det er konvergens.
    diff_loops = 10 # Hvor mange ganger på rad man skal få "konvergens" før man slutter av.

    # Epsilon settings
    epsilon = 0.1 # Verdien som blir satt her skal representere hvor høy "random" Eks: 0.1 = 10% random.

    # Greedy
    greedy_max_steps = 1000 # Hvor mange steps man skal kjøre før man bestemmer seg for at greedy ikke finner veien.
                            # Og deretter returnere en tom array.

    # Reward sum
    mc_total_reward = -math.inf
    mc_steps = []

    mc_episode_reward = 0
    mc_episode_steps = []

    # Plotting
    plot_active = False  # True om man vil ha plotting.
    plot_bar = False # Dersom man vil ha bar plot.
                    # Denne vil vise hvor mange som har optimal rute.
    plot_reward = np.array([])
    plot_q_simulations = 100 # Hvor mange q-learning simuleringer skal kjøres og plottes.
    plot_max_reward = 50 # Hva er den største oppnålige rewarden



    def __init__(self):
        """
        Initialiserer robotens belønningsmatrise
        Setter opp verdier for ulike typer terreng (vegg, vann, fjell, slette og mål).
        Initialiserer Q-matrise og en besøksmatrise.
        """
        

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
        """
        Returnerer neste tilstand basert på Monte Carlo-simulering
        
        Keyword arguments:
        current_state -- Nåverende tilstand
        Return: tuple(next_state, action) next_state er den neste tilstanden og action er valgt handling.
        """
        
        x,y = self.get_state_cord(current_state)
        action = random.randint(0,3)
        x,y = self.pos_move(current_state, action)
        self.x_pos = x
        self.y_pos = y
        return self.get_state(y_pos = y, x_pos=x), action

    def get_next_state_eg(self, state:int) -> tuple:
        """
        Returnerer neste tilstand basert på Epsilon-Greedy policy.
        
        Keyword arguments:
        state -- Nåverende tilstand
        Return: tuple(next_state, action) next_state er den neste tilstanden og action er valgt handling.
        """
        
        if random.uniform(0, 1) > self.epsilon:
            return self.get_next_state_mc(state)
        else:
            max_reward = max(self.q_matrix[state])
            action = self.q_matrix[state].index(max_reward)
            x,y = self.pos_move(state, action)
            next_state = self.get_state(x, y)
            return next_state, action

    def monte_carlo_exploration(self, epochs: int, start_pos: dict, goal_pos: dict) -> tuple:
        """
        Utfører Monte Carlo exploration for å finne den beste ruten til målet
        
        Keyword arguments:
        epochs -- Antall episoder
        start_pos - Start posisjon i dict ('X' og 'Y')
        goal_pos - Mål posisjon i dict ('X' og 'Y')
        Return: Route en liste med de posisjonene som blir tatt.
        """
        self.plot_reward = np.array([])
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

            # Legger verdi til plot, dersom det er aktivert
            if self.plot_active:
                self.plot_reward = np.append(self.plot_reward, total_reward)
            
            
            self.reset_pos(start_pos)

            if total_reward > best_reward:
                best_reward = total_reward
                best_route = current_route

            epochs -= 1

        # Dersom plot er aktivert.
        # Plot reward til alle verdiene.
        if self.plot_active:
            self.plot_rewards(self.plot_reward)

        self.running = False
        return best_route, best_reward

    def q_learning(self, epochs_input: int, start_pos: dict, goal_pos: dict, policy: str) -> tuple:
        """
        Utfører Q-learning for å finne den optimale ruten
        
        Keyword arguments:
        epochs_input -- Antall episoder som skal kjøres
        start_pos - Start posisjon i dict ('X' og 'Y')
        goal_pos - Mål posisjon i dict ('X' og 'Y')
        policy - Valgt policy (Q-learning eller Epsilon)
        Return: Returnerer en tuple som inneholder en lsite med steps og total belønning
        """
        
        if self.plot_active:
            simulations = self.plot_q_simulations
        else:
            simulations = 1

        best_scores = np.array([])
        while simulations > 0:
            epochs = epochs_input
            self.reset_pos(start_pos)

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

            done = False
            steps = 0
            q_total_reward = 0
            while not done:
                x, y, q_step_reward = self.get_next_state_greedy(state)
                route.append([x, y])
                q_total_reward += q_step_reward
                state = self.get_state(x_pos=x, y_pos=y)
                if state == goal_state:
                    done = True
                steps += 1
                if steps > 100:
                    done = True
                    route = []
            simulations -= 1
            best_scores = np.append(best_scores, q_total_reward)
        self.running = False
        if self.plot_active:
            self.plot_rewards(best_scores)
        return route, q_total_reward

        
    def one_step_q_learning(self, policy):
        # Get action based on policy
        # Get the next state based on the action
        # Get the reward for going to this state
        # Update the Q-matrix
        # Go to the next state
        self.plot_reward = np.array([])

        if policy == 'Q-Learning':
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
    
    def reset_pos(self, start_pos: int):
        """
        Setter roboten i start posisjon.
        """
        
        self.x_pos = start_pos['X']
        self.y_pos = start_pos['Y']

    def pos_move(self, state: int, action: int) -> tuple:
        """
        Flytter roboten basert på state og action.
        Dersom man ikke prøver å gå i veggen.
        """
        
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
    

    def get_next_state_greedy(self, state: int) -> tuple:
        """
        Returnerer neste tilstand sine kordinater basert på greedy policy
        
        Keyword arguments:
        state -- Den nåværende tilstanden
        Return: tuple med x, y og reward
        """
        

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
        q_step_reward = 0
        while next_step_found == False:
            max_reward = max(self.q_matrix[state])
            action = self.q_matrix[state].index(max_reward)
            # visited check
            x, y = self.pos_move(state, action)
            if self.visited[x][y]:
                self.q_matrix[state][action] -= self.step_visited_punishment
                action_locked[action] = True
            else:
                q_step_reward += self.reward_matrix[state][action]
                next_step_found = True
                    
            if action_locked[0] and action_locked[1] and action_locked[2] and action_locked[3]:
                r_num = random.randint(0,3)
                x, y = self.pos_move(state, r_num)
                q_step_reward += self.reward_matrix[state][action]
                next_step_found = True
        return x,y, q_step_reward
    
    def plot_rewards(self, rewards: np.array) -> None:
        plt.figure(1)
        plt.plot(rewards, marker='o', color='green')

        # Basert på:
        # https://medium.com/@mdnu08/automatically-annotate-the-maximum-value-in-a-plot-created-using-the-python-matplotlib-library-54c43001e39c

        # For å merke den som har høyest verdi
        max_index = np.argmax(rewards)
        max_value = rewards[max_index]
        plt.scatter(max_index, max_value, color='red', s=100, label=f'Høyeste verdi: {max_value:.2f}')
        plt.annotate(f'Max: {max_value}', xy=(max_index, max_value))

        # For å merke den med fårligst score
        min_index = np.argmin(rewards)
        min_value = rewards[min_index]
        plt.scatter(min_index, min_value, color='blue', s=100, label=f'Minste Verdi: {min_value:.2f}')
        plt.annotate(f'Min: {min_value}', xy=(min_index, min_value))

        # Gjennomsnitt
        # Legg til en horisontal linje for gjennomsnittsverdien
        mean_reward = np.mean(rewards)
        plt.axhline(y=mean_reward, color='green', linestyle='--', label=f'Gjennomsnitt: {mean_reward:.2f}')

        # Median
        median_reward = np.median(rewards)
        plt.axhline(y=median_reward, color='purple', linestyle='--', label=f'Median: {median_reward:.2f}')

        plt.title('Rewards')
        plt.xlabel('Indeks')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

        if self.plot_bar:
            plt.figure(2)

            # Teller opp antall som har oppnådd max score
            best_result = np.sum(rewards == self.plot_max_reward)
            not_best_result = len(rewards) - best_result

            best_result_per = (best_result/len(rewards)) * 100
            not_best_result_per = (not_best_result/len(rewards)) * 100

            categories = [f'R: {self.plot_max_reward} \n {best_result_per}%', f'R: !{self.plot_max_reward}\n {not_best_result_per}%']
            counts = [best_result, not_best_result]

            plt.bar(categories, counts, color=['red', 'blue'])
            plt.ylim(0, 100)
            plt.show()


    
    def reward_update(self, current_state, action, next_state):
        """
        Oppdaterer Q-Matrisen basert på Bellman sin Q formel.
        
        Keyword arguments:
        current_state -- Nåværende tilstand
        action -- Handling som ble utført i nåværende tilstand
        next_state -- Den nye tilstanden etter handlingen er utført.
        """
        
        self.q_matrix[current_state][action] = (1 - self.alpha) * self.q_matrix[current_state][action]\
                                                + self.alpha * (self.reward_matrix[current_state][action]\
                                                + self.gamma * max(self.q_matrix[next_state]))
