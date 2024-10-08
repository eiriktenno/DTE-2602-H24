# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.




import random
import time
import math
class Robot:

    x_pos = 0
    y_pos = 0
    alpha = 0.2 # I et statisk miljø har det liten hensikt å bruke en stor alpha/læringsrate.
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

        self.wall_value = -1000
        self.hill_value = -25
        self.water_value = -50
        self.plain_value = 0
        self.goal_value = 100
        self.step_visited_punishment = 2
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
        self.q_matrix = {i: [-2] * 4 for i in range(1, 37)}
        self.visited = {i: {u: False for u in range(0, 8)} for i in range(0, 8)}

        # LEgge til visited i alle "vegger"
        # TOPP
        for x in range(6):
            self.visited[x][0] = True
            self.visited[x][7] = True
        for y in range(6):
            self.visited[0][y] = True
            self.visited[7][y] = True




    def get_x(self):
        # Return the current column of the robot, should be in the range 0-5.
        return self.x_pos-1


    def get_y(self):
        # Return the current row of the robot, should be in the range 0-5.
        return self.y_pos-1
    
    def reset_q_matrix(self):
        self.q_matrix = {i: [-2] * 4 for i in range(1, 37)}


    def get_next_state_mc(self, state):
        #print(state)
        x,y = self.get_state_pos(state)
        print(f"x: {x} y: {y}")

        # Legger de som er besøkt til en visited liste.
        # Disse skal ikke besøkes igjen.
        self.visited[x][y] = True
        print(self.visited)
        
        # Henter neste state fra q-matrix
        max_reward = max(self.q_matrix[state])
        action = self.q_matrix[state].index(max_reward)

        next_step_found = False
        action_locked = [False for _ in range(4)]
        while next_step_found == False:
            max_reward = max(self.q_matrix[state])
            action = self.q_matrix[state].index(max_reward)

            # visited check
            x, y = self.get_new_mc_pos(state, action)
            if self.visited[x][y]:
                # Dersom man prøver å besøke samme step flere ganger, legger til en straff.
                print("VISITED")
                self.q_matrix[state][action] -= self.step_visited_punishment
                action_locked[action] = True
                #time.sleep(1)
            else:
                next_step_found = True
            for i in range(1, 37):
                print(f"State {i}: {self.q_matrix[i]}:")
            print(f"GET NEXT: X: {x} Y: {y} State: {state} Action {action} Reward: {max_reward}")
            ######### MÅ GJØRE BEGRENSNINGER FOR Å IKKE GÅ GJENNOM VEGGER: VELDIG BUGGED ########################################
            
            all_visited = action_locked[0] and action_locked[1] and action_locked[2] and action_locked[3]
                    
            if all_visited:
                print(action_locked)
                 ######### ISTEDE FOR Å LÅSE NED, PRØVE MED RANDOM ########################################
                return -1
                #raise ValueError("Klarer ikke finne letteste rute, med denne q-matrisen.")

        return x,y

    def get_new_mc_pos(self, state, action):
        x,y = self.get_state_pos(state)
        if action == 0: #Opp
            y -= 1 
        if action == 1: #Ned
            y +=1
        if action == 2: #Venstre
            x -= 1
        if action == 3: #Høyre
            x+=1
        return x,y
            

    def get_next_state_eg(self):
        # Return the next state based on Epsilon-greedy.
        pass

    def get_next_state_greedy(self):
        # Return the next state based on Greedy.
        pass

    def greedy_exploration(self):
        pass

    def epsilon_exploration(self):
        pass

    def monte_carlo_exploration(self):
         # Return the next state based on Monte Carlo.
        action = random.randint(0,3)
        current_state = self.get_state()
        if action == 0: # Opp
            
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
            if self.y_pos != 6:
                self.y_pos += 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)
            
        if action == 2: # Venstre
            #reward = self.reward_matrix[self.y_pos][self.x_pos - 1]

            #self.reward_update(current_state, action, reward)
            if self.x_pos != 1:
                self.x_pos -= 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)
            
        if action == 3: # Høyre
            # reward = self.reward_matrix[self.y_pos][self.x_pos + 1]
            # self.reward_update(current_state, action, reward)

            if self.x_pos != 6:
                self.x_pos += 1
                next_state = self.get_state()
            else:
                next_state = current_state
            self.reward_update(current_state, action, next_state)


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
            self.monte_carlo_exploration()
            # self.get_next_state_mc() # Sette opp en løsning som viser svaret.
        if policy == "Greedy":
            # Denne vil kanskje aldri kunne nå målet med en ren greedy.
            # Bør innføre en max steps.
            self.greedy_exploration()
            self.running = False
        if policy == "Epsilon":
            # Basert på monte carlo
            # Bruker prosentvis epsilon
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

    def get_route(self, start_pos, goal_pos, policy):
        # Skal returnere en liste med posisjonene som blir brukt for å komme til mål.
        route = []
        route.append([start_pos['X']-1, start_pos['Y']-1])
        # Get route
        goal_state = self.get_state(x_pos=goal_pos['X'], y_pos=goal_pos['Y'])
        state = self.get_state(x_pos=start_pos['X'], y_pos=start_pos['Y'])
        print(f"Start state: {self.get_state(x_pos=start_pos['X'], y_pos=start_pos['Y'])} X: {start_pos['X']}, Y: {start_pos['Y']}")

        if policy == "MC":
            done = False
            steps = 0
            while done == False:
                print(state)
                x,y = self.get_next_state_mc(state)
                print(f"X: {x} Y: {y}")
                route.append([x-1, y-1])
                state = self.get_state(x_pos=x, y_pos=y)
                if state == goal_state:
                    done = True
                steps += 1
                if steps > 100:
                    done = True
                    print(route)
                    route = []
                    print("ERROR")
            print("GOAL REACHED.")
            print(route)
        if policy == "Greedy":
            pass
        if policy == "Epsilon":
            pass
        return route
        #route.append([goal_pos['X'], goal_pos['Y']])

        
    def reset_random(self):
        # Place the robot in a new random state.
        import random
        self.x_pos = random.randint(1, 6)
        self.y_pos = random.randint(1, 6)

    def greedy_path(self):
        pass

    def get_state(self, y_pos=None, x_pos=None):
        y_pos = y_pos if y_pos is not None else self.y_pos
        x_pos = x_pos if x_pos is not None else self.x_pos
        return (y_pos-1)*6 + x_pos
        # print(f"State: {(self.y_pos-1)*6 + self.x_pos} X: {self.x_pos} Y: {self.y_pos}")
        #     return (self.y_pos-1)*6 + self.x_pos
    
    def get_state_pos(self, state):
        #print(f"State: {state}")
        # y = state // 6
        # x = state % 6
        # print(f"GET STATE POS:::: State: {state} X: {x} Y: {y+1}")
        # return x, (y+1)
        x = (state - 1)%6 + 1
        y = (state - 1)//6 + 1
        return x, y

    def reward_update(self, current_state, action, next_state):
        #current_state = self.get_state()
            #next_state = self.get_state()
        #x, y = self.get_state_pos(new_state)
        # if x == 0 and y == 5:
        #     print(f"Reward: {self.reward_matrix[x][y]}")


        # self.q_matrix[current_state][action] = \
        #     self.reward_matrix[x+1][y+1] + self.gamma*max(self.q_matrix[self.get_state()])
        
        # print(f"State: {current_state} Action: {action} New: X: {x} Y: {y}")
        #self.q_matrix[current_state][action] = reward + self.gamma*max(self.q_matrix[self.get_state()])

        self.q_matrix[current_state][action] = (1 - self.alpha) * self.q_matrix[current_state][action]\
                                                + self.alpha * (self.reward_matrix2[current_state][action]\
                                                                + self.gamma * max(self.q_matrix[next_state]))
                                                
        
        # self.q_matrix[current_state][next_state] = \
        #     self.reward_matrix[self.x_pos][self.y_pos] + self.gamma*max(self.q_matrix[self.get_state()])

    def get_next_state(self, state, action):
        x,y = self.get_state_pos(state)
        if action == 0: #Opp
            y += 1
        if action == 1: #Ned
            y -=1
        if action == 2: #Venstre
            x -= 1
        if action == 3: #Høyre
            x+=1
        return x,y

# Feel free to add additional classes / methods / functions to solve the assignment...