import sys
import time
import pygame
import math
from pygame.locals import *
from robot import Robot
from radiobutton import RadioButton, RadioGroup
pygame.font.init()

# Add colors as needed.
GREEN_COLOR = pygame.Color(0, 255, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)
RED_COLOR = pygame.Color(255, 0, 0)

if __name__ == "__main__":
    pygame.init()
    fps_clock = pygame.time.Clock()

    play_surface = pygame.display.set_mode((500, 600))
    pygame.display.set_caption('Karaktersatt Oppgave 1 DTE2602')
    simulator_speed = 1000 # Adjust this value to change the speed of the visualiztion. Bigger number = more faster...

    bg_image = pygame.image.load("grid.jpg").convert() # Loads the simplified grid image.
    bg_transp_image = pygame.image.load("map.jpg").convert_alpha()
    bg_transp_image.set_alpha(200)

    robot = Robot() # Create a new robot.
    #robot.reset_random()

    # CUSTOM:
    #Checkbox for hvilken policy som skal kjøres
    radio_group = RadioGroup()
    radio_mc = RadioButton(play_surface, 10, 510, 20, 20, GREEN_COLOR, BLACK_COLOR, radio_group, 'MC')
    radio_q_learning = RadioButton(play_surface, 10, 540, 20, 20, GREEN_COLOR, BLACK_COLOR, radio_group, 'Q-Learning')
    radio_epsilon = RadioButton(play_surface, 10, 570, 20, 20, GREEN_COLOR, BLACK_COLOR, radio_group, 'Epsilon')

    # Epochs/Episodes
    epoch_input = ''
    epoch_font = pygame.font.Font('freesansbold.ttf', 40)
    # epoch_text = epoch_font.render(epoch_input, True, (0, 255, 0), BLACK_COLOR)
    # epoch_textRect = epoch_text.get_rect()
    # epoch_textRect.bottomleft = (400, 500)
    epoch_number = 0

    # Button

    # Forklarende tekst.
    epoch_description = "Bruk knappene 1-9 og backspace for å sette antall episoder."
    epoch_description_font = pygame.font.Font('freesansbold.ttf', 10)
    epoch_d_text = epoch_description_font.render(epoch_description, True, BLACK_COLOR, WHITE_COLOR)
    epoch_d_text_rect = epoch_d_text.get_rect()
    epoch_d_text_rect.bottomright = (480, 510)

    run_description = "Trykk ENTER for å kjøre."
    run_description_font = pygame.font.Font('freesansbold.ttf', 10)
    run_d_text = run_description_font.render(run_description, True, BLACK_COLOR, WHITE_COLOR)
    run_d_text_rect = run_d_text.get_rect()
    run_d_text_rect.bottomright = (480, 590)

    # Setter at ingen policy kjører når man starter opp
    policy_running = False
    goal_pos = {'X': 1, 'Y': 6}
    start_pos = {'X': 4, 'Y': 1}
    route = []

    # Monte Carlo reward visning
    reward = 0

    # Pygame boilerplate code.
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    print(epoch_input)
                    pygame.event.post(pygame.event.Event(QUIT))
                    running = False
                    break
                #CUSTOM
                # https://stackoverflow.com/questions/16044229/how-to-get-keyboard-input-in-pygame
                # https://www.geeksforgeeks.org/how-to-get-keyboard-input-in-pygame/
                if event.key == K_0 or event.key == K_KP0:
                    epoch_input += '0'
                if event.key == K_1 or event.key == K_KP1:
                    epoch_input += '1'
                if event.key == K_2 or event.key == K_KP2:
                    epoch_input += '2'
                if event.key == K_3 or event.key == K_KP3:
                    epoch_input += '3'
                if event.key == K_4 or event.key == K_KP4:
                    epoch_input += '4'
                if event.key == K_5 or event.key == K_KP5:
                    epoch_input += '5'
                if event.key == K_6 or event.key == K_KP6:
                    epoch_input += '6'
                if event.key == K_7 or event.key == K_KP7:
                    epoch_input += '7'
                if event.key == K_8 or event.key == K_KP8:
                    epoch_input += '8'
                if event.key == K_9 or event.key == K_KP9:
                    epoch_input += '9'
                if event.key == K_BACKSPACE:
                    epoch_input = epoch_input[:-1]
                #epoch_font.render(epoch_input, True, (0, 255, 0), (0, 0, 128))
                if event.key == K_RETURN:
                    # policy_running = True
                    # robot.running = True
                    # robot.reset_q_matrix()
                    # robot.visited_matrix_reset()
                    # robot.mc_steps = []
                    # robot.mc_total_reward = -math.inf
                    # route = []
                    if epoch_input == '':
                        epoch_number = 0
                    else:
                        epoch_number = int(epoch_input)
                        #robot.reset_random()
                    # robot.start(epoch_number,start_pos,goal_pos, radio_group.get_active())
                    # running_text = pygame.font.Font('freesansbold.ttf', 20).render("RUNNING", True, BLACK_COLOR, None)
                    # running_rect = running_text.get_rect()
                    # running_rect.center = (500/2, 600/2)
                    # play_surface.blit(running_text, running_rect)
                    robot.running = True


                # https://stackoverflow.com/questions/20842801/how-to-display-text-in-pygame
                # epoch_text = epoch_font.render(epoch_input, True, (0, 255, 0), (0, 0, 128))
                # epoch_textRect = epoch_text.get_rect()
                # epoch_textRect.bottomleft = (400, 500)
                # play_surface.blit(epoch_text, epoch_textRect)
                
                
            # CUSTOM
            if event.type == pygame.MOUSEBUTTONDOWN:
                radio_mc.update(event)
                radio_q_learning.update(event)
                radio_epsilon.update(event)


        play_surface.fill(WHITE_COLOR) # Fill the screen with white.
        play_surface.blit(bg_image, (0, 0)) # Render the background image.
        play_surface.blit(bg_transp_image, (0, 0)) # Rendrer map bildet over grid (transparent)

        # Render the robot over the image.
        pygame.draw.rect(play_surface, BLACK_COLOR, Rect(robot.get_x() * 70 + 69, robot.get_y() * 70 + 69, 22, 22)) # A black outline.
        pygame.draw.rect(play_surface, GREEN_COLOR, Rect(robot.get_x() * 70 + 70, robot.get_y() * 70 + 70, 20, 20)) # The robot is rendered in green, you may change this if you want.

        #Radiobutton render
        radio_mc.render_checkbox()
        radio_q_learning.render_checkbox()
        radio_epsilon.render_checkbox()

        #Epoch input code
        epoch_text = epoch_font.render(epoch_input, True, BLACK_COLOR, WHITE_COLOR)
        epoch_text_rect = epoch_text.get_rect()
        epoch_text_rect.bottomright = (480, 550)
        play_surface.blit(epoch_text, epoch_text_rect)
        play_surface.blit(epoch_d_text, epoch_d_text_rect)
        play_surface.blit(run_d_text, run_d_text_rect)


        if robot.running:
            running_text = pygame.font.Font('freesansbold.ttf', 20).render("RUNNING", True, BLACK_COLOR, None)
            running_rect = running_text.get_rect()
            running_rect.center = (500/2, 600/2)
            play_surface.blit(running_text, running_rect)
            pygame.display.flip()
            
            #robot.start(epoch_number,start_pos,goal_pos, radio_group.get_active())
            if radio_group.get_active() == 'MC':
                route, reward = robot.monte_carlo_exploration(epoch_number,start_pos,goal_pos)
            elif radio_group.get_active() == 'Q-Learning':
                route = robot.q_learning(epoch_number,start_pos,goal_pos, 'Q-Learning')
            elif radio_group.get_active() == 'Epsilon':
                route = robot.q_learning(epoch_number,start_pos,goal_pos, 'Epsilon')
                #route, reward = robot.greedy_path(goal_pos)


        if route != []:
            #if radio_group.get_active() == 'MC':
            for step_number, step in enumerate(route, start=1):
                pygame.draw.rect(play_surface, BLACK_COLOR, Rect((step[0]-1) * 70 + 69, (step[1]-1) * 70 + 69, 22, 22)) # A black outline.
                pygame.draw.rect(play_surface, RED_COLOR, Rect((step[0]-1) * 70 + 70, (step[1]-1) * 70 + 70, 20, 20))
                step_number_text = pygame.font.Font('freesansbold.ttf', 20).render(str(step_number), True, BLACK_COLOR, None)
                step_number_rect = step_number_text.get_rect()
                step_number_rect.topleft = ((step[0]-1) * 70 + 69, (step[1]-1) * 70 + 69)
                play_surface.blit(step_number_text, step_number_rect)
                
                if radio_group.get_active() == 'MC':
                    reward_text = pygame.font.Font('freesansbold.ttf', 20).render(f"Reward {reward}", True, BLACK_COLOR, None)
                    reward_rect = reward_text.get_rect()
                    reward_rect.center = (500/2, 600/2)
                    play_surface.blit(reward_text, reward_rect)

# BACKUP
        # if route != []:
        #     if radio_group.get_active() == 'MC':
        #         for step_number, step in enumerate(route, start=1):
        #                 pygame.draw.rect(play_surface, BLACK_COLOR, Rect((step[0]-1) * 70 + 69, (step[1]-1) * 70 + 69, 22, 22)) # A black outline.
        #                 pygame.draw.rect(play_surface, RED_COLOR, Rect((step[0]-1) * 70 + 70, (step[1]-1) * 70 + 70, 20, 20))
        #                 step_number_text = pygame.font.Font('freesansbold.ttf', 20).render(str(step_number), True, BLACK_COLOR, None)
        #                 step_number_rect = step_number_text.get_rect()
        #                 step_number_rect.topleft = ((step[0]-1) * 70 + 69, (step[1]-1) * 70 + 69)
        #                 play_surface.blit(step_number_text, step_number_rect)

        #                 reward_text = pygame.font.Font('freesansbold.ttf', 20).render(f"Reward {reward}", True, BLACK_COLOR, None)
        #                 reward_rect = reward_text.get_rect()
        #                 reward_rect.center = (500/2, 600/2)
        #                 play_surface.blit(reward_text, reward_rect)
        

        

        # Calls related to Q-learning.
        # if policy_running:
        #     if robot.has_reached_goal(goal_pos, radio_group.get_active()) or not robot.running:
        #         epoch_number -= 1
        #         robot.reset_random()
        #     else:
        #         robot.one_step_q_learning(radio_group.get_active())
        #     if epoch_number == 0:
        #         print("Doing something...")
        #         route = robot.get_route(start_pos, goal_pos, radio_group.get_active())
        #         print(route)
        #         print("Done doing something...")
                
        # if epoch_number == 0:   
        #         policy_running = False
                #robot.get_route(start_pos, goal_pos, radio_group.get_active())

        # if route != []:
        #     for step_number, step in enumerate(route, start=1):
        #             pygame.draw.rect(play_surface, BLACK_COLOR, Rect(step[0] * 70 + 69, step[1] * 70 + 69, 22, 22)) # A black outline.
        #             pygame.draw.rect(play_surface, RED_COLOR, Rect(step[0] * 70 + 70, step[1] * 70 + 70, 20, 20))
        #             step_number_text = pygame.font.Font('freesansbold.ttf', 20).render(str(step_number), True, BLACK_COLOR, None)
        #             step_number_rect = step_number_text.get_rect()
        #             step_number_rect.topleft = (step[0] * 70 + 69, step[1] * 70 + 69)
        #             play_surface.blit(step_number_text, step_number_rect)

        
                

        # Refresh the screen.
        pygame.display.flip()
        fps_clock.tick(simulator_speed)
