import sys
import time
import pygame
from pygame.locals import *
from robot import Robot
from radiobutton import RadioButton, RadioGroup
pygame.font.init()

# Add colors as needed.
GREEN_COLOR = pygame.Color(0, 255, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)

if __name__ == "__main__":
    pygame.init()
    fps_clock = pygame.time.Clock()

    play_surface = pygame.display.set_mode((500, 600))
    pygame.display.set_caption('Karaktersatt Oppgave 1 DTE2602')
    simulator_speed = 10 # Adjust this value to change the speed of the visualiztion. Bigger number = more faster...

    bg_image = pygame.image.load("grid.jpg").convert() # Loads the simplified grid image.
    bg_transp_image = pygame.image.load("map.jpg").convert_alpha()
    bg_transp_image.set_alpha(200)

    robot = Robot() # Create a new robot.
    robot.reset_random()

    # CUSTOM:
    #Checkbox for hvilken policy som skal kjøres
    radio_group = RadioGroup()
    radio_mc = RadioButton(play_surface, 10, 510, 20, 20, GREEN_COLOR, BLACK_COLOR, radio_group, 'MC')
    radio_greedy = RadioButton(play_surface, 10, 540, 20, 20, GREEN_COLOR, BLACK_COLOR, radio_group, 'Greedy')
    radio_epsilon = RadioButton(play_surface, 10, 570, 20, 20, GREEN_COLOR, BLACK_COLOR, radio_group, 'Epilson')

    # Epochs/Episodes
    epoch_input = ''
    epoch_font = pygame.font.Font('freesansbold.ttf', 40)
    # epoch_text = epoch_font.render(epoch_input, True, (0, 255, 0), BLACK_COLOR)
    # epoch_textRect = epoch_text.get_rect()
    # epoch_textRect.bottomleft = (400, 500)

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
    goal_pos = {'X': 0, 'Y': 5}

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
                    robot.reset_random()
                    policy_running = True
                    robot.running = True
                    print(f"RUNNING WITH: {epoch_input} episodes.")

                # https://stackoverflow.com/questions/20842801/how-to-display-text-in-pygame
                # epoch_text = epoch_font.render(epoch_input, True, (0, 255, 0), (0, 0, 128))
                # epoch_textRect = epoch_text.get_rect()
                # epoch_textRect.bottomleft = (400, 500)
                # play_surface.blit(epoch_text, epoch_textRect)
                
                
            # CUSTOM
            if event.type == pygame.MOUSEBUTTONDOWN:
                radio_mc.update(event)
                radio_greedy.update(event)
                radio_epsilon.update(event)


        play_surface.fill(WHITE_COLOR) # Fill the screen with white.
        play_surface.blit(bg_image, (0, 0)) # Render the background image.
        play_surface.blit(bg_transp_image, (0, 0)) # Rendrer map bildet over grid (transparent)

        # Render the robot over the image.
        pygame.draw.rect(play_surface, BLACK_COLOR, Rect(robot.get_x() * 70 + 69, robot.get_y() * 70 + 69, 22, 22)) # A black outline.
        pygame.draw.rect(play_surface, GREEN_COLOR, Rect(robot.get_x() * 70 + 70, robot.get_y() * 70 + 70, 20, 20)) # The robot is rendered in green, you may change this if you want.

        #Radiobutton render
        radio_mc.render_checkbox()
        radio_greedy.render_checkbox()
        radio_epsilon.render_checkbox()

        #Epoch input code
        epoch_text = epoch_font.render(epoch_input, True, BLACK_COLOR, WHITE_COLOR)
        epoch_text_rect = epoch_text.get_rect()
        epoch_text_rect.bottomright = (480, 550)
        play_surface.blit(epoch_text, epoch_text_rect)
        play_surface.blit(epoch_d_text, epoch_d_text_rect)
        play_surface.blit(run_d_text, run_d_text_rect)
        

        # Calls related to Q-learning.
        if policy_running:
            if robot.has_reached_goal(goal_pos) or not robot.running:
                policy_running = False
            else:
                robot.one_step_q_learning(radio_group.get_active())

        # Refresh the screen.
        pygame.display.flip()
        fps_clock.tick(simulator_speed)
