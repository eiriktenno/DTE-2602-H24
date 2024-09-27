import pygame
import sys

# Initialiser Pygame
pygame.init()

# Definer farger
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Definer skjermstørrelse
screen_size = (400, 300)
screen = pygame.display.set_mode(screen_size)

# Gi vinduet en tittel
pygame.display.set_caption("Enkelt Pygame Eksempel")

# Definer posisjon og størrelse for firkanten
rect_x = 50
rect_y = 50
rect_width = 50
rect_height = 50
rect_speed_x = 3
rect_speed_y = 3

# Hovedløkke
running = True
while running:
    # Håndter hendelser
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Beveg firkanten
    rect_x += rect_speed_x
    rect_y += rect_speed_y

    # Sjekk for kollisjoner med skjermkanten
    if rect_x > screen_size[0] - rect_width or rect_x < 0:
        rect_speed_x = -rect_speed_x
    if rect_y > screen_size[1] - rect_height or rect_y < 0:
        rect_speed_y = -rect_speed_y

    # Fyll skjermen med hvit farge
    screen.fill(WHITE)

    # Tegn firkanten
    pygame.draw.rect(screen, BLUE, [rect_x, rect_y, rect_width, rect_height])

    # Oppdater skjermen
    pygame.display.flip()

    # Kontroller hastighet på loopen
    pygame.time.Clock().tick(60)

# Avslutt Pygame
pygame.quit()
sys.exit()