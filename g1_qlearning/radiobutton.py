import pygame
pygame.font.init()

# Add colors as needed.
GREEN_COLOR = pygame.Color(0, 255, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)

class RadioButton:
    """sumary_line
    Basert på: https://stackoverflow.com/questions/38551168/radio-button-in-pygame
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    def __init__(self, surface, x, y, width, height, color_checked, color_unchecked, radio_group, text) -> None:
        self.surface = surface
        self.radio_obj = pygame.Rect(x, y, width, height)
        self.color_unchecked = color_unchecked
        self.color_checked = color_checked
        self.checked = False
        self.radio_group = radio_group
        self.radio_group.append_button(self)
        self.x_pos = x
        self.y_pos = y
        self.width = width
        self.height = height
        self.text_raw = text

        # https://www.geeksforgeeks.org/python-display-text-to-pygame-window/
        # https://stackoverflow.com/questions/20842801/how-to-display-text-in-pygame
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.text = self.font.render(self.text_raw, True, BLACK_COLOR, WHITE_COLOR)
        self.textRect = self.text.get_rect()

        self.offset = 2
        self.textRect.bottomleft = (x + width + self.offset, y + height)
        self.surface.blit(self.text, self.textRect)


    def render_checkbox(self):
        if self.checked:
            pygame.draw.rect(self.surface, BLACK_COLOR, pygame.Rect(self.x_pos - 1, self.y_pos - 1, self.width+2, self.height +2)) # Outline når man har checked.
            pygame.draw.rect(self.surface, self.color_checked, self.radio_obj)
        else:
            pygame.draw.rect(self.surface, self.color_unchecked, self.radio_obj)
        self.surface.blit(self.text, self.textRect)

    def update(self, event):
        # Henter ut posisjon til museklikk
        x, y = pygame.mouse.get_pos()
        # Henter ut posisjonene til knappen
        px, py, w, h = self.radio_obj

        if px < x < (px + w) and py < y < (py + h): # Sjekker at museklikket skjer innenfor de spesifikk checkboksene.
            self.radio_group.toggle_button(self) # Kaller på toggle funksjonen som skrur av alle andre radioknapper.
        self.render_checkbox()


class RadioGroup:
    """
    
    """

    def __init__(self):
        self.radio_grp = []

    def append_button(self, button):
        self.radio_grp.append(button)
        if len(self.radio_grp) == 1:
            self.toggle_button(button)

    def toggle_button(self, selected_button):
        for button in self.radio_grp:
            button.checked = False
        selected_button.checked = True

    def get_active(self):
        for button in self.radio_grp:
            if button.checked:
                return button.text_raw
