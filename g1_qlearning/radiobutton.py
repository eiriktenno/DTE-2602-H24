"""
Diverse kilder benyttet for denne klassen:
    https://www.geeksforgeeks.org/python-display-text-to-pygame-window/
    https://stackoverflow.com/questions/20842801/how-to-display-text-in-pygame
"""


import pygame
pygame.font.init()

# Add colors as needed.
GREEN_COLOR = pygame.Color(0, 255, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)


class RadioGroup:
    """
        Dette er gruppen hvor man kan legge til radioknappene.
        Her kan man legge til knapper til gruppen.
        Aktivere de spesifikkke knappene
        Hente ut den aktive knappen
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

class RadioButton:
    """
    Denne klassen lager en radioknapp.
    Her må man legge til knappen i en radio_group, som vil si en gruppe med alle valgene man skal kunne ta.
    Når man trykker på en av knappene i en radiogruppe, så vil de andre bli deaktiverte.

    Basert på: https://stackoverflow.com/questions/38551168/radio-button-in-pygame
    
    Keyword arguments:
    surface         -- pygame sin display surface
    x               -- kordinat x for plassering av knapp
    y               -- kordinat y for plassering av knapp
    width           -- Bredde på knappen
    height          -- Høgd på knappen
    color_checked   -- Farge dersom den er aktiv
    color_unchecked -- Farge dersom den ikke er aktiv
    radio_group     -- Hvilken radiogruppe den skal være knyttet til
    text            -- Hvilken text skal stå vedsiden av knappen

    """
    
    def __init__(self, surface: pygame.display, x: int, y: int, width: int, height: int, color_checked: pygame.color, \
                 color_unchecked: pygame.color, radio_group: RadioGroup, text: str) -> None:
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



