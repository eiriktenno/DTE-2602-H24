import pygame

class RadioButton:
    """sumary_line
    Baser på: https://stackoverflow.com/questions/38551168/radio-button-in-pygame
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    def __init__(self, surface, x, y, width, height, color_checked, color_unchecked) -> None:
        self.surface = surface
        self.radio_obj = pygame.Rect(x, y, width, height)
        self.color_unchecked = color_unchecked
        self.color_checked = color_checked
        self.checked = False

    def render_checkbox(self):
        if self.checked:
            pygame.draw.rect(self.surface, self.color_checked, self.radio_obj)
        else:
            pygame.draw.rect(self.surface, self.color_unchecked, self.radio_obj)

    def update(self, event):
        # Henter ut posisjon til museklikk
        x, y = pygame.mouse.get_pos()
        # Henter ut posisjonene til knappen
        px, py, w, h = self.radio_obj

        if px < x < (px + w) and py < y < (py + h):
            if self.checked:
                self.checked = False
            else:
                self.checked = True
            print(f"Checked: {self.checked}")
        self.render_checkbox()

