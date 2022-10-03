import math
import numpy as np


def draw_line_on_image(image, x0, y0, x1, y1):
    line_pixels = []

    if  abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            line_pixels = get_line_points_down(x1, y1, x0, y0)
        else:
            line_pixels = get_line_points_down(x0, y0, x1, y1)
    else:
        if y0 > y1:
            line_pixels = get_line_points_up(x1, y1, x0, y0)
        else:
            line_pixels = get_line_points_up(x0, y0, x1, y1)

    for pix in line_pixels:
        image[pix[0], pix[1]] = 0


def get_line_points_down(x0, y0, x1, y1):
    pixels = []

    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    
    if dy < 0:
        yi = -1
        dy = -dy
    
    D = (2 * dy) - dx
    y = y0

    for x in range(x0, x1):
        pixels.append((x, y))

        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy

    return pixels

def get_line_points_up(x0, y0, x1, y1):
    pixels = []

    dx = x1 - x0
    dy = y1 - y0
    xi = 1

    if dx < 0:
        xi = -1
        dx = -dx

    D = (2 * dx) - dy
    x = x0

    for y in range(y0, y1):
        pixels.append((x, y))

        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2*dx

    return pixels
