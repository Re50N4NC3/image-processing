import math
import numpy as np


def draw_line_on_image(image, x0, y0, x1, y1, width: int=1):
    line_pixels = []

    x0 = check_if_point_in_image(image, x0, 0)
    x1 = check_if_point_in_image(image, x1, 0)
    y0 = check_if_point_in_image(image, y0, 1)
    y1 = check_if_point_in_image(image, y1, 1)

    print(x0)
    print(x1)
    print(y0)
    print(y1)

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
    
        if width > 1:
            for x in range(-(width - 1), width - 1):
                for y in range(-(width - 1), width - 1):
                    try:
                        image[pix[0] + x, pix[1] + y] = 0
                    except IndexError:
                        break


def check_if_point_in_image(image, point, direction):
    if point >= np.size(image, direction):
        return np.size(image, direction) - 1
    if point <= 0:
        return 1
    return point


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
