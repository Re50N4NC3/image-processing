import numpy as np
import line_drawing


# TODO refactor into two functions, one that draws image with lines, other that returns line points
def line_detection_vectorized(image, edge_image, draw=True, num_rhos=180, num_thetas=180, t_count=500):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    
    accumulator = np.zeros((len(rhos), len(rhos)))
    edge_points = np.argwhere(edge_image != 0)
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
    
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    
    accumulator, theta_vals, rho_vals = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )
    
    accumulator = np.transpose(accumulator)
    lines = np.argwhere(accumulator > t_count)
    rho_idxs, theta_idxs = lines[:, 0], lines[:, 1]
    r, t = rhos[rho_idxs], thetas[theta_idxs]
    
    a_thresh = 1
    b_thresh = 1
    
    it = 0
    line_points_final = []
    
    for line in lines:
        y, x = line
        rho = rhos[y]
        theta = thetas[x]
        
        if it == 0:
            it = 1
            a_thresh = rho
            b_thresh = theta
        
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
            
        if a_thresh == 0:
            a_thresh += 0.001
            
        if b_thresh == 0:
            b_thresh += 0.001
        
        at = abs(rho / a_thresh)
        bt = abs(theta / b_thresh)
        
        if (at > 0.5 and at < 1.5) and (bt > 0.5 and bt < 1.5):
            it += 1
        else:
            x0 = (a * rho) + edge_width_half
            y0 = (b * rho) + edge_height_half
            x0 = int(x0 + 1000 * (-b))
            y0 = int(y0 + 1000 * (a))
            x1 = int(x0 - 1000 * (-b))
            y1 = int(y0 - 1000 * (a))
                
            line_points_final.append((y0, y1,x0, x1))
        
            a_thresh = rho
            b_thresh = theta
    return line_points_final

    # if draw:
    #     for line in line_points_final:
    #         image_with_lines = line_drawing.draw_line_on_image(image_with_lines, line[0], line[1], line[2], line[3], 2)
    #     return image_with_lines

    # return np.copy(image)
    