import numpy as np
import line_drawing


def line_detection_vectorized(image, edge_image, num_rhos=180, num_thetas=180, t_count=500):
    ## Hough transform
    #TODO give ability to show transform plot
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
    
    accumulator = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )
    
    accumulator = np.transpose(accumulator)
    lines = np.argwhere(accumulator > t_count)
    
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

def draw_image_lines(image, edge_image, line_width=2, num_rhos=180, num_thetas=180, t_count=500):
    line_image_points = line_detection_vectorized(image, edge_image, num_rhos, num_thetas, t_count)
    line_drawing.draw_multiple_lines_on_image(image, line_image_points, line_width)
    
