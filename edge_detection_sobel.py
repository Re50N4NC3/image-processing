import numpy as np
import convolution


def sobel_edge_detection(image, fltr=0):
	# define kernels for edge detection
	# sobel operator
	if fltr == 0:
		kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	# sobel-feldman operator
	if fltr == 1:
		kernel = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
	# sharr operator
	if fltr == 2:
		kernel = np.array([[47, 0, -47], [162, 0, -162], [47, 0, -47]])

    # vertical lines
	decompose_x = convolution.conv_2d(image, kernel)
	square_x = np.square(decompose_x)
    
    # flip kernel for horizontal lines
	decompose_y = convolution.conv_2d(image, np.flip(kernel.T, axis=0))
	square_y = np.square(decompose_y)
    
    # calculate gradient magnitude
	grad_magnitude = np.sqrt(square_x + square_y)
	grad_max = grad_magnitude.max()
    
    # sometimes max is 0
	if grad_max != 0:
		grad_magnitude *= 255.0 / grad_max
    
	grad_magnitude = grad_magnitude.astype(np.uint8)
    
	return grad_magnitude
