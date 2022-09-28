import numpy as np
import convolution


def generate_gauss_kernel(size=5, sigma=1):
    center = round((size/2), 0)
    kernel = np.zeros((size, size))

    for i in range(size):
       for j in range(size):
          diff = np.sqrt((i - center)**2 + (j - center)**2)
          kernel[i,j] = np.exp(-(diff**2) / (2*sigma**2))
    return kernel/np.sum(kernel)

def gaussian_blur(image_array, kernel_size=5, sigma=1):
	if kernel_size > 8:
		print("Kernel size too big, please keep it below 8")
		return

	kernel = generate_gauss_kernel(kernel_size, sigma)
	blurred_image = convolution.conv_2d(image_array, kernel)
	
	return blurred_image
