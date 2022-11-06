import numpy as np
import convolution


def generate_chunks(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(A)

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

def pixelate(image, pixel_size=4):
    pixelated_image = np.zeros(np.shape(image))
    image_dimensions = np.shape(image)[2]

    w = int(np.shape(image)[0] / pixel_size)
    h = int(np.shape(image)[1] / pixel_size)

    for x in range(0, w):
        for y in range(0,h):
            x_step = x * pixel_size
            y_step = y * pixel_size

            mean_color = np.mean(image[x_step:x_step+pixel_size, y_step:y_step+pixel_size], axis=(0, 1)).astype(int)
            mean_chunk = np.full((pixel_size, pixel_size,image_dimensions), mean_color)

            print(mean_color)

            pixelated_image[x_step:x_step+pixel_size, y_step:y_step+pixel_size] = np.copy(mean_color)

    return pixelated_image