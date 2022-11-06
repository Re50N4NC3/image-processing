from PIL import Image
from monochrome import rgb_to_mono
import edge_detection_sobel
import thresholding
import numpy as np
import filters
import line_drawing
import line_detection


def open_image_as_array(file_name):
	try:
		img = Image.open(file_name)
		return np.array(img)
	except FileNotFoundError:
		print("File doesn't exist")
		return

def show_image_from_array(image_array, save_name='show.png', remove_alpha=True):
	if remove_alpha:
		if len(np.shape(image_array)) > 2:
			image_array = image_array[:,:,:3]

	try:
		img = Image.fromarray(image_array, 'RGB')
	except ValueError:
		print("Image is not RGB, showing as greyscale")
		img = Image.fromarray(image_array, 'L')
		
	img.save(save_name)
	img.show()


if __name__ == "__main__":
	sample_image = open_image_as_array('test_square.png')
	# blurred_image = blur.gaussian_blur(sample_image, 5, 12)
	
	monochrome_image = rgb_to_mono(sample_image)
	## show_image_from_array(monochrome_image)
	thrsh = thresholding.threshold_adaptive(monochrome_image)
	edge_image = edge_detection_sobel.sobel_edge_detection(thrsh, 0)
	## show_image_from_array(blurred_image)

	##line_drawing.draw_line_on_image(sample_image, 20, 10, -300, 6000, 4)
	## line_image_points = line_detection.line_detection_vectorized(sample_image, edge_image)
	#line_detection.draw_image_lines(sample_image, edge_image,4,180,180,600)
	pixelated = filters.pixelate(sample_image,8)
	show_image_from_array(pixelated)
