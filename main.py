from PIL import Image
from monochrome import rgb_to_mono
import edge_detection_sobel
import thresholding
import numpy as np
import blur
import line_drawing


def open_image_as_array(file_name):
	try:
		img = Image.open(file_name)
		return np.array(img)
	except FileNotFoundError:
		print("File doesn't exist")
		return

def show_image_from_array(image_array, save_name='show.png'):
	try:
		img = Image.fromarray(image_array, 'RGB')
	except ValueError:
		img = Image.fromarray(image_array, 'L')
		
	img.save(save_name)
	img.show()


if __name__ == "__main__":
	sample_image = open_image_as_array('ponczek.png')
	## monochrome_image = rgb_to_mono(sample_image)
	## show_image_from_array(monochrome_image)
	## thrsh = thresholding.threshold_adaptive(monochrome_image)
	## edge_image = edge_detection_sobel.sobel_edge_detection(thrsh, 0)
	## blurred_image = blur.gaussian_blur(sample_image, 5, 12)
	## show_image_from_array(blurred_image)

	line_drawing.draw_line_on_image(sample_image, 20, 10, 300, 600, 4)

	show_image_from_array(sample_image)
