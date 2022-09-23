from PIL import Image
from monochrome import rgb_to_mono
import numpy as np

def open_image_as_array(file_name):
	try:
		img = Image.open(file_name)
		return np.array(img)
	except ValueError:
		print("File doesn't exist")
		return

def show_image(image_array, save_name='show.png'):
	try:
		img = Image.fromarray(image_array, 'RGB')
	except ValueError:
		img = Image.fromarray(image_array, 'L')
		
	img.save(save_name)
	img.show()


sample_image = open_image_as_array('test.jpg')
monochrome_image = rgb_to_mono(sample_image)
show_image(monochrome_image)
