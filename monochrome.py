import numpy as np


def rgb_to_mono(rgb_array):
	try:
		# return dot product of the array scaled by balancing values using itu R 601 2 luma transform values
		return np.dot(rgb_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
	except ValueError:
		print("Provided argument is not valid, 3 dimensional numpy array")
