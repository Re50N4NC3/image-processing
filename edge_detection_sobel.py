import numpy as np


def gen_idx_conv_1d(in_size, ker_size):
    # generate indices corresponding to 1D input tensor
    f = lambda dim1, dim2, axis: np.reshape(np.tile(np.expand_dims(np.arange(dim1),axis),dim2),-1)
    
    out_size = in_size - ker_size+1
    out_list = f(ker_size, out_size, 0) + f(out_size, ker_size, 1)
    
    return out_list 


def repeat_idx_2d(idx_list, nbof_rep, axis):
    # repeat 1D indices assertion throughout axis
    tile_axis = (nbof_rep, 1) if axis else (1, nbof_rep)
    
    return np.reshape(np.tile(np.expand_dims(idx_list, 1), tile_axis), -1)


def conv_2d(img, ker):
	# if the image is a 2D array it is reshaped by expanding with dimension w
	if len(img.shape) == 2:
		img = np.expand_dims(img, -1)

	img_x, img_y, img_w = img.shape

	# if the kernel is a 2D array, it is reshaped so it will be applied to all of the image channels
	if len(ker.shape) == 2: 
		ker = np.tile(np.expand_dims(ker, -1),[1, 1, img_w]) # the same kernel will be applied to all of the channels

	# check if kernel and image dimensions match
	# if base image is monochrome 2D and kernel is 2D, it should be fine
	if ker.shape[-1] != img.shape[-1]:
		print("Kernel and image dimensions do not match")

	ker_x = ker.shape[0]
	ker_y = ker.shape[1]

	# shape of the output image
	out_x = img_x - ker_x + 1 
	out_y = img_y - ker_y + 1

	# reshape the image to (out_x, ker_x, out_y, ker_y, im_w)
	# 1D indices assertion
	idx_list_x = gen_idx_conv_1d(img_x, ker_x)
	idx_list_y = gen_idx_conv_1d(img_y, ker_y)

	# go along axes
	idx_reshaped_x = repeat_idx_2d(idx_list_x, len(idx_list_y), 0)
	idx_reshaped_y = repeat_idx_2d(idx_list_y, len(idx_list_x), 1)

	# reshape the image
	img_reshaped = np.reshape(img[idx_reshaped_x, idx_reshaped_y, :], [out_x, ker_x, out_y, ker_y, img_w])

	# reshape the 2D kernel
	ker = np.reshape(ker,[1, ker_x, 1, ker_y, img_w])

	# apply kernel to the image
	img_sum = np.sum(img_reshaped*ker, axis=(1,3))

	# return back to initial dimensions
	img_sqz = np.squeeze(img_sum)

	return img_sqz.astype(np.uint8)


def sobel_edge_detection(image, fltr=0):
	# define kernels for edge detection
	if fltr == 0:
		kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # vertical lines
	decompose_x = conv_2d(image, kernel)
	square_x = np.square(decompose_x)
    
    # flip kernel for horizontal lines
	decompose_y = conv_2d(image, np.flip(kernel.T, axis=0))
	square_y = np.square(decompose_y)
    
    # calculate gradient magnitude
	grad_magnitude = np.sqrt(square_x + square_y)
	grad_max = grad_magnitude.max()
    
    # sometimes max is 0
	if grad_max != 0:
		grad_magnitude *= 255.0 / grad_max
    
	grad_magnitude = grad_magnitude.astype(np.uint8)
    
	return grad_magnitude
