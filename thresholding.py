import numpy as np


def threshold_naive(img, threshold_val=50.0):
	return (np.around(img / 255) * 255).astype(np.uint8)

def threshold_adaptive(img, threshold_val=5.0):
    # Bradley, Roth technique vectorized
    
    # default window size
    s = np.round(img.shape[1]/8)

    # ensure that s is even for indexing
    s = s + np.mod(s,2)

    # compute integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # define grid of points
    (rows,cols) = img.shape[:2]
    (X,Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    # make into 1D grid of coordinates
    X = X.ravel()
    Y = Y.ravel()

    # access the four corners of each neighbourhood
    x1 = X - s / 2
    x2 = X + s / 2
    y1 = Y - s / 2
    y2 = Y + s / 2

    # cooridantes out of bounds check
    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols-1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows-1

    # ensure coordinates are integer
    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    # count neighbouring pixels
    count = (x2 - x1) * (y2 - y1)

    # compute the row and column coordinates to access
    # each corner of the neighbourhood for the integral image
    f1_x = x2
    f1_y = y2
    
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    
    f3_x = x1-1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    
    f4_x = f3_x
    f4_y = f2_y

    # compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # compute thresholded image
    out = np.ones(rows * cols, dtype=np.bool)
    out[img.ravel() * count <= sums * (100.0 - threshold_val) / 100.0] = False

    # reshape back into 2D grid
    out = 255 * np.reshape(out, (rows, cols))

    # return image array
    return out.astype(np.uint8)
