import numpy as np
import cv2
import matplotlib.pyplot as plt

#Load the grayscale image
image = cv2.imread('Lena-size-256x256.png', cv2.IMREAD_GRAYSCALE)

#image resize
image = cv2.resize(image, (256, 256))

#image matrix
image_matrix = image.astype(np.float64)

laplace_grid = np.copy(image_matrix)

#Set Dirichlet Boundary Conditions
laplace_grid[0, :] = 0
laplace_grid[-1, :] = 0
laplace_grid[:, 0] = 0
laplace_grid[:, -1] = 0

#relative error
tolerance = 1e-6
def relative_error(new_grid, old_grid):
    numerator = np.linalg.norm(new_grid - old_grid)
    denominator = np.linalg.norm(old_grid)
    return numerator / (denominator + 1e-10) #add a very small number to avoid division by zero


def jacobi_method(grid, tolerance=1e-6, max_iterations=1000):
    new_grid = np.copy(grid)
    for _ in range(max_iterations):
        old_grid = new_grid.copy()
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                new_grid[i, j] = 0.25 * (old_grid[i+1, j] + old_grid[i-1, j] + old_grid[i, j+1] + old_grid[i, j-1])

        if relative_error(new_grid, old_grid) < tolerance:
            break
    return new_grid

#gauss-siedel method
def gauss_siedel_method(grid, tolerance=1e-6, max_iterations=1000):
    new_grid = np.copy(grid)
    for _ in range(max_iterations):
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                new_grid[i, j] = 0.25 * (new_grid[i+1, j] + new_grid[i-1, j] + new_grid[i, j+1] + new_grid[i, j-1])

        if relative_error(new_grid, grid) < tolerance:
            break
    return new_grid




#applying the methods
jacobi_result = jacobi_method(laplace_grid.copy())
gauss_siedel_result = gauss_siedel_method(laplace_grid.copy())


#visualization
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(laplace_grid, cmap='gray')
ax[0].set_title('laplace grid')

ax[1].imshow(jacobi_result, cmap='gray')
ax[1].set_title('Jacobi Method Result')

ax[2].imshow(gauss_siedel_result, cmap='gray')
ax[2].set_title('Gauss-Siedel Method Result')

for a in ax:
    a.axis('off')

plt.show()

