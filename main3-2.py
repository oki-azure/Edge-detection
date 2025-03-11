import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((256, 256))  # Resize to 256x256
    image_array = np.array(image, dtype=float)
    return image_array

# Laplace operator kernel
def laplace_operator(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    laplace = convolve(image, kernel, mode='constant', cval=0)
    return laplace

# Detect edges using the Laplace operator
def detect_edges(image, threshold=0.1):
    # Apply the Laplace operator
    laplace = laplace_operator(image)
    
    # Take the absolute value to highlight edges
    edges = np.abs(laplace)
    
    # Normalize to [0, 1]
    edges = edges / np.max(edges)
    
    # Apply a threshold to binarize the edges
    edges = edges > threshold
    
    return edges

# Jacobi method for smoothing
def jacobi_method(image, max_iter=5, tol=1e-4):  # Small number of iterations
    u = image.copy()
    u_new = np.zeros_like(u)
    error = np.inf
    iteration = 0

    while error > tol and iteration < max_iter:
        u_new[1:-1, 1:-1] = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        error = np.linalg.norm(u_new - u) / np.linalg.norm(u)
        u = u_new.copy()
        iteration += 1

    print(f"Jacobi converged in {iteration} iterations with error {error}")
    return u

# Gauss-Seidel method for smoothing
def gauss_seidel_method(image, max_iter=5, tol=1e-4):  # Small number of iterations
    u = image.copy()
    error = np.inf
    iteration = 0

    while error > tol and iteration < max_iter:
        u_old = u.copy()
        for i in range(1, u.shape[0]-1):
            for j in range(1, u.shape[1]-1):
                u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        error = np.linalg.norm(u - u_old) / np.linalg.norm(u_old)
        iteration += 1

    print(f"Gauss-Seidel converged in {iteration} iterations with error {error}")
    return u

# Sharpen the image to enhance edges
def sharpen_image(image, alpha=0.1):
    laplace = laplace_operator(image)
    sharpened = image - alpha * laplace
    sharpened = np.clip(sharpened, 0, 255)  # Ensure pixel values are within valid range
    return sharpened

# Visualize results
def visualize_results(original, smoothed_jacobi, smoothed_gauss_seidel, edges_original, edges_jacobi, edges_gauss_seidel):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(2, 3, 2)
    plt.imshow(smoothed_jacobi, cmap='gray')
    plt.title('Smoothed (Jacobi)')
    
    plt.subplot(2, 3, 3)
    plt.imshow(smoothed_gauss_seidel, cmap='gray')
    plt.title('Smoothed (Gauss-Seidel)')
    
    plt.subplot(2, 3, 4)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Edges (Original)')
    
    plt.subplot(2, 3, 5)
    plt.imshow(edges_jacobi, cmap='gray')
    plt.title('Edges (Jacobi Smoothed)')
    
    plt.subplot(2, 3, 6)
    plt.imshow(edges_gauss_seidel, cmap='gray')
    plt.title('Edges (Gauss-Seidel Smoothed)')
    
    plt.show()

# Main function
def main():
    # Load the image
    image_path = 'lena.png'  # Replace with your image path
    original_image = load_image(image_path)
    
    # Apply Jacobi smoothing (with fewer iterations)
    smoothed_jacobi = jacobi_method(original_image, max_iter=5, tol=1e-4)
    
    # Apply Gauss-Seidel smoothing (with fewer iterations)
    smoothed_gauss_seidel = gauss_seidel_method(original_image, max_iter=5, tol=1e-4)
    
    # Sharpen the smoothed images to enhance edges
    sharpened_jacobi = sharpen_image(smoothed_jacobi, alpha=0.1)
    sharpened_gauss_seidel = sharpen_image(smoothed_gauss_seidel, alpha=0.1)
    
    # Detect edges on the original image
    edges_original = detect_edges(original_image, threshold=0.1)
    
    # Detect edges on the Jacobi-smoothed and sharpened image
    edges_jacobi = detect_edges(sharpened_jacobi, threshold=0.1)
    
    # Detect edges on the Gauss-Seidel-smoothed and sharpened image
    edges_gauss_seidel = detect_edges(sharpened_gauss_seidel, threshold=0.1)
    
    # Visualize the results
    visualize_results(original_image, smoothed_jacobi, smoothed_gauss_seidel, edges_original, edges_jacobi, edges_gauss_seidel)

if __name__ == "__main__":
    main()