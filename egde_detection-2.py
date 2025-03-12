import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import time

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
def jacobi_method(image, max_iter=5, tol=1e-6):
    u = image.copy()
    u_new = np.zeros_like(u)
    error = np.inf
    iteration = 0
    errors = []  # Store errors at each iteration

    while error > tol and iteration < max_iter:
        u_new[1:-1, 1:-1] = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        error = np.linalg.norm(u_new - u) / np.linalg.norm(u)
        errors.append(error)  # Store the error
        u = u_new.copy()
        iteration += 1

    print(f"Jacobi converged in {iteration} iterations with final error {error}")
    return u, errors

# Gauss-Seidel method for smoothing
def gauss_seidel_method(image, max_iter=5, tol=1e-6):
    u = image.copy()
    error = np.inf
    iteration = 0
    errors = []  # Store errors at each iteration

    while error > tol and iteration < max_iter:
        u_old = u.copy()
        for i in range(1, u.shape[0]-1):
            for j in range(1, u.shape[1]-1):
                u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        error = np.linalg.norm(u - u_old) / np.linalg.norm(u_old)
        errors.append(error)  # Store the error
        iteration += 1

    print(f"Gauss-Seidel converged in {iteration} iterations with final error {error}")
    return u, errors

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

# Plot convergence rates
def plot_convergence(jacobi_errors, gauss_seidel_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(jacobi_errors, label='Jacobi Method', color='blue')
    plt.plot(gauss_seidel_errors, label='Gauss-Seidel Method', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.title('Convergence Rates: Jacobi vs. Gauss-Seidel')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    # Load the image
    image_path = 'lena.png'  # Replace with your image path
    original_image = load_image(image_path)
    
    # Step 1: Apply the Laplace operator to the original image
    edge_enhanced_image = laplace_operator(original_image)
    
    # Step 2: Apply Jacobi smoothing to the edge-enhanced image
    start_time = time.time()
    smoothed_jacobi, jacobi_errors = jacobi_method(edge_enhanced_image, max_iter=5, tol=1e-6)
    jacobi_time = time.time() - start_time
    print(f"Jacobi method took {jacobi_time:.4f} seconds")
    
    # Step 3: Apply Gauss-Seidel smoothing to the edge-enhanced image
    start_time = time.time()
    smoothed_gauss_seidel, gauss_seidel_errors = gauss_seidel_method(edge_enhanced_image, max_iter=5, tol=1e-6)
    gauss_seidel_time = time.time() - start_time
    print(f"Gauss-Seidel method took {gauss_seidel_time:.4f} seconds")
    
    # Step 4: Detect edges on the original image
    edges_original = detect_edges(original_image, threshold=0.1)
    
    # Step 5: Detect edges on the Jacobi-smoothed image
    edges_jacobi = detect_edges(smoothed_jacobi, threshold=0.1)
    
    # Step 6: Detect edges on the Gauss-Seidel-smoothed image
    edges_gauss_seidel = detect_edges(smoothed_gauss_seidel, threshold=0.1)
    
    # Step 7: Visualize the results
    visualize_results(original_image, smoothed_jacobi, smoothed_gauss_seidel, edges_original, edges_jacobi, edges_gauss_seidel)
    
    # Step 8: Plot convergence rates
    plot_convergence(jacobi_errors, gauss_seidel_errors)

if __name__ == "__main__":
    main()