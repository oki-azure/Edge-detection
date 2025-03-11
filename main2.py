import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((256, 256))  # Resize to 256x256
    image_array = np.array(image, dtype=float)
    return image_array

# Jacobi method for solving the Laplace equation
def jacobi_method(image, max_iter=100, tol=1e-4):
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

# Gauss-Seidel method for solving the Laplace equation
def gauss_seidel_method(image, max_iter=100, tol=1e-4):
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

# Visualize results
def visualize_results(original, jacobi_result, gauss_seidel_result):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(jacobi_result, cmap='gray')
    plt.title('Jacobi Method')
    
    plt.subplot(1, 3, 3)
    plt.imshow(gauss_seidel_result, cmap='gray')
    plt.title('Gauss-Seidel Method')
    
    plt.show()

# Main function
def main():
    # Load the image
    image_path = 'lena.png'  # Replace with your image path
    original_image = load_image(image_path)
    
    # Apply Jacobi method
    jacobi_result = jacobi_method(original_image)
    
    # Apply Gauss-Seidel method
    gauss_seidel_result = gauss_seidel_method(original_image)
    
    # Visualize the results
    visualize_results(original_image, jacobi_result, gauss_seidel_result)

if __name__ == "__main__":
    main()