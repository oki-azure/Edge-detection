import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import requests
import time

# Function to download Lena image
def download_lena():
    url = "https://raw.githubusercontent.com/mikolalysenko/lena/master/lena.png"
    filename = "lena.png"
    if not os.path.exists(filename):
        print("Downloading Lena image from the internet...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

# Function to load the image (Lena or synthetic fallback)
def load_image():
    download_lena()
    if os.path.exists('lena.png'):
        print("Loading Lena image...")
        img = imageio.imread('lena.png', mode='F')  # Updated mode
    else:
        print("Lena image not found. Generating a synthetic 256x256 image instead.")
        img = np.random.rand(256, 256) * 255
        for i in range(256):
            img[i, :] += i / 256  # Add a gradient
        img = np.clip(img, 0, 255)
    
    img = img.astype(float) / 255.0  # Normalize to [0, 1]
    
    if img.shape != (256, 256):
        from scipy.ndimage import zoom
        scale_x = 256 / img.shape[0]
        scale_y = 256 / img.shape[1]
        img = zoom(img, (scale_x, scale_y))
    
    return img

# Jacobi iteration method with relative error stopping criterion
def jacobi_method(u, tol=1e-6, max_iter=3000):
    n, m = u.shape
    u_new = u.copy()
    iteration = 0
    error_history = []

    start_time = time.time()
    for iteration in range(max_iter):
        u_old = u_new.copy()
        for i in range(1, n-1):
            for j in range(1, m-1):
                u_new[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1])
        
        error = np.linalg.norm(u_new - u_old) / np.linalg.norm(u_old) if np.linalg.norm(u_old) > 0 else 0
        error_history.append(error)
        if error < tol:
            break
    execution_time = time.time() - start_time

    return u_new, iteration + 1, error_history[-1], execution_time

# Gauss-Seidel iteration method with relative error stopping criterion
def gauss_seidel_method(u, tol=1e-6, max_iter=1000):
    n, m = u.shape
    u_new = u.copy()
    iteration = 0
    error_history = []

    start_time = time.time()
    for iteration in range(max_iter):
        u_old = u_new.copy()
        for i in range(1, n-1):
            for j in range(1, m-1):
                u_new[i, j] = 0.25 * (u_new[i+1, j] + u_new[i-1, j] + u_new[i, j+1] + u_new[i, j-1])
        
        error = np.linalg.norm(u_new - u_old) / np.linalg.norm(u_old) if np.linalg.norm(u_old) > 0 else 0
        error_history.append(error)
        if error < tol:
            break
    execution_time = time.time() - start_time

    return u_new, iteration + 1, error_history[-1], execution_time

# Function to run with fixed iterations for specific analysis
def run_fixed_iterations(u, iterations):
    # Jacobi Method
    start_time_jacobi = time.time()
    n, m = u.shape
    u_jacobi = u.copy()
    for _ in range(iterations):
        u_old = u_jacobi.copy()
        for i in range(1, n-1):
            for j in range(1, m-1):
                u_jacobi[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1])
        error_jacobi = np.linalg.norm(u_jacobi - u_old) / np.linalg.norm(u_old) if np.linalg.norm(u_old) > 0 else 0
    time_jacobi = time.time() - start_time_jacobi
    rate_jacobi = iterations / time_jacobi if time_jacobi > 0 else 0

    # Gauss-Seidel Method
    start_time_gauss = time.time()
    u_gauss = u.copy()
    for _ in range(iterations):
        u_old = u_gauss.copy()
        for i in range(1, n-1):
            for j in range(1, m-1):
                u_gauss[i, j] = 0.25 * (u_gauss[i+1, j] + u_gauss[i-1, j] + u_gauss[i, j+1] + u_gauss[i, j-1])
        error_gauss = np.linalg.norm(u_gauss - u_old) / np.linalg.norm(u_old) if np.linalg.norm(u_old) > 0 else 0
    time_gauss = time.time() - start_time_gauss
    rate_gauss = iterations / time_gauss if time_gauss > 0 else 0

    return (iterations, time_jacobi, error_jacobi, rate_jacobi,
            iterations, time_gauss, error_gauss, rate_gauss)

# Main execution
def main():
    # Load and preprocess image
    img = load_image()
    if img.shape != (256, 256):
        print(f"Image resized to 256x256 from {img.shape}")

    # Set Dirichlet boundary conditions (borders = 0)
    u = img.copy()
    u[0, :] = 0  # Top border
    u[-1, :] = 0  # Bottom border
    u[:, 0] = 0  # Left border
    u[:, -1] = 0  # Right border

    # Run with relative error stopping criterion
    print("Running with relative error stopping criterion (tol = 1e-6):")
    u_jacobi, iter_jacobi, final_error_jacobi, time_jacobi = jacobi_method(u.copy())
    u_gauss, iter_gauss, final_error_gauss, time_gauss = gauss_seidel_method(u.copy())
    print(f"Jacobi Method: {iter_jacobi} iterations, Time = {time_jacobi:.3f} seconds, Final Error = {final_error_jacobi:.6e}")
    print(f"Gauss-Seidel Method: {iter_gauss} iterations, Time = {time_gauss:.3f} seconds, Final Error = {final_error_gauss:.6e}")

    # Run with fixed iterations (200, 300, 400)
    print("\nResults for Fixed Iterations:")
    for iterations in [200, 300, 400]:
        (iter_j, time_j, error_j, rate_j,
         iter_g, time_g, error_g, rate_g) = run_fixed_iterations(u.copy(), iterations)
        print(f"{iterations} iterations:")
        print(f"  Jacobi Method: Time = {time_j:.3f} seconds, Final Error = {error_j:.6e}, "
              f"Convergence Rate = {rate_j:.1f} iterations/second")
        print(f"  Gauss-Seidel Method: Time = {time_g:.3f} seconds, Final Error = {error_g:.6e}, "
              f"Convergence Rate = {rate_g:.1f} iterations/second")

    # Visualization (optional)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(u_jacobi, cmap='gray')
    plt.title(f'Jacobi Method\n{iter_jacobi} iterations')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(u_gauss, cmap='gray')
    plt.title(f'Gauss-Seidel Method\n{iter_gauss} iterations')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()