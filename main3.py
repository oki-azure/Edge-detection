import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

# Step 1: Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((256, 256))  # Resize to 256x256
    image_array = np.array(image, dtype=float)
    return image_array

# Step 2: Apply the Laplace operator for edge detection
def laplace_edge_detection(image):
    # Laplace kernel
    laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])
    
    # Apply the Laplace kernel using convolution
    edges = convolve(image, laplace_kernel, mode='constant', cval=0)
    
    # Take absolute value to highlight edges
    edges = np.abs(edges)
    
    # Normalize to [0, 1]
    edges = edges / np.max(edges)
    
    # Apply a threshold to binarize the edges
    edges = edges > 0.1  # Adjust threshold as needed
    
    return edges

# Step 3: Apply the Sobel operator for edge detection
def sobel_edge_detection(image):
    # Sobel kernels
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    
    sobel_kernel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    
    # Apply Sobel kernels using convolution
    edges_x = convolve(image, sobel_kernel_x, mode='constant', cval=0)
    edges_y = convolve(image, sobel_kernel_y, mode='constant', cval=0)
    
    # Combine x and y gradients
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Normalize to [0, 1]
    edges = edges / np.max(edges)
    
    # Apply a threshold to binarize the edges
    edges = edges > 0.1  # Adjust threshold as needed
    
    return edges

# Step 4: Visualize the results
def visualize_results(original, laplace_edges, sobel_edges):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(laplace_edges, cmap='gray')
    plt.title('Laplace Edge Detection')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    
    plt.show()

# Main function
def main():
    # Load the image
    image_path = 'lena.png'  # Replace with your image path
    original_image = load_image(image_path)
    
    # Apply Laplace edge detection
    laplace_edges = laplace_edge_detection(original_image)
    
    # Apply Sobel edge detection
    sobel_edges = sobel_edge_detection(original_image)
    
    # Visualize the results
    visualize_results(original_image, laplace_edges, sobel_edges)

if __name__ == "__main__":
    main()