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

# Visualize results
def visualize_results(original, edges):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection (Laplace Operator)')
    
    plt.show()

# Main function
def main():
    # Load the image
    image_path = 'lena.png'  # Replace with your image path
    original_image = load_image(image_path)
    
    # Detect edges using the Laplace operator
    edges = detect_edges(original_image, threshold=0.1)
    
    # Visualize the results
    visualize_results(original_image, edges)

if __name__ == "__main__":
    main()