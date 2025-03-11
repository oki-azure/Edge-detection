% MATLAB Implementation for Edge Detection using Jacobi and Gauss-Seidel Methods

% Step 1: Load and preprocess the image
image_path = 'lena.png'; % Replace with your image path
original_image = imread(image_path);
original_image = rgb2gray(original_image); % Convert to grayscale if needed
original_image = imresize(original_image, [256, 256]); % Resize to 256x256
original_image = double(original_image); % Convert to double for numerical computations

% Step 2: Define the Laplace operator kernel
laplace_kernel = [0 1 0; 1 -4 1; 0 1 0];

% Step 3: Set parameters
max_iter = 5; % Small number of iterations
tol = 1e-4; % Larger tolerance
alpha = 0.5; % Sharpening strength
threshold = 0.1; % Edge detection threshold

% Step 4: Apply Jacobi smoothing
smoothed_jacobi = jacobi_method(original_image, max_iter, tol);

% Step 5: Apply Gauss-Seidel smoothing
smoothed_gauss_seidel = gauss_seidel_method(original_image, max_iter, tol);

% Step 6: Sharpen the smoothed images
sharpened_jacobi = sharpen_image(smoothed_jacobi, alpha, laplace_kernel);
sharpened_gauss_seidel = sharpen_image(smoothed_gauss_seidel, alpha, laplace_kernel);

% Step 7: Detect edges on the original image
edges_original = detect_edges(original_image, threshold, laplace_kernel);

% Step 8: Detect edges on the Jacobi-smoothed and sharpened image
edges_jacobi = detect_edges(sharpened_jacobi, threshold, laplace_kernel);

% Step 9: Detect edges on the Gauss-Seidel-smoothed and sharpened image
edges_gauss_seidel = detect_edges(sharpened_gauss_seidel, threshold, laplace_kernel);

% Step 10: Visualize the results
visualize_results(original_image, smoothed_jacobi, smoothed_gauss_seidel, edges_original, edges_jacobi, edges_gauss_seidel);

% Function Definitions (must appear at the end of the file)

% Jacobi method for smoothing
function u = jacobi_method(image, max_iter, tol)
    u = image;
    u_new = zeros(size(u));
    error = inf;
    iteration = 0;
    
    while error > tol && iteration < max_iter
        u_new(2:end-1, 2:end-1) = 0.25 * (u(2:end-1, 1:end-2) + u(2:end-1, 3:end) + ...
                                     u(1:end-2, 2:end-1) + u(3:end, 2:end-1));
        error = norm(u_new - u, 'fro') / norm(u, 'fro');
        u = u_new;
        iteration = iteration + 1;
    end
    fprintf('Jacobi converged in %d iterations with error %e\n', iteration, error);
end

% Gauss-Seidel method for smoothing
function u = gauss_seidel_method(image, max_iter, tol)
    u = image;
    error = inf;
    iteration = 0;
    
    while error > tol && iteration < max_iter
        u_old = u;
        for i = 2:size(u, 1)-1
            for j = 2:size(u, 2)-1
                u(i, j) = 0.25 * (u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1));
            end
        end
        error = norm(u - u_old, 'fro') / norm(u_old, 'fro');
        iteration = iteration + 1;
    end
    fprintf('Gauss-Seidel converged in %d iterations with error %e\n', iteration, error);
end

% Sharpen the image to enhance edges
function sharpened = sharpen_image(image, alpha, laplace_kernel)
    laplace = conv2(image, laplace_kernel, 'same');
    sharpened = image - alpha * laplace;
    sharpened = max(min(sharpened, 255), 0); % Ensure pixel values are within valid range
end

% Detect edges using the Laplace operator
function edges = detect_edges(image, threshold, laplace_kernel)
    laplace = conv2(image, laplace_kernel, 'same');
    edges = abs(laplace);
    edges = edges / max(edges(:)); % Normalize to [0, 1]
    edges = edges > threshold; % Apply threshold to binarize edges
end

% Visualize results
function visualize_results(original, smoothed_jacobi, smoothed_gauss_seidel, edges_original, edges_jacobi, edges_gauss_seidel)
    figure;
    subplot(2, 3, 1);
    imshow(uint8(original));
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(uint8(smoothed_jacobi));
    title('Smoothed (Jacobi)');
    
    subplot(2, 3, 3);
    imshow(uint8(smoothed_gauss_seidel));
    title('Smoothed (Gauss-Seidel)');
    
    subplot(2, 3, 4);
    imshow(edges_original);
    title('Edges (Original)');
    
    subplot(2, 3, 5);
    imshow(edges_jacobi);
    title('Edges (Jacobi Smoothed)');
    
    subplot(2, 3, 6);
    imshow(edges_gauss_seidel);
    title('Edges (Gauss-Seidel Smoothed)');
end