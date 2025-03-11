% MATLAB Implementation of Jacobi and Gauss-Seidel Methods for Edge Detection

% Step 1: Load and preprocess the image
image_path = 'lena.png'; % Replace with your image path
original_image = imread(image_path);
original_image = rgb2gray(original_image); % Convert to grayscale if needed
original_image = imresize(original_image, [256, 256]); % Resize to 256x256
original_image = double(original_image); % Convert to double for numerical computations

% Step 2: Initialize variables
u = original_image; % Initial guess (original image)
u_jacobi = u; % For Jacobi method
u_gauss_seidel = u; % For Gauss-Seidel method
max_iter = 1000; % Maximum number of iterations
tol = 1e-6; % Stopping criterion (tolerance)
error_jacobi = inf; % Initialize error for Jacobi
error_gauss_seidel = inf; % Initialize error for Gauss-Seidel
iter_jacobi = 0; % Iteration counter for Jacobi
iter_gauss_seidel = 0; % Iteration counter for Gauss-Seidel

% Step 3: Implement Jacobi Method
while error_jacobi > tol && iter_jacobi < max_iter
    u_old_jacobi = u_jacobi; % Store the old values
    u_jacobi(2:end-1, 2:end-1) = 0.25 * (u_old_jacobi(1:end-2, 2:end-1) + u_old_jacobi(3:end, 2:end-1) + ...
                                          u_old_jacobi(2:end-1, 1:end-2) + u_old_jacobi(2:end-1, 3:end));
    error_jacobi = norm(u_jacobi - u_old_jacobi, 'fro') / norm(u_old_jacobi, 'fro'); % Relative error
    iter_jacobi = iter_jacobi + 1;
end
fprintf('Jacobi converged in %d iterations with error %e\n', iter_jacobi, error_jacobi);

% Step 4: Implement Gauss-Seidel Method
while error_gauss_seidel > tol && iter_gauss_seidel < max_iter
    u_old_gauss_seidel = u_gauss_seidel; % Store the old values
    for i = 2:size(u_gauss_seidel, 1)-1
        for j = 2:size(u_gauss_seidel, 2)-1
            u_gauss_seidel(i, j) = 0.25 * (u_gauss_seidel(i-1, j) + u_gauss_seidel(i+1, j) + ...
                                    u_gauss_seidel(i, j-1) + u_gauss_seidel(i, j+1));
        end
    end
    error_gauss_seidel = norm(u_gauss_seidel - u_old_gauss_seidel, 'fro') / norm(u_old_gauss_seidel, 'fro'); % Relative error
    iter_gauss_seidel = iter_gauss_seidel + 1;
end
fprintf('Gauss-Seidel converged in %d iterations with error %e\n', iter_gauss_seidel, error_gauss_seidel);

% Step 5: Visualize the results
figure;
subplot(1, 3, 1);
imshow(uint8(original_image)); % Display original image
title('Original Image');

subplot(1, 3, 2);
imshow(uint8(u_jacobi)); % Display Jacobi result
title('Jacobi Method');

subplot(1, 3, 3);
imshow(uint8(u_gauss_seidel)); % Display Gauss-Seidel result
title('Gauss-Seidel Method');