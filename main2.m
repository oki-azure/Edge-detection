% MATLAB Implementation for Edge Detection using Laplace Operator

% Step 1: Load and preprocess the image
image_path = 'lena.png'; % Replace with your image path
original_image = imread(image_path);
original_image = rgb2gray(original_image); % Convert to grayscale if needed
original_image = imresize(original_image, [256, 256]); % Resize to 256x256
original_image = double(original_image); % Convert to double for numerical computations

% Step 2: Apply the Laplace operator for edge detection
laplace_filter = [0 1 0; 1 -4 1; 0 1 0]; % Laplace kernel
edges = imfilter(original_image, laplace_filter, 'same'); % Apply Laplace filter

% Step 3: Normalize and threshold the edges
edges = abs(edges); % Take absolute value to highlight edges
edges = edges / max(edges(:)); % Normalize to [0, 1]
edges = edges > 0.1; % Apply a threshold to binarize the edges

% Step 4: Visualize the results
figure;
subplot(1, 2, 1);
imshow(uint8(original_image)); % Display original image
title('Original Image');

subplot(1, 2, 2);
imshow(edges); % Display edge-detected image
title('Edge Detection (Laplace Operator)');