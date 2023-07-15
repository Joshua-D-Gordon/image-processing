# Convolution & Edge Detection

This repository contains a collection of Python functions for performing convolution, image derivatives, blurring, edge detection, Hough circles detection, and bilateral filtering. These functions were implemented as part of an assignment to demonstrate proficiency in image processing techniques. The assignment covered the following tasks:

## Functions

### 1. Convolution

- `conv1D`: Convolve a 1D array with a given kernel.
- `conv2D`: Convolve a 2D array with a given kernel.

### 2. Image derivatives & blurring

- `convDerivative`: Calculate the gradient of an image.
- `blurImage1`: Blur an image using a Gaussian kernel.
- `blurImage2`: Blur an image using a Gaussian kernel using OpenCV built-in functions.

### 3. Edge detection

- `edgeDetectionZeroCrossingSimple`: Detect edges using the "ZeroCrossing" method.
- `edgeDetectionZeroCrossingLOG`: Detect edges using the "ZeroCrossingLOG" method.

### 4. Hough Circles

- `houghCircle`: Find circles in an image using a Hough Transform algorithm extension.

### 5. Bilateral filter

- `bilateral_filter_implement`: Perform bilateral filtering on an input image.

## Usage

To use the functions in this toolbox, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/convolution-edge-detection-toolbox.git
   ```

2. Import the necessary functions into your Python script:

   ```python
   from convolution_edge_detection_toolbox import conv1D, conv2D, convDerivative, blurImage1, blurImage2, edgeDetectionZeroCrossingSimple, edgeDetectionZeroCrossingLOG, houghCircle, bilateral_filter_implement
   ```

3. Call the desired function with the appropriate inputs and utilize the returned values as needed.

## Examples

Here are some examples demonstrating how to use the functions in this toolbox:

```python
# Example usage of conv1D
conv1D_result = conv1D(in_signal, k_size)

# Example usage of conv2D
conv2D_result = conv2D(in_image, kernel)

# Example usage of convDerivative
directions, magnitude = convDerivative(in_image)

# Example usage of blurImage1
blurred_image1 = blurImage1(in_image, k_size)

# Example usage of blurImage2
blurred_image2 = blurImage2(in_image, k_size)

# Example usage of edgeDetectionZeroCrossingSimple
edges_simple = edgeDetectionZeroCrossingSimple(img)

# Example usage of edgeDetectionZeroCrossingLOG
edges_LOG = edgeDetectionZeroCrossingLOG(img)

# Example usage of houghCircle
circles = houghCircle(img, min_radius, max_radius)

# Example usage of bilateral_filter_implement
cv_result, custom_result = bilateral_filter_implement(in_image, k_size, sigma_color, sigma_space)
```

Feel free to explore the toolbox and modify the functions to suit your specific needs.

