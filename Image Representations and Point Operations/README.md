# Image Representations and Point Operations

This repository contains a set of Python functions for basic image processing tasks. The main purpose of this exercise was to get acquainted with Python's basic syntax and some of its image processing facilities. The tasks covered in this exercise include loading grayscale and RGB image representations, displaying figures and images, transforming RGB color images back and forth from the YIQ color space, performing intensity transformations (histogram equalization), and performing optimal quantization.

## Functions

### 1. `imReadAndConvert`

This function reads an image file and converts it into a given representation. It takes the path to the image file and the desired representation (grayscale or RGB) as inputs and returns the image as a NumPy array. The output image is represented by a matrix of class `np.float` with intensities normalized to the range [0, 1].

### 2. `imDisplay`

This function utilizes `imReadAndConvert` to display a given image file in a specified representation. It takes the path to the image file and the desired representation (grayscale or RGB) as inputs and opens a new figure window to display the loaded image using `plt.imshow`.

### 3. `RGB2YIQ` and `YIQ2RGB`

These two functions transform an RGB image into the YIQ color space and vice versa. `RGB2YIQ` takes an RGB image as input and returns the corresponding YIQ image as a NumPy array. `YIQ2RGB` takes a YIQ image as input and returns the corresponding RGB image as a NumPy array.

### 4. `histogramEqualize`

This function performs histogram equalization on a given grayscale or RGB image. It takes the original image as input and returns the equalized image, the histogram of the original image, and the histogram of the equalized image. If an RGB image is given, the equalization procedure operates only on the Y channel of the corresponding YIQ image and then converts it back to RGB.

### 5. `quantizeImage`

This function performs optimal quantization of a given grayscale or RGB image. It takes the original image, the number of colors to quantize the image to, and the number of optimization loops as inputs. It returns a list of quantized images in each iteration and a list of mean square error (MSE) values in each iteration.

### 6. `gammaCorrection`

This function performs gamma correction on an image with a given γ (gamma) value. It takes the image in BGR format and the gamma value as inputs and returns the gamma-corrected image.

## Usage

To use the functions in this toolbox, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/image-processing-toolbox.git
   ```

2. Import the necessary functions into your Python script:

   ```python
   from image_processing_toolbox import imReadAndConvert, imDisplay, RGB2YIQ, YIQ2RGB, histogramEqualize, quantizeImage, gammaCorrection
   ```

3. Call the desired function with the appropriate inputs and utilize the returned values as needed.

## Examples

Here are some examples demonstrating how to use the functions in this toolbox:

```python
# Example usage of imReadAndConvert
image = imReadAndConvert('path/to/image.jpg', representation=2)

# Example usage of imDisplay
imDisplay('path/to/image.jpg', representation=2)

# Example usage of RGB2YIQ and YIQ2RGB
imYIQ = RGB2YIQ(image)
imRGB = YIQ2RGB(imYIQ)

# Example usage of histogramEqualize
imgEq, histOrig, histEq = histogramEqualize(image)

# Example usage of quantizeImage
quantized_images, mse_values = quantizeImage(image, nQuant=16, nIter=5)

# Example usage of gammaCorrection
gamma_corrected_image = gammaCorrection(image, gamma=2.2)
```

Feel free to explore the toolbox and modify the functions to suit your specific needs.
