# vision6
Color

6.1. Color space
6.1.1. Convert Lena to HSI format, and display the HIS components as separate grayscale images. Observe these
images to comment on what does each of the H, S, I components represent. The HSI images should be saved in
double precision.
6.1.2. *Present and discuss new color space (at least three) in detail which was not introduced in class (Application,
Equation, etc.).
6.2. Quantization
6.2.1. Implement uniform quantization of a color image. Your program should do the following:
1. Read a grayscale image into an array.
2. Quantize and save the quantized image in a different array.
3. Compute the MSE and PSNR between the original and quantized images.
4. Display and print the quantized image.
Notice, your program should assume the input values are in the range of (0,256), but allow you to vary the
reconstruction level. Record the MSE and PSNR obtained with � = 64, 32, 16, 8 and display the quantized images
with corresponding � values. Comment on the image quality as you vary �. (Test on Lena Image).
6.2.2. For the Lena image, quantize the R, G, and B components to 3, 3, and 2 bits, respectively, using a uniform
quantizer. Display the original and quantized color image. Comment on the difference in color accuracy.6.2.3. We want to weave the Baboon image on a rug. To do so, we need to reduce the number of colors in the image
with minimal visual quality loss. If we can have 32, 16 and 8 different colors in the weaving process, reduce the
color of the image to these three special modes. Discuss and display the results.
Note: you can use immse and psnr for problem 6.2
