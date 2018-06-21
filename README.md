# Image-colourisation-using-Conditional-GAN

This project uses a conditional GAN to convert grayscale images into colour images. The generator was fed a grayscale image and was tasked to produce the output in the LAB format. The discriminator was fed a concatenation of the grayscale image and its LAB counterpart, and was tasked to recognise the true images from synthetically generated ones. The architecture of the GAN follows that of the U-Net mode.

![Input test image](https://github.com/abhishekvasu94/Image-colourisation-using-Conditional-GAN/blob/master/src/dog.jpg)
The test image shown above was resized to an image of size 32x32, and fed to the input of the generator. 


![Output of the GAN](https://github.com/abhishekvasu94/Image-colourisation-using-Conditional-GAN/blob/master/src/color_img.png)
The output of the GAN is as shown above
