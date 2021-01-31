# Idea on Data Augmentation

Xiaoyu Liu, 31 Jan 2021. Redistribution under permission.

## Rationale

Instead of treating the problem as `single_image - to - parameters` prediction problem, we try to expand a raw image into a whole batch, which can be directly fed into the fitting algorithm and obtain the fitting results.

## Arguments for Feasibility

### Variance and Noise Distribution is Encoded in a Single Image

    The image has many pixels next to each other that create a large redundancy. This is already a large sample space that we can use to determine the noise.

### We know noise ALA we know the ideal value

    In the traditional fitting procedure, we compute an average image from many observations, and then obtain the "noise" images by substracting each image from the mean.

    The average image is a good approximation to the "ideal image". The "ideal image" is an image without any noise, which should look perfectly smooth and beautiful.

    My opinion is, it's possible to guess the "ideal image" even from a single image. Then the noise can be determined directly by comparing the raw image with the ideal image.

### Determine the Ideal Image from Single Image is possible 

    Intuitively, we just "smooth out" the image in a way that make it "look like" an averaged OD image.

    When we look at a raw image, we can easily imagine what the ideal image would look like. This human intuition can possibly be realized by ML algo.

    This process does not require more supply of information. It's essentially a loss of information. Hence it's possible to train an ML algo to do this.

### We Can Generate a Batch from a Single Image

    Since we know the ideal image, we can know the noise image by substracting from the input image.

    Since we know the noise image, we can know (approximate) the noise distribution.

    Since we know the noise distribution, we can generate more noise consistent with the distribution image and apply these noises on the ideal image.

    The input image and the noises applied to ideal image form an augmented batch.

## Realization

### Basic Approaches w/o ML

    Opencv playing-around. Try to perform convolution and smoothing on the raw image, to even out the noise.

    2d FFT and remove high-frequency information. It is a "compress-decompress" process that might drop the details (which is the noise) and leave the overall tendency (which is the ideal image)

    If these somehow works, they can be baseline model for the ML algos. 

### Approach w ML

    Image GAN. It can learn what an ideal image "look like", and then generate an ideal that is closest to the raw image.
    
    Image CNN AutoEncoder. It is a "compress-decompress" process that might drop the details (which is the noise) and leave the overall tendency (which is the ideal image)