# Idea on Pretraining

Xiaoyu Liu, 31 Jan 2021. Redistribution under permission.

## Rationale

We pre-train the model with generated fake data, and then train with actual data and validate its result.

## Model Construction

### Input and Output

    input: the OD. This is a better choice than directly using the raw image is that, OD is more structured and provides more information to the algorithm.

    output: the parameters. 

### Model Selection

Which ML algo to use?

    Since the input is an image, CNN is somewhere to go. 
    Since the input is not a time series, we might not need memory / recurrency in our network.

    We might also consider other ML algorithms and test them empirically.

The rationale of using ML

    We want to summarize the input-output relationship in a way that is both precise and efficient.

    We are essentially producing a "rainbow sheet" that covers the whole domain of possible measurements. Then we pass it partially to a "compressor" (in this case the NN) and see if the information can be represented with lower number of parameters and low loss.

## Pretraining Schemes

### A. Generated Params and Image Reconstruction

1. Determine the range of following parameters: 

    ```python
    # example of a set of params
    tau     =  1.0526 # describe the decaying of transmission efficiency with radius
    S0      = -1.0497 # spherical aberration
    alpha   =  1.0266 # astigmatism
    beta    =  0.8316 # defocus
    delta_s = -0.0667 # phase imprint by atom scattering
    ```

1. Generate a large set of parameters configuratioins that cover the domain.

1. For each set of the generated parameters, we have a fully-defined OTF. Then we can generate the OD measurements backwards and add random noise.

1. Pretrain the model with the OD - parameter pairs.

### B. Generated Images and Crude Fitting

1. We generate a large set of fake raw image, or fake OD image. Note that we need to generate batches.

1. To generate a batch, we can first generate an "ideal" image and then apply random noise to that image for multiple times.

1. For each generated batch we can apply the fitting algorithm and obtain the fitting result. Even if the input image is nonsense, the fitting will still produce some result. This is ok because it provides information about the fitting process.

1. Pretrain the model with the OD - parameter pairs.

## Applying the pretraining schemes

We may discuss how to use A and B cooperatively. When the model is pre-trained and converges in scheme A and scheme B, we can use real data to test the model.

If the model has learned well, training the pre-trained model with more real data will only slightly optimize its parameters. If the model prediction drastically deviates the actual value, then the pre-training is unsuccessful.