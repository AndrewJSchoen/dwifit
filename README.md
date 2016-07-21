# dwifit
Fit diffusion-weighted imaging data to a variety of models.
Using the fits we compute residuals which are used in estimating noise standard deviation. The mean signal is defined as the average of the non-diffusion weighted (T2 weighted) or b=0 images. The SNR then is defined as the ratio of the signal to the noise std images. Note that the assumption is that noise is independent across voxels but "same" across directions.
