# backscattered-light-cnn
A final year project of a engineering physics degree. The project is entirely simulation based. We study the influence of the thickness of a biological tissu regarding the resulting backscattered light profil. We also study the influence of thermal and shot noise on the profils. More precisely, we wish to classify the backscattered light profil with respect to the thickness. The main task/goal is to classify the thickness into specifics intervals (mm) : ]8, 12] ; [12, 16] ; [16, 20]. The project itself is a proof of concept, where we want to test if it possible to apply DL models to classify the thickness of an eperdimis with a backscattered light profil. 

# Methodology 

## Dataset
We generate our own dataset of backscattered light profils with random thickness between 8 to 20mm, using a RNG based on the actual time that follows an uniform distribution U[8,20]. The profils are generated with Monte-Carlo simulation for different number of photons (light intensity) : 1e4, 1e5 and 1e6. We use an open-source libray to generate those profils : [MMCLAB](http://mcx.space/wiki/index.cgi?MMC/Doc/MMCLAB). 

We work with considerably small datasets approximately 1500 profils for each number of photons : we were time restricted and the project itself is only a proof of concept. We accord 20% of the dataset for validation and test purposes. 

Each profil is a 28x28 image of grey-scale (the intensity of light). Example with 1e6 :
<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/intensity_nonoise.png" width="500" height="300">

We apply different perturbation on the images, such as thermal and shot noise. Example with 1e6 :
<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/intensity_th.png" width="500" height="300">

The physical parameters are typical for a 1100nm laser setup. See gen_data(signal, nb_photons, wavelength, dim) from utils/data_utils.py for more details.

## DL models
We use the PyTorch library to implement different DL models : a CNN, a FCC and a hybrid CNN-FCC. Without any suprise, the CNN performs better than other models since the task implies image classification. 

The hyperparameters of the CNN model were hand-tuned, thus the model isn't properly optimized. 

# Packages
```
numpy
```
```
torch
```
```
matplotlib.pyplot
```
```
tqdm 
```

# Run

-The data profils are initially contained in a .zip file, for efficiency space management purposes. With the first run, the code should automatically unzip the datasets and generate a .npy file, for the corresponding number of photons (e.g. 1e4), in the local repos. For the next runs, the code will directly load the .npy file.

-If you don't have acces to a GPU-cuda, make sure to unflag the GPU boolean variable at the beginning of the main section.

-Choose a number of photon between : 1e4, 1e5 or 1e6.


# Results with 1e6
With 5 different runs we obtained an average of 91.6% and a standard-deviation of 1.8% for 3 classes. The precision isn't that great considering we have only 3 classes, however the precision was considered great enough as a proof of concept. In addition, the physical noise artifically added decreases significantly the quality of the 28x28 images and the dataset itselft is pretty small for a context of DL. 

<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/accuracy_1e6.png.png" width="500" height="300">

<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/losses_1e6.png.png" width="500" height="300">

Without any noise, we obtained better results with an average of 95.1% and a standard 1.4% for 3 classes. Our best run gave a precision of 97.5%.






