# backscattered-light-cnn
An undergrad final year project in engineering physics at Polytechnique Montréal. The project is entirely simulation-based. We study backscattered light profiles of biological tissues with respect to their thickness. We also study the influence of thermal and shot noise on these profiles. More precisely, we wish to classify the profiles : profiles are inputs and thickness are labels (supervised learning). The main task/goal is to classify each profile into a specific interval of thickness (mm) : [8, 12] ; ]12, 16] ; ]16, 20]. The project itself is a proof of concept, where we want to test if it is conceivable to apply DL models to classify the thickness of an eperdimis with a backscattered light profile. 

# Methodology 

## Dataset
We generated our own datasets of backscattered light profiles with random thickness between 8 to 20mm, using a RNG based on the actual time that follows an uniform distribution U[8,20]. The profiles are generated with Monte-Carlo simulations for a given number of photons (light intensity) : 1e4, 1e5 and 1e6 (we have 3 datasets). We use an open-source library to generate those profiles : [MMCLAB](http://mcx.space/wiki/index.cgi?MMC/Doc/MMCLAB). 

We generated with considerably small datasets, approximately 1500 profiles for each dataset : because of time restrictions, the project itself was only a proof of concept. We allow 20% of the dataset for validation and test purposes. 

Each profile is a 28x28 image grey-scale (the intensity of light). Example with 1e6 :
<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/intensity_nonoise.png" width="500" height="300">

We apply different perturbations on the images, such as thermal and shot noises. Example with 1e6 :
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

-For an efficient space management, the datasets are initially contained in a .zip file. With the first run, the code should automatically unzip the datasets and generate a .npy file in the local repos. For the next runs, the code will directly load the .npy file.

-If you don't have access to a GPU-cuda, make sure to unflag the GPU boolean variable at the beginning of the main section.

-Choose a number of photon between : 1e4, 1e5 or 1e6.


# Results with 1e6
With 5 different runs, we obtained an average of 91.6% and a standard-deviation of 1.8% for 3 classes. The accuracy isn't that great considering that we only have 3 classes. However, the accuracy was considered great enough as a proof of concept. In addition, the physical noise artifically added decreases significantly the quality of the 28x28 images. Plus, the datasets themselves are pretty small for a context of DL. 

<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/accuracy_1e6.png" width="500" height="300">

<img src="https://github.com/edhhan/backscattered-light-cnn/blob/main/images/losses_1e6.png" width="500" height="300">

Without any noise, we obtained better results with an average of 95.1% and a standard 1.4% for 3 classes. Our best run gave an accuracy of 97.5%.






