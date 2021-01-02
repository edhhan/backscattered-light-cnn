# backscattered-light-cnn
A final year project of a engineering physics degree. The project is entirely simulation based. We study the influence of the thickness of a biological tissu regarding the resulting backscattered light profil. We also study the influence of thermal and shot noise on the profils. More precisely, we wish to classify the backscattered light profil with respect to the thickness. The task at hand his to classify the thickness into specifics intervals (mm) : ]8, 12] ; [12, 16] ; [16, 20]. 

# Methodology 

## Dataset
We generate our own dataset of backscattered light profils with random thickness between 8 to 20mm, using a RNG based on the actual time that follows an uniform distribution U[8,20]. The profils are generated with Monte-Carlo simulation for different number of photons (light intensity) : 1e4, 1e5 and 1e6. We use an open-source libray to generate those profils : [MMCLAB](http://mcx.space/wiki/index.cgi?MMC/Doc/MMCLAB). 

## DL models

