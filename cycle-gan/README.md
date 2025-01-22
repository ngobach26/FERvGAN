# cycle-gan-fer
Cycle GAN is one of the renowned models for generating solving image-to-image translation for an un-paired dataset. In this repository we aim to provide
a cycle gan model for generating new samples for FER-2013(facial expression recognition) dataset. The aforementioned dataset is biased which means that the number of samples in classes are the same or even close together. 
To tackle this issue we utilize cycle-gan model to generate new samples for `disgust` class which has the lowest number of samples.  

## Train 
To train the model run the following command:
```commandline
python train.py
```
## Test
```commandline
python test.py
```
...
```

