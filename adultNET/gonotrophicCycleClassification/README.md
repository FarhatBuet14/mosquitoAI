## Classifying Stages in the Gonotrophic Cycle of Mosquitoes
We have designed computer vision techniques to determine stages in the gonotrophic cycle (unfed, fully fed, semi-gravid and gravid)) of female mosquitoes (*Aedes aegypti, Anopheles stephensi, and Culex quinquefasciatus*) from images captured by smartphones.

See the paper [here](https://assets.researchsquare.com/files/rs-3191730/v1_covered_6cfac5b8-31ac-4e10-897c-b692ac1255ff.pdf?c=1690863911)

## Abdominal Conditions of a Female Mosquito According to the Stages of its Gonotrophic Cycle

![gonotrphic_cycle.png](https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/blob/main/images/gonotrphic_cycle.png)

## Dataset Details

![dataset.png](https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/blob/main/images/dataset.png)

## Data Augmentation Techniques

![augmentation.png](https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/blob/main/images/augmentation.png)

## Requirements
* Python 3.10.12
* Tensorflow 2.12
* Keras 2.12

## Folder Details

| Folder       | Description                                                               |
|--------------|---------------------------------------------------------------------------|
| `codes/`     | Provides the source code.                                                 |
| `data/`      | Contains the dataset - training (with augmentation), validation, test.    |
| `models/`    | Saves training models according to the model architecture.                |


## Installation
~~~~{.python}
git clone https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes.git
cd Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/codes
pip install -r requirements.txt
~~~~

## How To Run

#### * Dataset

Download the dataset from [here](https://drive.google.com/drive/folders/1PkaNq0hh7OimMKJmGqwa0y2gQbkGRbOO?usp=sharing) and place the *data* folder in the current directory. It has a *data.npz* file containing all the images - training (with augmentation), validation and the test dataset. The size of this file is 1.88GB. 

#### * Training

For Training run the *train.py* and provide a model name i.e. *EfficientNetB0*. 
~~~~{.python}
python train.py --name EfficientNetB0
~~~~
Other parameters can be passed as arguments. 
~~~~{.python}
python train.py --name EfficientNetB0 --ep 500 --batch 16 
~~~~

#### * Pretrained models

Get all the pretrained models from [here](https://drive.google.com/drive/folders/16HtdoMFrDejoFo8WATZ5xa3gGaRWAxMb?usp=sharing). Place the *models* folder in the current directory. It has subfolders with the four AI architectures names (ConvNeXtTiny, EfficientNetB0, MobileNetV2, ResNet50) which have been used to train the dataset. Each subfolder contains a *.h5* file, storing the model architecture with the weights and parameters. Loading these pretrained models, all the test results and Grad-CAMs can be re-genrated. 

Run the *test_model.py* and provide a model name i.e. *EfficientNetB0*, model directory. 
~~~~{.python}
python test_model.py --name EfficientNetB0 --model ../models/EfficientNetB0/model00000533.h5
~~~~

#### * Generate the TSNE Plot

To generate a TSNE plot with a trained model, put the model directory link to the *model* variable in *tsne.py* file and then run *tsne.py* file. 

~~~~{.python}
python tsne.py --name EfficientNetB0 --model ../models/EfficientNetB0/model00000533.h5
~~~~

It will generate the TSNE plot which will be saved to the current folder with the name *tsne.png*


#### * Test an Image and Generate Grad-CAM

To test a trained model with an image, run *test_image.py* file with and provide a model name i.e. *EfficientNetB0*, model directory and the directory of the test image
~~~~{.python}
python test_image.py --name EfficientNetB0 --model ../models/EfficientNetB0/model00000533.h5 --test test_image.jpg
~~~~

It will print the prediction with the confidence and generate the Grad-CAM which will be saved to the current folder with the name *gradCam_test_image.jpg*.

We are also hosting this classification test process in the [website](https://mosquito-classifier.firebaseapp.com) where you can upload an image and check the results with the Grad-CAM within less than a minute. The model in this website is the *EfficientNetB0*. 
