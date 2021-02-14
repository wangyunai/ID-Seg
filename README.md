# ID-Seg
This repositories include python code for infant deep learning segmentation framework we developed. 

## Training steps
First, we trained our model on a big dataset (dhcp dataset) using [pretrain](pretrain.py) script on three main plain views (axial, coronal, and sagittal) and tested the performance of our 3D model using [pretrain_validation](pretrain_validation.py) code.

After that, we used [train](train.py) script for transfering our knowledge on a smaller dataset (MCRIB), where we used Leave One Out Cross Validation (LOOCV) technique. Again, we trained our model on three different plain views. Using [save_prediction](save_prediction.py) code, we saved predictions of our model on our dataset and then used [test](test.py) script to calculate accuracy and dice scores of results.

## How to use
The codes are ready to use. You just need to install needed requirements and provide the paths to dataset as instructed by comments in the code.
Requirements are nibabel, scikit-image, scikit-learn and pytorch.
