# Project Road Segmentation

This is the code and report for the Road Segmentation project of the CS-433 class.

## Report

The report can be found in Report.pdf

## Required libraries
To run the notebooks in local you will need the following libraries :
 -  matplotlib
 -  pyTorch v1.5.1 and torchvision v0.6.1 using `conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch`
 -  scikit-image
 -  opencv-python
 
 Add the folders `images`, `groundtruth` and `test_set_images` to the directory
 
 In `run.py` you can update the line 26 `root_dir = ""` with the path to this directory if needed.
 Then you can execute `run.py` in local, which will give a file `final_submission.csv`.
 
## Code organization

The code is split into two main parts:
* The UNet architecture is contained in the `Unet.ipynb` file
* The CNN architecture is contained in the `Road_segmentation.ipynb` or `run.py` file

Both file were run on Google colab and contains a variable `root_dir` which indicates the directory with the data and the pretrained model. This root directory must have the following structure :
- `images/`
    - all images for the training set
- `groundtruth/`
    - all groundtruth for the training set
- `test_set_images/`  
    - `test_i/` : for i from 1 to 50
        - `test_i.png`
- `results/` : the folder to save the prediction
- `convmodel100.pth` : pretrained CNN with 100 epochs
- `modelUnet.pth` : pretrained UNet with 100 epochs

`mask_to_submission.py` transforms the submission found in a `results` directory into a `submission.csv` file

## Pretrained models
`convmodel100.pth` contains the pretrained CNN with 100 epochs
`modelUnet.txt` contains a link to get the pretrained UNet with 100 epochs. The file is too big for GitHub


## Results

`results_cnn/` contains the predictions made by the CNN  
`results_postproc/` contains the postprocessed predictions made by the CNN  
`results_unet/` contains the predictions made by the UNet  

## AIcrowd

The best submission is `Submission #109779`

## Authors

- Simon Wicky simon.wicky@epfl.ch
- Gaultier Lonfat gaultier.lonfat@epfl.ch
- Joachim Dunant joachim.dunant@epfl.ch
