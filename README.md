![](./Images/img017.png)

# Binary Semantic Segmentation of Daytime Sky Images

Training a computer vision network to identify clouds in images from the Singapore Whole Sky Imaging Segmentation Database

## **Introduction**

The goal of this project is to apply semantic segmentation to images of the sky in order to categorize each pixel as belonging either to a cloud or to the background atmosphere. The dataset used is the Singapore Whole Sky Imaging Segmentation Database found [here](http://vintage.winklerbros.net/swimseg.html). The description reads: 

"The SWIMSEG dataset contains 1013 images of sky/cloud patches, along with their corresponding binary segmentation maps. The ground truth annotation was done in consultation with experts from Singapore Meteorological Services. Representative sample images are shown below.

![](./Images/swimseg.jpeg)

All images were captured in Singapore using WAHRSIS, a calibrated ground-based whole sky imager, over a period of 22 months from October 2013 to July 2015. Each patch covers about 60-70 degrees of the sky with a resolution of 600x600 pixels."

## **Results**

The model used was a standard U-net style architecture with a binary output layer, created using Keras. A hold out set of 203 images (20%) was used for validation. Below are 3 sample results from this validation set — each figure contains the original image, the labeled ground truth, and the model's segmentation predictions.

![](./Images/img016.png)

![](./Images/img017.png)

![](./Images/img101.png)

## **Links**

[Original Research Paper by Dev, Lee, and Winkler](https://stefan.winkler.site/Publications/jstars2017.pdf)  

[Jupyter Notebook](https://github.com/LindstromKyle/Binary-Semantic-Segmentation-of-Daytime-Sky-Images/blob/master/Sky_Segmentation.ipynb)  





