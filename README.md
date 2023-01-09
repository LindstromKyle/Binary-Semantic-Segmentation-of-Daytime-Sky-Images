# Binary Semantic Segmentation of Daytime Sky Images

Training a computer vision network to identify clouds in images from the Singapore Whole Sky Imaging Segmentation Database

![](./Images/img017.png)

## **Introduction**

There are currently close to 1 million people across the United States and Canada that use sign language as their primary form of communication. However, less than 1% of the population in each country understands sign language, which can make interactions for the deaf or hard of hearing difficult. This project's focus will be to address this issue with computer vision and machine learning. My hypothetical stakeholder in this context will be the Google Translate team— they would like to expand their tool in order to incorporate the translation of live action sign language gestures. I’ll be photographing and recording myself performing various ASL actions, and training deep neural networks to detect, track, and translate my symbols and gestures.

## **Data Collection and Labeling**

For this project, I decided to create my own dataset of images and videos to train my models with. This dataset will be split into two sections— a set of still frame photos where to goal will be to identify and box the hand symbol that appears on screen, and a set of videos (30 frames each) with the goal of identifying the gesture being performed and writing the translated English equivalent to screen. 

### **Still Frames**

I used the OpenCV python library to collect 120 unique images of myself holding up the international signs for thumbs up, thumbs down, and peace— 40 instances of each. In order to annotate these images, I employed another python library called LabelImg to manually box the symbols and apply the correct training labels. One example of this process is shown in *Figure 1*:

![](./Images/image3.png)

> *Figure 1: Here, boxes have been drawn around the two instances of ‘Thumbs Down’ I’m holding up, and the label has been assigned to each.*

### **Action Gestures**

Next, I again used OpenCV to collect 240 unique 30-frame videos of myself performing ASL gestures. These videos are split into 4 categories: ‘Hello’, ‘Nice to meet you’, ‘Thank you’, and ‘Goodbye’. However, the classification task this time is different— instead of identifying and boxing an object in a single image, the gesture needs to be tracked through all 30 frames in order to make a detection. For this task, I used a python library called MediaPipe, which marks ‘keypoints’ in each image to track your face, arms, and hands. These same keypoints are given coordinates in each of the 30 frames in a sequence, and thus the positioning and movement of your body can be followed. An example of this process is shown in *Figure 2*:

![](./Images/image1.png)

> *Figure 2: One frame of a video in which MediaPipe has labeled the keypoints on my face, arms, and hands.* 

## **Model Training**

### **Single Frame Detections**

For this task, I used the SSD MobileNet V2 FPNLite 320x320 model architecture from the TensorFlow 2 Detection Model Zoo. The benefit of this model is its speed, averaging detections in 22 ms or less. This is important when you need to make many detections every second, as is the case in this project. The ultimate goal was to identify, box, and track symbols as you move them around in real time, and this architecture suits that need perfectly. *Figure 3* shows the training loss in intervals of 100 time steps, generated from TensorBoard logs:

![](./Images/image7.png)

> *Figure 3: TensorBoard plot of the total loss as the model is trained through iteration 2,000*

To evaluate the model’s performance on unseen data, it was fed images from the test set and its predictions were compared to the ground truth created from the labeling process. On a set of 24 images, the model was 100% accurate. One example of a correct prediction is shown in *Figure 4*:

![](./Images/image2.png)

> *Figure 4: The model correctly detects and boxes these two unseen instances of ‘Peace’ that I am holding up*

Finally, the model was put to the test on a live video feed of myself holding these symbols up, and moving them around the screen to demonstrate its ability to quickly track in real time. The results are shown in *Figure 5*:

![](./Images/Object_Detection.gif)

> *Figure 5: Real time predictions of the model from my webcam. It is able to make detections and track my hands around the screen with great accuracy*

### **Action Gestures**

In order to detect more complex motions across the 30 frame videos, I used a sequential Keras model with LSTM layers. While it may seem like a task with more input parameters than a still image, MediaPipe only stores the locations of each keypoint in the images rather than every individual pixel. This vastly reduces the number of inputs when compared to the 320x320 resolution photos being fed to the MobileNet. Because of this, the Keras sequential model will suffice and training speed is much higher. *Figure 6* shows the loss and accuracy of the model on the training and validation data across 500 epochs:

![](./Images/image4.png)

> *Figure 6: Loss and accuracy of the model on the training and validations sets through 500 epochs*

This model also performed well on unseen data, with an accuracy of 100% on 24 test instances. The confusion matrix for the test data is shown in *Figure 7*:

![](./Images/image5.png)

> *Figure 7: Confusion matrix for the 24 instances of test data split between 4 labels*

And finally, this model was also used to detect live action gestures from my webcam and translate them in real time by printing the English equivalent to the screen. *Figure 8* shows a recording of this process:

![](./Images/Action.gif)

> *Figure 8: Real time predictions of the model from my webcam, and the printed sentence translated directly on the screen*

## **Conclusions**

I believe these models indeed demonstrate the capability of neural networks to identify, track, and translate live action sign language. Their predictive power is exceptional and their speed is sufficient for normal conversation. However, there are some limits to be aware of— First, every picture in my training dataset is of me. Second, they were all taken around the same time of day with similar lighting and backgrounds. While the models still performed well in my tests on other individuals and in different environments, to obtain a truly robust model the dataset should contain a wide variety of these parameters. Regardless, this project succeeds as a proof of concept and could be a valuable jumping off point for future iterations. Next steps may include adding a more comprehensive suite of ASL gestures and exploring the possibility of combining the two models into one. I’d also like to give thanks and credit to Nicholas Renotte, as several of his python scripts were adapted for this project. 

## **Links**

[Action Gesture Jupyter Notebook](https://github.com/LindstromKyle/Flatiron-Capstone/blob/main/Action/Action.ipynb)  

[Single Frame Jupyter Notebook](https://github.com/LindstromKyle/Flatiron-Capstone/blob/main/SingleFrame/Single_Frame.ipynb)  

[Non Technical Presentation (For video playback check README)](https://github.com/LindstromKyle/Flatiron-Capstone/blob/main/Presentation/Non%20Technical%20Presentation.pdf)  




