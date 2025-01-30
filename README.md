# FACIAL EXPRESSION RECOGNITION BASED AUTHORIZATION

  This is basically an Authorization model which works based on facial expressions. Face expression recognition method based on a convolutional neural network
(CNN) and an image edge detection is proposed. Firstly, the facial expression image is normalized, and the edge of each layer of the image is extracted in the convolution process. The extracted edge information is superimposed on each feature image to preserve the edge structure information of the texture image. Then, the dimensionality reduction of the extracted implicit features is processed by the maximum pooling method. Finally, the expression of the test sample image is classified and recognized by using a Softmax classifier. To verify the robustness of this method for facial expression recognition under a complex background, a simulation experiment is designed by scientifically mixing the Fer-2013 facial expression database with the LFW data set.

**SYSTEM ARCHITECTURE**

![image](https://github.com/user-attachments/assets/75372084-d592-4b1b-9a1b-5b1df52f452f)


**DATA FLOW**

![image](https://github.com/user-attachments/assets/43acb441-b5eb-4b3e-9b63-6ce9b3fc1a0b)


**FROM PYTHON FILES**

**train.py** - use it to train model based on human expressions. (I used FER 2013 model dataset)
**main.py**  - this file use to get user face samples for expression recognition and mood set for authorization.
**det.py**   - finally we can detect and authorize user based on identity with expression.


**SCREENSHOTS**

![image](https://github.com/user-attachments/assets/db073dfc-5dd6-4c98-b776-13caa2f1a138)

from above i just use my neutral face to register.


![image](https://github.com/user-attachments/assets/1eb1427b-462c-4940-a361-fbc5b4806624)

here we can see only neutral face authorized. same way we can do for neutral, anger, disgust, fear, happiness, sadness, and surprise



