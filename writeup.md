# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./distribution.jpg "Visualization"
[image2]: ./new-images/14.jpg "Stop"
[image3]: ./new-images/7.jpg "Speed limit (100km/h)"
[image4]: ./new-images/4.jpg "Speed limit (70km/h)"
[image5]: ./new-images/12.jpg "Priority road"
[image6]: ./new-images/26.jpg "Traffic signals"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SS47816/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how different classes of traffic signs are distributed among training, validation and testing sets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

There was **NO** preprocessing for the image data in my model

In this step, I experimented some preprocessing tricks to the training images such as normalizing(`(img - 128)/128`), grayscaling, and rotating the images etc. However, I observed that the accuracy of the model dropped significantly after applying preprocessing to the raw images. The model accuracy with processed images generally are 70%~90%, while that without preprocessing could reach above 90~95% (The best I have got so far is 95.6%).

One possible reason could be that I have added dropout function at all layers in my model. Normallization of images will significantly reduce the magnitude of differences between pixels, between features, and between different classes. Since the model was already experiencing lost of information due to dropout functions, further reducing the magnitude of differences may not be a good idea. Same concept for the other preprocessing methods. Hence, in the final implementation, I choose not to apply any preprocessing to the raw images to preserve their original information. I believe that using preprocessed image data with a low learning rate, plus a bit of tuning, and reduce the number of dropout layers, could result in better performance. However, due to time constrain, I dicided to keep the model this way and improve it later on.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on LeNet-5 Architecture and it consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x6 	|
| RELU					|											    |
| Dropout				|											    |
| Max pooling	      	| 2x2 stride,  outputs 15x15x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 11x11x16    |
| RELU					|											    |
| Dropout				|											    |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
| Flatten       		| Input = 5x5x16, Output = 400  				|
| Fully connected		| Input = 400, Output = 120  					|
| RELU					|											    |
| Dropout				|											    |
| Fully connected		| Input = 120, Output = 84  				    |
| RELU					|											    |
| Dropout				|											    |
| Fully connected		| Input = 84, Output = 43  					    |
| Softmax				|         									    |
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I am using the following parameters:

* number of epochs = 50
* batch size = 32
* dropout keep_prob = 0.8
* learning rate = 0.00004

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.4%
* test set accuracy of 93.6%

This model is based on LeNet-5 Architecture, with some minor changes:
* Added dropout layers after conv1, conv2, fc1, and fc2
* Changed the first layer's kernel size from 5x5x6 to 3x3x6

Initially, I added dropout layers to only the first conv layer, and found it very useful. Then I applied dropout to all layer and tuned the `keep_prob=0.8`, so that throughout the entire network not too much information getting discarded during the training. The network performance reached 90%+ after this modification.

To further imporve the network, I tried to add more layers in it. I tried adding one convolutional layer, adding one dense layer, and tried add both together. I also tried to increase the number of parameters at all layers to make the network more complex and hoping it can be more powerful. However, the accuracy on the validation set didn't improve much. Maybe solely increase the network complexity without feeding more data isn't effective. Then I discareded these changes, and only tuned the kernel size of the first layer, form 5x5x6 to 3x3x6.

Generally, during the training, I tuned the hyperparameters such as the epoch,  batch size, dropout rate, and the learning rate. I tired to keep a relatively low learning rate and a small batch size to train the model slower and deeper. In addition, I tried to keep the training time within 10min or so so that I can try more configurations and compare their performance roughly.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The third image might be difficult to classify because most part of the useful features are covered by reflections

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop      		    | Stop   									    | 
| Speed limit (100km/h) | Speed limit (80km/h)							|
| Speed limit (70km/h)	| Speed limit (70km/h)							|
| Priority road     	| Priority road 				 				|
| Traffic signals		| Traffic signals      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| .0     				| No vehicles 									|
| .0					| No entry							            |
| .0	      			| Speed limit (80km/h)					 		|
| .0				    | Bicycles crossing     						|


For the second image the model is no so sure that this is a Speed limit (80km/h) sign (probability of 0.39), however, the correct answer is at the third place with a probability of 0.18. The top five soft max probabilities were
 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .39         			| Speed limit (80km/h)   						| 
| .20     				| No passing for vehicles over 3.5 metric tons 	|
| .18					| Speed limit (100km/h)				            |
| .06	      			| No passing        					 		|
| .05				    | Speed limit (60km/h)     						|


For the third image, the model is very sure that this is a Speed limit (70km/h) sign (probability of 0.99), though the image contains strong reflection. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (70km/h)   						| 
| .0     				| Speed limit (30km/h) 							|
| .0					| Speed limit (50km/h)							|
| .0	      			| Speed limit (120km/h)					 		|
| .0				    | Speed limit (20km/h)     						|


For the fourth image, the model is very sure that this is a Speed limit (70km/h) sign (probability of 1.0), and the image is very clear. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road  					        	| 
| .0     				| Yield 							            |
| .0					| No vehicles							        |
| .0	      			| Speed limit (30km/h)					 		|
| .0				    | End of all speed and passing limits     		|


For the fifth image, the model is very sure that this is a Speed limit (70km/h) sign (probability of 0.99), and the image contains a traffic light sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Traffic signals       						| 
| .0     				| Bumpy road        							|
| .0					| General caution		    					|
| .0	      			| Pedestrians				        	 		|
| .0				    | Road narrows on the right     				|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

