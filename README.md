# Flower Image Classification using Tensorflow hub

In this notebook, Flower image classification is done with transfer learning approach using pre trained models from tensorflow hub
[MobileNet_v2 and Inception_V3]. Classification is done on 5 different flower species on tensorflow dataset 'tf_flower'. 
After training, performance of models are compared through Training and Validation data Accuracy/Loss plot visualizations.      

## Project Dependencies 

This project requires following libraries to be imported. 

	- tensorflow_hub==0.4.0
	- tensorflow
	- matplotlib
	- numpy
	- tensorflow_datasets


## Data
Flowers dataset is downloaded from tensorflow dataset 'tf_flowers'. After download training data is split into training_set and 
validation_set into ratio of 70:30. Both datasets are further normalized and batched before feeding into model. 
Please note shuffle is only applied to training data. 


## Implementation

Sequential model is defined with feature part of network represented by features extracted from pre trained model using hub.KerasLayer. 
Final output classification layer is added with number of classes and softmax activation. Model is then complied with loss function, 
optimizer and metrics followed by fit function to train model. Model is run for 6 epochs to compare accuracy between both models. 

## Project Observations:

From the visualization plot for Training and Validation Accuracy/Loss, we conclude MobileNet_v2 performed(acc~87%) much better than 
Inception_V3 model(acc~82%).  

### Below are some of the further improvements I think can be done.

	- Adding more layers to fully connected classification layer to improve accuracy. 
	- Overfitting can be reduced by adding Dropout layers, L1/L2 Regularization. 
	- We can try changing different hyperparameter like Epochs, Batch Size, learning rate etc to further improve accuracy of model. 
