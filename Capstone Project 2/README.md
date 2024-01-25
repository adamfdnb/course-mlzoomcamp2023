# Venomous Labyrinth: Classifying Venomous Snakes
## Venomous Snake Classification project

<p align="center">
  <img src="https://github.com/adamfdnb/course-mlzoomcamp2023/blob/main/Capstone%20Project%202/images/collage_v2.jpg">
</p>

*collage from a dataset*

#### This repository contains a Capstone project 2 conducted as part of the [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) online course designed and taught by [Alexey Grigorev](https://github.com/alexeygrigorev) and his team from [DataTalks.Club](https://datatalks.club/). This project took 2 weeks to complete. The idea behind this project is to implement everything we have learned over the past several weeks and many hours exploring deep learning.

## Contents:
1. problem & goal description
2. about the dataset
3. problem solving approach<br>
3.1 EDA to understand the dataset<br>
3.2 Training the model<br>
3.3 Implementing the model in the cloud<br>
4. development system<br>
   4.1 Deploy Model with Flask<br>
   4.2 Deploying Model Locally with Docker<br>
   4.3 Cloud Deploying <br>
5. Summary with conclusions

## Setup
Follow the instructions in [SETUP.md](./SETUP.md)  

### Clone the repo
Open a terminal and execute:  
`git clone https://github.com/MarcosMJD/ml-mango-classification.git`

### 1. Problem & Goal Description

The "Venomous Snake Classification" project aims to develop an algorithm to accurately categorize snakes through image analysis. With a dataset focused mainly on snake species found in India, the goal is to develop an effective system capable of distinguishing between different snake species and determining their venomousness.

Snakes, comprising more than 3,500 species, exhibit a variety of characteristics and are distributed throughout the world, except Antarctica. Reaching sizes ranging from a few centimeters to several meters, they prey on a wide variety of animals, including other reptiles, mammals, birds, amphibians and fish.

Using advanced image recognition techniques, the project aims to streamline the snake identification process, providing valuable insight into the diverse population of snakes found in India. I would like the resulting algorithm to be able to contribute to snakebite prevention and treatment strategies by quickly identifying venomous snake species.

## 2. About the Dataset and technologies

You can get the dataset from [kaggle](https://https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india)
I have prepared a code in notebook that imports a set of data from within notenook. You will need an individual API Token downloaded from Kaggle. [Details](https://github.com/Kaggle/kaggle-api)

The Snake dataset India contains  
Train dataset<br>
```Found 1775 images belonging to 2 classes.```<br>
Test dataest<br>
```Found 269 images belonging to 2 classes.```<br>

Clssses<br>
```{'Non Venomous': 0, 'Venomous': 1}```<br>

Images format <br>
```<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=400x400 at 0x7F863819B3A0>```
 
 ### Technologies
- Python
- Numpy, Pandas, MatplotLib, Seaborn
- PIL, os
- Tensorflow / Tensorflow Lite  
- Keras  
- Models: Xception, MobileNetV2
- Flask and FastAPI
- Docker / docker-compose  
- Streamlit  
- AWS EKS  
  

## 3. Problem solving approach
### 3.1 EDA to understand the dataset

I performed the following EDA for this dataset:
+ Loading the data: Loading the data into the application for further analysis.
+ Create a data frame to read information about a set of files
+ Creating paths to the files
+ Analyzing the distribution of files in the data set by class
+ Loading a sample photo from the training set

## 3.2. Training the model

I tested seve models and tuned hyperparameters to optimize their performance. A short summary: 

### Xception model

+ The Random Forest model tends to overfit the training data, as evidenced by its perfect accuracy on the training datasets but lower accuracy on the testing datasets. Logistic Regression and Naive Bayes exhibit more stable performance but struggle to predict the positive class, especially on the original data. The models trained on the cleaned dataset do not consistently outperform their counterparts trained on the original data. Naive Bayes, the accuracy on the training dataset is 63.16%, indicating moderate performance. The performance on the testing dataset is consistent with the training dataset, with an accuracy of 62.96%.

### MobileNetV2

+ It seems like the SVM model with different values of the regularization parameter C (0.001, 0.01, 0.1, 1, 10, 100) is not performing well on both the training and testing sets. The accuracy is around 60-63%, and the confusion matrix and classification report show that the model is not effectively distinguishing between the two classes (0 and 1). The precision, recall, and F1-score for class 1 are consistently low, indicating that the model struggles to correctly identify instances of class 1.

### XGBClassifier with GridSearchCV

+ The cross-validation results indicate that the XGBClassifier model is able to achieve a reasonable level of accuracy in predicting the potability of drinking water, with average test set accuracies ranging from 68.54% to 72.65%. The best performing hyperparameters vary depending on the dataset (original or cleaned), but generally involve a learning rate (eta) of 0.1, a maximum depth between 3 and 6, a minimum child weight between 1 and 7, a number of estimators between 25 and 100, and a subsample of 0.7.
+ The models trained on the cleaned dataset generally achieved slightly higher accuracy compared to the models trained on the original dataset. This suggests that the cleaning process may have removed some noise or irrelevant features that were negatively impacting the model's performance.
+ The XGBClassifier model shows promising results for predicting the potability of drinking water. Further optimization of hyperparameters and exploration of different data preprocessing techniques could potentially improve the model's performance.

  + Finally, preparing the data, training the XGBClassifier model and saving it to the ``` model_wpp.model ``` file was prepared in the ``` train.py ``` file. 
Having the model prepared, you can easily import it in the future to make predictions on new data.


## 5. Development system
### To deploy Model with Flask 

1. To activate a virtual environment using Pipenv on a Linux system, follow these steps:
	- Open a terminal in your Linux system.
 	- Navigate to the directory where your project is located or create a new directory for your project if you haven't already.
  	- Use the + ```pipenv install``` command to create a new virtual environment and install the project's dependencies.
     + This command will automatically create a Pipfile and Pipfile.lock and set up a virtual environment in your project directory.
       
2. To activate the virtual environment, use the
   	+ ```pipenv shell```
	+ After running this command, you will be inside the activated virtual environment, which means that all Python commands and packages installed within this environment will be available.

3. This line of code is to install all the necessary dependencies listed in the Pipfile files of the virtual environment.
   	+ ``` pip install name_of_package ```
   
4. Run service app (predict.py)
   	+ ``` python predict.py ```
 	+ ``` pipenv run python predict.py ``` / if using virtual environment

5. Run test file in a sepearate virtual environment terminal (test.py)
	+ ``` python test.py ```
 	+ ``` python test_webapp.py ``` / if you are using network services / remember to specify your own address

![alt text](images/testapp_local.png)

	+ ``` python test_webapp.py ``` / if you are using network services / remember to specify your own address
![alt text](images/test_webapp_c.png)

    

### Deploying Model Locally with Docker
#### Install and run docker on local machine
About Docker [Docker overview](https://docs.docker.com/get-started/overview/)
1. Installing Docker
Docker is a tool that makes it easy to create, deploy and run applications in containers. Containers are isolated units that contain everything you need to run an application,including code, the execution environment, libraries and other dependencies. Overall, Docker speeds up development processes, makes it easier to deploy and manage applications, and improves the consistency of the environment between different stages of the application lifecycle.

	- Ubuntu 

```bash
sudo apt-get install docker.io
```
Install and run docker, follow [more information about installation](https://docs.docker.com/engine/install/ubuntu//) <br>
or using the resources of [DataTalskClub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md)
 
2. Build docker image in terminal

  +  ``` docker build -t water-predict_cp1 . ```

Remember that you must be in the project folder :
You can check what folder you are currently in in the Linux terminal using the pwd command. pwd stands for "print working directory" and will display the full path to the current directory.

```bash
pwd
```

![alt text](images/docker_b_c.png)

3. Run docker image:
  - ``` docker run -it --rm -p 9696:9696 water-predict_cp1 ```

![alt text](images/docker_test2.png)

### Cloud Deploying 

To deploy the application in the cloud, I used [Render](https://docs.render.com/docs/docker), which is a unified cloud for creating and running all applications and websites

![alt text](images/webapp_c.png)

![alt text](images/webapp_3.png)

![alt text](images/webservice_c.png)


5. Run test file test water quality prediction app in cloud
	+ ``` python test_webapp.py ``` / if you are using network services / remember to specify your own address
 	+ in order to improve the transfer of data for prediction, it is possible to transfer data in various formats: list, dict, DataFrame or numpy array

![alt text](images/test_webapp_c.png)


## 6. Summary with conclusions

### Data preparation:
Proper data preparation is crucial to the effectiveness of the model.
Converting column names to lowercase, converting spaces to underscores and converting values in the evaluation column allowed for better data representation.

### Analysis and comparison of models:
I tested four different models: Logistics Regresion, Support Vector Machine, Random Forest Classifier, Gaussian Naive Bayesr, K-Nearest Neighbors Classifier and XGBClassifier with GridSearchCV. I used different hyperparameters to optimize each model.

### Model evaluation:
I evaluated the models using various metrics such as Accuracy Score, Classification Report, Confusion Matrix, ROC AUC Score, Precision, Recall, ROC Curve, Mean Squared Error, R-squared, F1 Score and AUC.
I tested the models on different subsets of data (training, validation, testing) to evaluate their overall performance and two datasets

### Model selection:
Based on the test results, I decided to use the XGBClassifier model with following hyperparameters 
```
{
    `eta': 0.1,
    'max_depth': 5,
    'min_child_weight': 5,
    'n_estimators': 25,
    'subsample': 0.7,
    'objective': 'binary:logistic',
    'use_label_encoder': False
}
```

### Model:
The final XGBClassifier model was saved in the model_mqp.model, which makes it easy to reuse the model for forecasting on new data.

### Model performance:
The final model achieved high accuracy on the validation set, suggesting that the model is effective in predicting milk class classes based on available characteristics. 
I will continue to develop this project using neural networks. 
