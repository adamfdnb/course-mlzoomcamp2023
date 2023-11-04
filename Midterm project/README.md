# Milk Quality Prediction

![Imgur](https://rnz-ressh.cloudinary.com/image/upload/s--9xz0DRPL--/c_scale,f_auto,q_auto,w_1050/v1644382333/4MH034R_copyright_image_252919)

*photo is from https://www.rnz.co.nz/*

#### This repository contains a midterm project conducted as part of the [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) online course designed and taught by [Alexey Grigorev](https://github.com/alexeygrigorev) and his team from [DataTalks.Club](https://datatalks.club/). This project lasted 2 weeks. The idea behind this project is to implement everything we learned in the last 6 weeks of classes.

## Contents:
1. Problem & Goal description
2. information about the dataset
3. approach to solving the problem
	3.1 EDA to understand the dataset
	3.2 Training the model
	3.3 Implementing the model in the cloud
4. information about files and folders in this repository
5. development system
6. how to reproduce this project
7. conclusions
8. references


### 1. Problem & Goal Description
Milk quality is an important factor affecting its nutritional value, taste, shelf life and safety. Evaluation of milk quality can be done on the basis of various physical parameters, such as pH, temperature, taste, odour, fat, turbidity and colour. These parameters can be measured using simple laboratory methods or electronic sensors. However, these methods can be time-consuming, expensive or inaccurate. Therefore, in my opinion, there is a need to develop predictive models that can quickly and accurately predict milk quality based on available data.

The goal of this project is to build and compare different machine learning models that can predict milk quality based on physical characteristics. 

## 2. About the Dataset

You can get the dataset from [kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality/data). 

The data I am using in this project was collected manually from observations. The dataset contains more than 1000 observations.
We can divide the dataset into 7 independent variables, i.e. pH, temperature, taste, odor, fat, turbidity and color an one dependend variable: Grade

1. pH: This characteristic determines the pH of the milk, which ranges from 3 to 9.5.
2. Temperature: This characteristic determines the temperature of the milk, and it ranges from 34'C to 90'C.
3. Taste: This characteristic determines the taste of the milk and takes possible values: 1 (good) or 0 (bad).
4. Odor: This characteristic determines the smell of milk and takes possible values: 1 (good) or 0 (bad).
5. Fat: This characteristic determines the fat content of the milk and takes possible values: 1 (good) or 0 (bad).
6. Turbidity: This characteristic determines the turbidity of the milk and takes possible values: 1 (good) or 0 (bad).
7. Color: *This characteristic determines the color of the milk, which ranges from 240 to 255.

The dependent variable is the grade of milk, which can be good, average or bad. 

8. Grade: This is the target value and takes the values: low_quality, medium_quality or high_quality.

