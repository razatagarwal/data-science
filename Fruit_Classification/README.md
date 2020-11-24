## TITLE <br>
Comparison of machine learning models for Fruit Classification

## INTRODUCTION <br>
The project focuses on the classification of fruits, given their image, from 95 different fruits. The dataset is taken from https://www.kaggle.com/moltean/fruits. Applying different classification models to find the best one.

## DATASET <br>
Total number of images: 65429. <br> 
Training set size: 48905 images (one fruit per image). <br>
Test set size: 16421 images (one fruit per image). <br>
Number of classes: 95 (fruits). <br>
Image size: 100x100 pixels. <br>

## SETTING UP <br>
```
make requirements
```

## RUN PROJECT <br>
```
make run
```

## RESULTS <br>

Model | Accuracy
--- | ---
Random Forest | 87.77%
Support Vector Machine | 93.28%
Softmax | 61.87%
Naive Bayes | 62.42%
KNN | 91.74%
