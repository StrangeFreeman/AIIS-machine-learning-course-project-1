# AIIS-machine-learning-course-project-1
this is a AIIS machine learning course project 1 using matlab

## pre requirements
- dataset
    - UCI HAR dataset
        - Human Activity Recognition Using Smartphones Dataset
        - Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K.

- matlab add-on
    - statistics and machine learning toolbox
    - parallel computing toolbox (optional)

## the dataset includes the following files
- `features_info.txt`: Shows information about the variables used on the feature vector.
- `features.txt`: List of all features.
- `activity_labels.txt`: Links the class labels with their activity name.
- `train/X_train.txt`: Training set.
- `train/y_train.txt`: Training labels.
- `test/X_test.txt`: Test set.
- `test/y_test.txt`: Test labels.

The following files are available for the train and test data. Their descriptions are equivalent. 

- `train/subject_train.txt`: Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 
- `train/Inertial Signals/total_acc_x_train.txt`: The acceleration signal from the smartphone accelerometer X axis in standard gravity units 'g'. Every row shows a 128 element vector. The same description applies for the 'total_acc_x_train.txt' and 'total_acc_z_train.txt' files for the Y and Z axis. 
- `train/Inertial Signals/body_acc_x_train.txt`: The body acceleration signal obtained by subtracting the gravity from the total acceleration. 
- `train/Inertial Signals/body_gyro_x_train.txt`: The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second. 

## machine learning steps
- Before machine learning model, I create a big feature table
    1. import datasets x_train, x_test, y_train, y_test
    2. combine y_train & y_test into a ground truth (GT) dataset, so as x_train & x_test into feature table (FT) dataset

- After I created the big feature table, I setup random index for k-fold cross validation.

- Then inside k-fold cross validation, is creating machine learning model and create predictmodel using
    - SVM
    ```
    t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 2);
    modelsvm = fitcecoc(FT_GT_train(:,1:561), FT_GT_train(:, 562), 'Learners', t);
    predictmodel = predict(modelsvm, FT_GT_test(:, 1: 561));
    ```

    - naive bayes
    ```
    modelnb = fitcnb(FT_GT_train(:,1:561), FT_GT_train(:, 562));
    predictmodel = predict(modelnb, FT_GT_test(:, 1: 561));
    ```

    - decision tree
    ```
    modeltree = fitctree(FT_GT_train(:,1:561), FT_GT_train(:, 562));
    predictmodel = predict(modeltree, FT_GT_test(:, 1: 561));
    ```

    - kNN
    ```
    modelknn = fitcknn(FT_GT_train(:,1:561), FT_GT_train(:, 562), 'NumNeighbors', 3);
    predictmodel = predict(modelknn, FT_GT_test(:, 1: 561));
    ```

- After test models using GT dataset, I create confusion matrix on every machine learning model

- And at the end I caculate Accuracy, Sensitivity, Precision persentages average, and made it a 3x4 matrix name "FinalMatrix"

- Also I run a K value between 1 to 100 on KNN model, and the results is KNN(1-100).png
    - btw I use gpu acceleration on this test to made the whole process faster

## final output description

a 3x4 matrix

| Model | Accuracy | Sensitivity | Precision |
|-------|---------|---------|---------|
| SVM   | 0.9906  | 0.9911  | 0.9912  |
| NB    | 0.7401  | 0.7468  | 0.7846  |
| DT    | 0.9365  | 0.9350  | 0.9351  |
| KNN   | 0.9699  | 0.9713  | 0.9715  |


### program output

|     |     |     |
|-----|-----|-----|
| 0.9906  | 0.9911  | 0.9912  |
| 0.7401  | 0.7468  | 0.7846  |
| 0.9365  | 0.9350  | 0.9351  |
| 0.9699  | 0.9713  | 0.9715  |
