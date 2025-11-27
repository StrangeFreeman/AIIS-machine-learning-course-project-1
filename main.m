clear;
clc;
close all;

% feature table
load('train/X_train.txt') %load x train
load('test/X_test.txt')   %load x test

% ground truth
load('train/y_train.txt') %load y train
load('test/y_test.txt')   %load y test

featuretable = [X_train; X_test];
GT = [y_train; y_test];
FT_GT = [featuretable, GT];
index = randi([1 10], 10299, 1 );

% gpu acc dataset prerequirements
featuretable2= gpuArray([X_train; X_test]);
GT2 = gpuArray([y_train; y_test]);
FT_GT2= gpuArray([featuretable2, GT2]);
index2 = gpuArray(randi([1 10], 10299, 1 ));

for kfold = 1:10
    test = (index==kfold);
    train = ~test;
    FT_GT_train = FT_GT(train, :);
    FT_GT_test  = FT_GT(test, :);
    
    % SVM model setup
    t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 2);
    modelsvm = fitcecoc(FT_GT_train(:,1:561), FT_GT_train(:, 562), 'Learners', t);
    predictmodel = predict(modelsvm, FT_GT_test(:, 1: 561));
    
    % confusion_matrix
    eval(['confusionmatrix_', num2str(kfold), '=confusionmat(FT_GT_test(:,562), predictmodel);'])
    
    % Accuracy, Sensitivity, Precision
    eval(['ACC=sum(diag(confusionmatrix_', num2str(kfold),'))/sum(sum(confusionmatrix_', num2str(kfold),'));'])
    for i = 1:6
        eval(['Sen(i, 1) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(i,:));'])
        eval(['Pre(1, i) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(:,i));'])
    end
    TotalMatrix(kfold, 1, 1) = ACC;
    TotalMatrix(kfold, 2, 1) = mean(Sen(:,1));
    TotalMatrix(kfold, 3, 1) = mean(Pre(1,:));



    % naive bayes model setup
    modelnb = fitcnb(FT_GT_train(:,1:561), FT_GT_train(:, 562));
    predictmodel = predict(modelnb, FT_GT_test(:, 1: 561));
    
    % confusion_matrix
    eval(['confusionmatrix_', num2str(kfold), '=confusionmat(FT_GT_test(:,562), predictmodel);'])
    
    % Accuracy, Sensitivity, Precision
    eval(['ACC=sum(diag(confusionmatrix_', num2str(kfold),'))/sum(sum(confusionmatrix_', num2str(kfold),'));'])
    for i = 1:6
        eval(['Sen(i, 1) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(i,:));'])
        eval(['Pre(1, i) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(:,i));'])
    end
    TotalMatrix(kfold, 1, 2) = ACC;
    TotalMatrix(kfold, 2, 2) = mean(Sen(:,1));
    TotalMatrix(kfold, 3, 2) = mean(Pre(1,:));



    % decision tree model setup
    modeltree = fitctree(FT_GT_train(:,1:561), FT_GT_train(:, 562));
    predictmodel = predict(modeltree, FT_GT_test(:, 1: 561));
    
    % confusion_matrix
    eval(['confusionmatrix_', num2str(kfold), '=confusionmat(FT_GT_test(:,562), predictmodel);'])
    
    % Accuracy, Sensitivity, Precision
    eval(['ACC=sum(diag(confusionmatrix_', num2str(kfold),'))/sum(sum(confusionmatrix_', num2str(kfold),'));'])
    for i = 1:6
        eval(['Sen(i, 1) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(i,:));'])
        eval(['Pre(1, i) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(:,i));'])
    end
    TotalMatrix(kfold, 1, 3) = ACC;
    TotalMatrix(kfold, 2, 3) = mean(Sen(:,1));
    TotalMatrix(kfold, 3, 3) = mean(Pre(1,:));



    % KNN
    modelknn = fitcknn(FT_GT_train(:,1:561), FT_GT_train(:, 562), 'NumNeighbors', 3);
    predictmodel = predict(modelknn, FT_GT_test(:, 1: 561));

    % confusion_matrix
    eval(['confusionmatrix_', num2str(kfold), '=confusionmat(FT_GT_test(:,562), predictmodel);'])
    
    % Accuracy, Sensitivity, Precision
    eval(['ACC=sum(diag(confusionmatrix_', num2str(kfold),'))/sum(sum(confusionmatrix_', num2str(kfold),'));'])
    for i = 1:6
        eval(['Sen(i, 1) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(i,:));'])
        eval(['Pre(1, i) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(:,i));'])
    end
    TotalMatrix(kfold, 1, 4) = ACC;
    TotalMatrix(kfold, 2, 4) = mean(Sen(:,1));
    TotalMatrix(kfold, 3, 4) = mean(Pre(1,:));

end    

% TotalMatrix spec
% TotalMatrix(kfold, Acc/Sen/Pre, SVM/NB/Tree/KNN)
for i = 1:4 % 1=SVM, 2=NB, 3=Decision Tree 4=KNN
    FinalMatrix(i,1) = mean(TotalMatrix(:, 1, i)); % Acc
    FinalMatrix(i,2) = mean(TotalMatrix(:, 2, i)); % Sen
    FinalMatrix(i,3) = mean(TotalMatrix(:, 3, i)); % Pre
end

% use gpu ac the process
for K = 1:1:100
for kfold = 1:10
    test = (index2==kfold);
    train = ~test;
    FT_GT_train2 = FT_GT2(train, :);
    FT_GT_test2  = FT_GT2(test, :);
    modelknn = fitcknn(FT_GT_train2(:,1:561), FT_GT_train2(:, 562), 'NumNeighbors', K);
    predictmodel = predict(modelknn, FT_GT_test2(:, 1: 561));
    
    % counting confusionmatrix
    eval(['confusionmatrix_', num2str(kfold), '=confusionmat(FT_GT_test2(:,562), predictmodel);']);

    % Accuracy, Sensitivity, Precision
    eval(['AccMatrix(kfold, K)=sum(diag(confusionmatrix_', num2str(kfold), '))/sum(sum(confusionmatrix_', num2str(kfold), '));'])
    for i = 1:6
        eval(['Sen(i, 1) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(i,:));'])
        eval(['Pre(1, i) = confusionmatrix_', num2str(kfold), '(i,i)/sum(confusionmatrix_', num2str(kfold), '(:,i));'])
    end
    
    knnCMatrix_fold(kfold,1,K) = mean(Sen(:,1));
    knnCMatrix_fold(kfold,2,K) = mean(Pre(1,:));

end

knnMatrix(K,1) = mean(AccMatrix(:, K)); % average acc
knnMatrix(K,2) = mean(knnCMatrix_fold(:, 1, K)); % average sen
knnMatrix(K,3) = mean(knnCMatrix_fold(:, 2, K)); % average pre

end

% display svm nb decision_tree results
disp(FinalMatrix);

% display knn results using plot
tiledlayout(3,1);
nexttile;
plot(knnMatrix(:, 1));
title('average Acc, KNN')

nexttile;
plot(knnMatrix(:, 2));
title('average Sen, KNN')

nexttile;
plot(knnMatrix(:, 3));
title('average Pre, KNN')


% disp(knnMatrix(:, 1));
% disp(knnMatrix(:, 2));
% disp(knnMatrix(:, 3));