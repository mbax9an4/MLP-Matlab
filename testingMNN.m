function [cvErr] = testingMNN()

load congress;
folds = cvpartition(labels,'kfold',20);
err = zeros(folds.NumTestSets,1);
errorNN = zeros(folds.NumTestSets,1);
lr = zeros(folds.NumTestSets,1);
learningRate = 0.01;

for i = 1:folds.NumTestSets
    trainingIndex = folds.training(i);
    testingIndex = folds.test(i);
    trainingData = data(trainingIndex,:);
    trainingLabels = labels(trainingIndex,:);
    testingData = data(testingIndex,:);
    testingLabels = labels(testingIndex,:);
    
    learningRate = learningRate+0.05;
    err(i) = MNNTesting(trainingData, trainingLabels, testingData, testingLabels, learningRate);
    errorNN(i) = err(i)/folds.TestSize(i);
    lr(i)=learningRate;
end
cvErr = sum(errorNN)/folds.NumTestSets;

barNN = ones(folds.NumTestSets,1)*std(errorNN);
%barSVM = ones(folds.NumTestSets,1)*std(errorSVM);

errorbar(lr, errorNN, barNN)
figure, plot(err)

clear;
clc;

load breast;
learningRate = 0.2;
folds = cvpartition(labels,'kfold',5);
err = zeros(folds.NumTestSets,1);
errorNN = zeros(folds.NumTestSets,1);
errorSVM = zeros(folds.NumTestSets,1);
tElapsedNN = zeros(folds.NumTestSets,1);
tElapsedSVM = zeros(folds.NumTestSets,1);

for i = 1:folds.NumTestSets
    trainingIndex = folds.training(i);
    testingIndex = folds.test(i);
    trainingData = data(trainingIndex,:);
    trainingLabels = labels(trainingIndex,:);
    testingData = data(testingIndex,:);
    testingLabels = labels(testingIndex,:);
    
    tStartNN = tic;
    err(i) = MNNTesting(trainingData, trainingLabels, testingData, testingLabels, learningRate);
    tElapsedNN(i) = toc(tStartNN);
    errorNN(i) = err(i)/folds.TestSize(i);
    
    model = svm('-t 1 -g 3 -c 1 -e 1 -d 1');
    tStartSVM = tic;
    model = model.train(trainingData,trainingLabels);
    tested = model.test(testingData,testingLabels);
    tElapsedSVM(i) = toc(tStartSVM);
    errorSVM(i) = tested.err();
end
cvErr = sum(errorNN)/folds.NumTestSets;

barNN = ones(folds.NumTestSets,1)*std(errorNN);
barSVM = ones(folds.NumTestSets,1)*std(errorSVM);

figure, errorbar(errorNN,barNN,'r');
hold on, errorbar(errorSVM,barSVM,'b');
figure, plot(tElapsedNN,'r');
hold on, plot(tElapsedSVM,'b');

end
