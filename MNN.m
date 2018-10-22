%function to call the main execution of the multilayered neural network
%implementation, without providing the training and testing datasets 
function [noErrors] = MNN(learningRate)

%load the dataset to train and test the network on 
load breast;
[sizeDataSet, ~] = size(data);

%set the training dataset to be three quarters of the provided dataset
trainingDataSize = int16(sizeDataSet*3/4);

%create the training dataset and training labels  
trainingData = data(1:trainingDataSize,:);
trainingLabels = labels(1:trainingDataSize);

[trainingEx, features] = size(trainingData);

testingData = data(trainingDataSize+1:sizeDataSet,:);
testingLabels = labels(trainingDataSize+1:sizeDataSet);

sizeIL = features+1;
sizeHL = features;
classes = unique(trainingLabels);
sizeOL = size(classes);

weightIH = 2*rand(sizeIL, sizeHL)-1;
weightHO = 2*rand(sizeHL, sizeOL(1,1))-1;

for E = 1:trainingEx
    for iter = 1:100
    [weightIH, weightHO] = trainMNN(trainingData(E,:),trainingLabels(E),weightIH,weightHO, classes, learningRate);
    end
end

noErrors = testMNN(testingData, testingLabels, weightIH, weightHO, classes);

model = svm('-t 2 -g 1.5 -c 1.7 -e 1.4');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorSVM = tested.err()
end