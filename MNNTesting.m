%function to call the main execution of the multilayered neural network
%implementation
function [noErrors] = MNNTesting(trainingData, trainingLabels, testingData, testingLabels, learningRate)
%load datad;
%[sizeDataSet, ~] = size(data);

%trainingDataSize = int16(sizeDataSet*3/4);

%trainingData = data(1:trainingDataSize,:);
%trainingLabels = labels(1:trainingDataSize);

[trainingEx, features] = size(trainingData);

%testingData = data(trainingDataSize+1:sizeDataSet,:);
%testingLabels = labels(trainingDataSize+1:sizeDataSet);

sizeIL = features+1;
sizeHL = features;
classes = unique(trainingLabels);
sizeOL = size(classes);

weightIH = rand(sizeIL, sizeHL);
weightHO = rand(sizeHL, sizeOL(1,1));

for E = 1:trainingEx
    [weightIH, weightHO] = trainMNN(trainingData(E,:),trainingLabels(E),weightIH,weightHO, classes, learningRate);
end

noErrors = testMNN(testingData, testingLabels, weightIH, weightHO, classes);

disp(noErrors);
end