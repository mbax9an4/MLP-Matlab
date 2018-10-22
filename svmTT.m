function [error] = svmTT()
load biggerdata;

[rows,~] = size(data);

trainingData = data(1:rows/2,:);
trainingLabels = labels(1:rows/2,:);

testingData = data(rows/2+1:rows,:);
testingLabels = labels(rows/2+1:rows,:);

model = svm('-t 2 -g 1.5');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
error = tested.err();
plotboundary(data,labels,model)
end