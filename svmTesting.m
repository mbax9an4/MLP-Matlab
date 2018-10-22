function [] = svmTesting()
load biggerdata;

[rows,~] = size(data);

trainingData = data(1:rows/2,:);
trainingLabels = labels(1:rows/2,:);

testingData = data(rows/2+1:rows,:);
testingLabels = labels(rows/2+1:rows,:);

model = svm('-t 2 -g 1.5 -c 1.7 -e 1.4');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorBest = tested.err()

model = svm('-t 1 -g 1.5 -c 1.7 -e 1.4');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorT = tested.err()

model = svm('-t 2 -g 4 -c 1.7 -e 1.4');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorG = tested.err()

model = svm('-t 2 -g 1.5 -c 10 -e 1.4');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorC = tested.err()

model = svm('-t 2 -g 1.5 -c 1.7 -e 2');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorE = tested.err()
plotboundary(data,labels,model);

figure; plot([errorBest, errorT, errorG, errorC, errorE]);
end