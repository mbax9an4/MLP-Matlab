function [] = svmPL()
load biggerdata;

[rows,~] = size(data);

trainingData = data(1:rows/2,:);
trainingLabels = labels(1:rows/2,:);

testingData = data(rows/2+1:rows,:);
testingLabels = labels(rows/2+1:rows,:);

model = svm('-t 1 -g 3 -c 1 -e 1 -d 1');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorPolynomial = tested.err()

model = svm('-t 0 -g 1.5 -c 1.7 -e 1.4');
model = model.train(trainingData,trainingLabels);
tested = model.test(testingData,testingLabels);
errorLiniar = tested.err()
plotboundary(data,labels,model);


figure; plot([errorPolynomial, errorLiniar]);
end