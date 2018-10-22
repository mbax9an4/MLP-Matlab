function [] = ScaledLRSVM()
load biggerdata;

[rows, columns] = size(data);

stdRows = std(data, 0, 2);
meanRows = mean(data,2);
for i = 1:rows
    for j = 1:columns
        data(i,j) = (data(i,j) - meanRows(i))/stdRows(i);
    end
end

trainingData = data(1:rows/2,:);
trainingLabels = labels(1:rows/2);

testingData = data(rows/2+1:rows,:);
testingLabels = labels(rows/2+1:rows);

modelLR = logreg('iterations', 1,'learningrate', 1);
tStartLR = tic;
modelLR = modelLR.train(trainingData,trainingLabels);
tElapsedLR = toc(tStartLR);
result = modelLR.test(testingData,testingLabels);
errorLR = result.err();

modelSVM = svm('-t 1 -g 3 -c 1 -e 3 -d 1');
tStartSVM = tic;
modelSVM = modelSVM.train(trainingData,trainingLabels);
tElapsedSVM = toc(tStartSVM);
tested = modelSVM.test(testingData,testingLabels);
errorSVM = tested.err();

figure;plot([tElapsedLR tElapsedSVM]);
figure;plot([errorLR errorSVM]);
end