function [] = polDEval()
load biggerdata;

[rows,~] = size(data);

trainingData = data(1:rows/2,:);
trainingLabels = labels(1:rows/2,:);

testingData = data(rows/2+1:rows,:);
testingLabels = labels(rows/2+1:rows,:);

model = svm('-t 1 -g 3 -c 1 -e 1 -d 1');
tStart1 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed1 = toc(tStart1);
tested = model.test(testingData,testingLabels);
error1 = tested.err()

model = svm('-t 1 -g 3 -c 1 -e 1 -d 2');
tStart2 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed2 = toc(tStart2);
tested = model.test(testingData,testingLabels);
error2 = tested.err()

model = svm('-t 1 -g 3 -c 1 -e 1 -d 3');
tStart3 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed3 = toc(tStart3);
tested = model.test(testingData,testingLabels);
error3 = tested.err()

model = svm('-t 1 -g 3 -c 1 -e 1 -d 4');
tStart4 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed4 = toc(tStart4);
tested = model.test(testingData,testingLabels);
error4 = tested.err()

model = svm('-t 1 -g 3 -c 1 -e 1 -d 5');
tStart5 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed5 = toc(tStart5);
tested = model.test(testingData,testingLabels);
error5 = tested.err()

model = svm('-t 1 -g 3 -c 1 -e 1 -d 6');
tStart6 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed6 = toc(tStart6);
tested = model.test(testingData,testingLabels);
error6 = tested.err()

model = svm('-t 1 -g 3 -c 1 -e 1 -d 7');
tStart7 = tic;
model = model.train(trainingData,trainingLabels);
tElapsed7 = toc(tStart7);
tested = model.test(testingData,testingLabels);
error7 = tested.err()
plotboundary(data,labels,model);


figure; plot([error1 error2 error3 error4 error5 error6 error7]);
figure; plot([tElapsed1, tElapsed2, tElapsed3, tElapsed4, tElapsed5, tElapsed6, tElapsed7]);
end