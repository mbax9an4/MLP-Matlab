function [] = part2
load heart;

% ------------------- ex 1 part 2 -------------------------- 
modelP = logreg('iterations', 1,'learningrate', 0.01);

modelP = modelP.train(data,labels);
result = modelP.test(data,labels);
performance(1) = result.err();

for i=1:100
modelP = modelP.train(data,labels);

result = modelP.test(data,labels);
performance(i) = result.err();
end

figure,
plot(performance)

% ---------------------- ex 2 part 2 -----------------------
model = logreg('iterations', 1,'learningrate', 0.01);
iterations=20;
maxVal = zeros(iterations,1);
minVal = ones(iterations,1)*3;
error = zeros(iterations,1);

for i=1:iterations
[data, labels] = shufflerows(data,labels);
trainingdata = data(1:170,:);
traininglabels = labels(1:170,:);
testingdata = data(171:270,:);
testinglabels = labels(171:270,:);
for j=1:10
model = model.train(trainingdata,traininglabels);
result = model.test(testingdata,testinglabels);
error(i) = result.err();

if maxVal(i) < error(i)
maxVal(i) = error(i);
end

if minVal(i) > error(i)
minVal(i) = error(i);
end

end
end

figure,
errorbar(1:iterations,error,minVal, maxVal)
clear;
% ------------------- ex 3 part 2 -----------------------
load heart
folds = 5;

[rows,columns] = size(data);
foldSize = fix(rows / folds);
extraRows = 0;
if foldSize*folds ~= rows
    extraRows = rows - (foldSize*folds);
end
foldsArray = zeros(folds, foldSize+extraRows, columns);
labelsArray = zeros(folds, foldSize+extraRows);

firstIndex = 1;
lastIndex = foldSize;
for i=1:folds
    if i ~= folds
        foldsArray(i,1:foldSize,:) = data(firstIndex:lastIndex,:);
        foldsArray(i,foldSize+1:foldSize+extraRows,:) = zeros(extraRows,columns);
        labelsArray(i,1:foldSize) = labels(firstIndex:lastIndex,1);
        labelsArray(i,foldSize+1:foldSize+extraRows) = zeros(extraRows,1);
    else
        foldsArray(i,:,:) = data(firstIndex:lastIndex+extraRows,:);
        labelsArray(i,:) = labels(firstIndex:lastIndex+extraRows,1);
    end
    firstIndex = lastIndex+1;
    lastIndex = lastIndex+foldSize;
end

iterations = 10;
averageError = zeros(iterations,1);

for iteration = 1:iterations
model = logreg('iterations', 1,'learningrate', 0.05/iteration);
error = zeros(folds,1);

for j=1:folds
    if j ~= folds
        testingData = zeros(foldSize, columns);
        testingLabels = zeros(foldSize,1);
        testingData(:,:) = foldsArray(j,1:foldSize,:);
        testingLabels(:,1) = labelsArray(j,1:foldSize);
    else
        testingData = zeros(foldSize + extraRows, columns);
        testingLabels = zeros(foldSize + extraRows,1);
        testingData(:,:) = foldsArray(j,:,:);
        testingLabels(:,1) = labelsArray(j,:);
    end
    first = 1;
    last = foldSize;
    
    [tr,~] = size(testingData);
    if tr == foldSize
       trainingData = zeros((folds-1)*foldSize + extraRows, columns);
       trainingLabels = zeros((folds-1)*foldSize + extraRows,1);
    else
       trainingData = zeros((folds-1)*foldSize, columns);
       trainingLabels = zeros((folds-1)*foldSize,1);
    end
    for k = 1:folds
        if k~=j
            if k ~= folds
                trainingData(first:last,:) = foldsArray(k,1:foldSize,:);
                trainingLabels(first:last) = labelsArray(k,1:foldSize);
            else
                trainingData(first:last+extraRows,:) = foldsArray(k,:,:);
                trainingLabels(first:last+extraRows) = labelsArray(k,:);
            end
        end
        first = last+1;
        last = last+foldSize;
    end

    model = model.train(trainingData,trainingLabels);
    result = model.test(testingData,testingLabels);
    error(j) = result.err();
end
averageError(iteration,1) = mean(error);
end

[~,minIndex] = min(averageError);
bestLearningRate = 0.05/minIndex
clear;
% -------------------------- ex 4 part 2 ------------------------
load datab;
th2 = naiveStump(data,labels);

load datac;
th3 = naiveStump(data,labels);

load datad;
th4 = naiveStump(data,labels);

load datae;
th5 = naiveStump(data,labels);

load dataf;
th6 = naiveStump(data,labels);

load datag;
th7 = naiveStump(data,labels);

figure,
plot(th2, th3, th4, th5, th6, th7)

load heart;
th1 = naiveStump(data, labels);

end

