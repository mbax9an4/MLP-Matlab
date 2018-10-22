function [] = crossValidation()
load heart
folds = 5;

[rows,columns] = size(data);
foldSize = fix(rows / folds);
extraRows = 0;
if foldSize*folds ~= rows
    extraRows = rows - (foldSize*folds);
end
foldsArray = zeros(folds, foldSize, columns);
labelsArray = zeros(folds, foldSize);

testingData = zeros(foldSize+extraRows, columns);
testingLabels = zeros(foldSize+extraRows,1);

firstIndex = 1;
lastIndex = foldSize;
for i=1:folds
    if i ~= folds
        foldsArray(i,1:foldSize,:) = data(firstIndex:lastIndex,:);
        labelsArray(i,1:foldSize) = labels(firstIndex:lastIndex,1);
    else
        testingData(:,:) = data(firstIndex:lastIndex+extraRows,:);
        testingLabels(:) = labels(firstIndex:lastIndex+extraRows,1);
    end
    firstIndex = lastIndex+1;
    lastIndex = lastIndex+foldSize;
end

iterations = 10;
averageError = zeros(iterations,1);
evaluationError = zeros(iterations,1);

for iteration = 1:iterations
model = logreg('iterations', 1,'learningrate', 0.05/iteration);
error = zeros(folds,1);

validationData = zeros(foldSize, columns);
validationLabels = zeros(foldSize,1);

trainingData = zeros((folds-2)*foldSize, columns);
trainingLabels = zeros((folds-2)*foldSize,1);

for j=1:folds-1
    validationData(:,:) = foldsArray(j,1:foldSize,:);
    validationLabels(:,1) = labelsArray(j,1:foldSize);
    
    first = 1;
    last = foldSize;
    
    for k = 1:folds-1
        if k~=j
            trainingData(first:last,:) = foldsArray(k,1:foldSize,:);
            trainingLabels(first:last) = labelsArray(k,1:foldSize);
        end
        first = last+1;
        last = last+foldSize;
    end

    model = model.train(trainingData,trainingLabels);
    result = model.test(validationData,validationLabels);
    error(j) = result.err();
end
evaluation = model.test(testingData,testingLabels);
evaluationError(iteration) = evaluation.err();

averageError(iteration,1) = mean(error);
end

[~,minIndex] = min(averageError);
bestLearningRate = 0.05/minIndex

figure,
plot(averageError, 'ro-');
hold on;
plot(evaluationError,'x-');
clear;
end