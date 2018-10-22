function [] = leaveOneOutCV()
load dataa

[folds,columns] = size(data);

foldsArray = zeros(folds, 1, columns);
labelsArray = zeros(folds, 1);

testingData = zeros(1, columns);
testingLabels = zeros(1,1);

for i=1:folds
    if i ~= folds
        foldsArray(i,:,:) = data(i,:);
        labelsArray(i,:) = labels(i,1);
    else
        testingData(:,:) = data(i,:);
        testingLabels(:) = labels(i,1);
    end
end

iterations = 20;
averageError = zeros(iterations,1);
evaluationError = zeros(iterations,1);

for iteration = 1:iterations
model = logreg('iterations', 1,'learningrate', 0.05/iteration);
error = zeros(folds,1);

validationData = zeros(1, columns);
validationLabels = zeros(1,1);

trainingData = zeros(folds-2, columns);
trainingLabels = zeros(folds-2,1);

for j=1:folds-1
    validationData(:,:) = foldsArray(j,:,:);
    validationLabels(:,1) = labelsArray(j,:);
    
    for k = 1:folds-1
        if k~=j
            trainingData(k,:) = foldsArray(k,:,:);
            trainingLabels(k) = labelsArray(k,:);
        end
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