function [errors] = testMNN(testingData, testingLabels, weightIH, weightHO, classes)
[sizeIL, sizeHL] = size(weightIH);
[~,sizeOL] = size(weightHO);

errors = 0;

[examples,~] = size(testingData);

for e=1:examples
inputLayer = ones(sizeIL,1);
inputLayer(2:sizeIL,:) = testingData(e,:)';

hiddenLayer = zeros(sizeHL,1);

outputLayer = zeros(sizeOL,1);

for h=1:sizeHL
    weightedSum = 0.0;
    for i=1:sizeIL
        weightedSum = weightedSum+inputLayer(i)*weightIH(i,h);
    end
    hiddenLayer(h) = double(1/(1+exp(-1*weightedSum)));
end

for o=1:sizeOL
    weightedSum = 0.0;
    for h=1:sizeHL
        weightedSum = weightedSum+hiddenLayer(h)*weightHO(h,o);
    end
    outputLayer(o) = double(1/(1+exp(-1*weightedSum)));
end

[ded, labelPredicted] = max(outputLayer);
labelPredicted = classes(labelPredicted);

if labelPredicted ~= testingLabels(e)
    errors = errors+1;
end
end
errors = errors/examples;
end