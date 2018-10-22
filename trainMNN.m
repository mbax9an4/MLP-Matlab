function [weightIH, weightHO] = trainMNN(example,label,weightIH,weightHO, classes, learningRate)

[sizeIL, sizeHL] = size(weightIH);
[~,sizeOL] = size(weightHO);

inputLayer = ones(sizeIL,1);
inputLayer(2:sizeIL,:) = example(:,:)';

hiddenLayer = zeros(sizeHL,1);
outputLayer = zeros(sizeOL,1);
error=0;


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

[~, labelPredicted] = max(outputLayer);

labelPredicted = classes(labelPredicted);

if(labelPredicted ~= label)
    error = error+1;
    outputLError = zeros(sizeOL,1);
    hiddenLError = zeros(sizeHL,1);

    %calculate error for the output layer
    for o=1:sizeOL
        outputLError(o) = outputLayer(o)*(1-outputLayer(o))*(classes(o)-outputLayer(o));
    end
    
    %calculate error for the hidden layer
    for h=1:sizeHL
        weightedOLErrSum = 0.0; 
        for o=1:sizeOL
            weightedOLErrSum = weightedOLErrSum+outputLError(o)*weightHO(h,o);
        end
        hiddenLError(h) = hiddenLayer(h)*(1-hiddenLayer(h))*weightedOLErrSum;
    end
    
    %update weights between input layer and hidden layer
    for i=1:sizeIL
        for h=1:sizeHL
            weightIH(i,h) = weightIH(i,h)+(learningRate*hiddenLError(h)*inputLayer(i));
        end
    end
    
    %update weights between hidden layer and output layer
    for h=1:sizeHL
        for o=1:sizeOL
            weightHO(h,o) = weightHO(h,o)+(learningRate*outputLError(o)*hiddenLayer(h));
        end
    end
    
end
error = error/(sizeIL-1)
end