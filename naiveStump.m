function [threshold] = naiveStump(data, labels)

stepsize = 0.5;
minErrors = intmax;

maxFeatures = double(max(data, [], 1)');
noFeatures = size(maxFeatures);

minFeatures = double(min(data, [], 1)');

threshold = zeros(noFeatures(1,1),1);
for fIndex = 1: noFeatures
    for t = minFeatures(fIndex):stepsize:maxFeatures(fIndex)
        errorsMade = numberOfErrors(fIndex,threshold,t, data, labels);
        if errorsMade <= minErrors
            minErrors = errorsMade;
            threshold(fIndex) = t;
        end
    end
end
threshold

end