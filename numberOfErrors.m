function errors = numberOfErrors(featureIndex,threshold, t, data, labels)

[rows,columns] = size(data);
errors = 0;
for x = 1:rows
    predLabel = 0;
    for feature = 1:columns
        if feature == featureIndex
            th = t;
        else
            th = threshold(feature);
        end
        if data(x,feature)-th>0
            predLabel = predLabel + 1;
        else
            predLabel = predLabel + 0;
        end
    end
    if predLabel == columns
        predLabel = 1;
    else
        predLabel = 0;
    end
    if predLabel ~= labels(x)
        errors = errors+1;
    end
end
end