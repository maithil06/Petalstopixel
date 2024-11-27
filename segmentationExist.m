imageDir = 'data_for_moodle/images_256/';
labelDir = 'data_for_moodle/labels_256/';
classNames = ["flower", "background"];
pixelLabelID = [1, 3];

imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID, ...
    'FileExtensions','.png','ReadFcn',@imread);

[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = prepareData(imageDir, labelDir, classNames, pixelLabelID, 0.20);

inputSize = [256 256 3];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(inputSize, numClasses, "resnet18");

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 15, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', combine(imdsVal, pxdsVal), ...
    'MiniBatchSize', 4, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');  

net = trainNetwork(combine(imdsTrain, pxdsTrain), lgraph, options);

save('segmentexistnet_deeplabadam.mat', 'net');
valResults = semanticseg(imdsVal,net);
metrics = evaluateSemanticSegmentation(valResults,pxdsVal);
metrics.NormalizedConfusionMatrix;
rmdir('semanticsegOutput', 's');

function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = prepareData(imDir, labelDir, classNames, pixelLabelID, validationFraction)
    
    imds = imageDatastore(imDir);
    pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID, ...
        "ReadFcn", @(x) relabel(x, pixelLabelID, classNames));
    
    validImageFiles = removeNonMatchingFiles(imds.Files, pxds.Files);
    imds = imageDatastore(validImageFiles);
    assert(numel(imds.Files) == numel(pxds.Files), 'The number of images and labels must match after filtering.');

    numFiles = numel(imds.Files);
    indices = randperm(numFiles);
    numValFiles = round(validationFraction * numFiles);
    valIndices = indices(1:numValFiles);
    trainIndices = indices(numValFiles+1:end);
    imdsTrain = subset(imds, trainIndices);
    imdsVal = subset(imds, valIndices);
    pxdsTrain = subset(pxds, trainIndices);
    pxdsVal = subset(pxds, valIndices);
end

function labelData = relabel(filePath, labelID, classNames)
    labelData = imread(filePath);
    labelData(labelData == 2 | labelData == 4 | labelData == 0) = 3; 
    labelData = categorical(labelData, labelID, classNames);
end

function validFiles = removeNonMatchingFiles(imageFiles, labelFiles)
    [~, imageNames] = cellfun(@fileparts, imageFiles, 'UniformOutput', false);
    [~, labelNames] = cellfun(@fileparts, labelFiles, 'UniformOutput', false);
    validIdx = ismember(imageNames, labelNames);
    validFiles = imageFiles(validIdx);
end