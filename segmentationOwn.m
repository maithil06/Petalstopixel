imDir = 'data_for_moodle/images_256';
labelDir = 'data_for_moodle/labels_256';
classNames = ["flower", "background"];
pixelLabelID = [1, 3];

[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = prepareData(imDir, labelDir, classNames, pixelLabelID, 0.20);

inputSize = [256 256 3]; 
numClasses = 2; % Flower and background
layers = createSimpleSegmentationCNN(inputSize, numClasses);

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 15, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', combine(imdsVal, pxdsVal), ...
    'MiniBatchSize', 4, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress'); 

dsTrain = combine(imdsTrain, pxdsTrain);
[net, info] = trainNetwork(dsTrain, layers, options); 

save('segmentexistnetadamown.mat', 'net');

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
    labelData(labelData == 2 | labelData == 4 | labelData == 5 | labelData == 0) = 3; % Map to background
    labelData = categorical(labelData, labelID, classNames);
end

function validFiles = removeNonMatchingFiles(imageFiles, labelFiles)
    [~, imageNames] = cellfun(@fileparts, imageFiles, 'UniformOutput', false);
    [~, labelNames] = cellfun(@fileparts, labelFiles, 'UniformOutput', false);
    
    validIdx = ismember(imageNames, labelNames);
    validFiles = imageFiles(validIdx);
end

function layers = createSimpleSegmentationCNN(inputSize, numClasses)
    layers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
        
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
        
        transposedConv2dLayer(4, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'transConv1')
        reluLayer('Name', 'relu4')
        
        transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'transConv2')
        reluLayer('Name', 'relu5')
        
        transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'transConv3')
        reluLayer('Name', 'relu6')
        
        convolution2dLayer(1, numClasses, 'Padding', 'same', 'Name', 'conv4')
        softmaxLayer('Name', 'softmax')
        pixelClassificationLayer('Name', 'pixelLabels')
    ];
end