imageDir = 'data_for_moodle/data_for_moodle/images_256/';
labelDir = 'data_for_moodle/data_for_moodle/labels_256/';
classNames = ["flower", "background"];
pixelLabelID = [1, 3];

% Load the datasets
imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID, ...
    'FileExtensions','.png','ReadFcn',@imread);

% Split into train and test sets
[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = prepareData(imageDir, labelDir, classNames, pixelLabelID, 0.20);

% Define network architecture using DeepLabv3+
inputSize = [256 256 3];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(inputSize, numClasses, "resnet18");

% Training options
options = trainingOptions('rmsprop', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', combine(imdsVal, pxdsVal), ...
    'MiniBatchSize', 4, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');  % Use GPU if available

% Train the network
net = trainNetwork(combine(imdsTrain, pxdsTrain), lgraph, options);

save('segmentexistnet_deeplabrmsprop.mat', 'net');

pxdsResults = semanticseg(imdsVal, net, 'WriteLocation', tempdir, 'Verbose', false);
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal, 'Verbose', false);
fprintf('Fold %d, IoU: %f, mAP: %f\n', fold, mean(metrics.ClassMetrics.IoU), mean(metrics.ClassMetrics.Accuracy)); % Adjust metric calculation as needed
% Assuming using the final trained model or an ensemble of models
pxdsResultsTest = semanticseg(imdsTest, net, 'WriteLocation', tempdir, 'Verbose', false);
metricsTest = evaluateSemanticSegmentation(pxdsResultsTest, pxdsTest, 'Verbose', false);

fprintf('Test Set Evaluation, IoU: %f, mAP: %f\n', mean(metricsTest.ClassMetrics.IoU))
fprintf(metricsTest.ClassMetrics)


%% Helper Functions
function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = prepareData(imDir, labelDir, classNames, pixelLabelID, validationFraction)
    % Create an imageDatastore and a pixelLabelDatastore
    imds = imageDatastore(imDir);
    pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID, ...
        "ReadFcn", @(x) relabel(x, pixelLabelID, classNames));
    
    % Filter and ensure matching files between images and labels
    validImageFiles = removeNonMatchingFiles(imds.Files, pxds.Files);
    imds = imageDatastore(validImageFiles);
    assert(numel(imds.Files) == numel(pxds.Files), 'The number of images and labels must match after filtering.');

    % Split the data into training and validation
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
    labelData(labelData == 2 | labelData == 4 | labelData == 0) = 3; % Map non-flower to background
    labelData = categorical(labelData, labelID, classNames);
end

function validFiles = removeNonMatchingFiles(imageFiles, labelFiles)
    [~, imageNames] = cellfun(@fileparts, imageFiles, 'UniformOutput', false);
    [~, labelNames] = cellfun(@fileparts, labelFiles, 'UniformOutput', false);
    validIdx = ismember(imageNames, labelNames);
    validFiles = imageFiles(validIdx);
end
