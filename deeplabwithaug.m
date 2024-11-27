imageDir = 'data_for_moodle/data_for_moodle/images_256/';
labelDir = 'data_for_moodle/data_for_moodle/labels_256/';
classNames = ["flower", "background"];
pixelLabelID = [1, 3];

% Load the datasets
imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID, ...
    'FileExtensions','.png','ReadFcn',@imread);

% Define data augmentation transformations
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-15, 15], ...
    'RandScale', [0.5, 1.5]);

% Split into train and test sets
[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = prepareData(imageDir, labelDir, classNames, pixelLabelID, 0.20);

% Apply augmentation manually to training images and labels
augImdsTrain = transform(imdsTrain, @(x) augmentImage(x, augmenter));
augPxdsTrain = transform(pxdsTrain, @(x) augmentLabel(x, augmenter));

% Define network architecture using DeepLabv3+ with ResNet-18
inputSize = [256 256 3];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(inputSize, numClasses, "resnet18");

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', combine(imdsVal, pxdsVal), ...
    'MiniBatchSize', 4, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');  % Use GPU if available

% Combine augmented image and label datastores for training
dsTrain = combine(augImdsTrain, augPxdsTrain);

% Train the network
net = trainNetwork(dsTrain, lgraph, options);

save('segmentexistnet_deeplabadam.mat', 'net');

%% Custom augmentation functions
function out = augmentImage(image, augmenter)
    out = augment(augmenter, image);
end

function out = augmentLabel(label, augmenter)
    % Ensure the label augmentation is compatible with image augmentation
    out = augment(augmenter, label);
end


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
