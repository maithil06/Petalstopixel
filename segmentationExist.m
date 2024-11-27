function segmentationExist()
    % Set the directory for images and labels
    baseDir = 'D:\Computer Vision\data_for_moodle\data_for_moodle\';  % Adjust this path as necessary
    imageDataDir = fullfile(baseDir, 'images_256');
    labelDataDir = fullfile(baseDir, 'labels_256');

    imds = imageDatastore(imageDataDir);
    pxds = pixelLabelDatastore(labelDataDir, ["background", "flower"], [0 255]);

    % Verify that labels are loaded
    if isempty(pxds.Labels)
        error('Labels property is empty. Ensure that label data is correctly loaded.');
    end

    % Display some info about the label datastore
    disp("Label classes:");
    disp(pxds.ClassNames);
    disp("Number of labeled files:");
    disp(length(pxds.Files));

    % Create an image data augmenter to perform random horizontal flipping
    imageAugmenter = imageDataAugmenter('RandXReflection',true);

    % Set up training and validation data
    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionData(imds, pxds);

    % Load a pre-trained network, modify it for two classes
    net = segnetLayers([256 256 3], 2, 'vgg16');

    % Check for GPU availability
    executionEnvironment = 'auto';
    if gpuDeviceCount > 0
        executionEnvironment = 'gpu';
    end

    % Set training options with GPU usage
    opts = trainingOptions('sgdm', ...
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ...
        'MiniBatchSize',2, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'VerboseFrequency',10, ...
        'ExecutionEnvironment', executionEnvironment);

    % Train the network
    net = trainNetwork(imdsTrain, pxdsTrain, net.Layers, opts);

    % Save the trained network
    save('segmentexistnet.mat', 'net');
end

function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionData(imds, pxds)
    % Partition data into training and testing, with a check for labels
    if isempty(pxds.Labels)
        error('Pixel Label Datastore must have non-empty Labels property to use splitEachLabel.');
    end
    [imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
    [pxdsTrain, pxdsTest] = splitEachLabel(pxds, 0.8, 'randomized');
end
