mypath=fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','training set');
imds=imageDatastore(mypath,'IncludeSubFolders',true,'LabelSource','foldernames');
figure;
perm = randperm(300,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
labelCount = countEachLabel(imds)
img = readimage(imds,1);
size(img)
numTrainFiles = 160;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    imageInputLayer([480 640 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(4,'Stride',4)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'MiniBatchSize',20, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
