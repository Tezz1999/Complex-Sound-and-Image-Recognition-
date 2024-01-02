positiveFolder = 'C://Users/user/Downloads/Proj4/Proj4/breast-histopathology-images/1';
negativeFolder = 'C://Users/user/Downloads/Proj4/Proj4/breast-histopathology-images/0';

[positiveImages, positiveLabels] = loadImagesFromFolder(positiveFolder, 1);
[negativeImages, negativeLabels] = loadImagesFromFolder(negativeFolder, 0);

% Combine positive and negative datasets
X = [positiveImages; negativeImages];
y = [positiveLabels; negativeLabels];

cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
idx = cv.test;

% Training set
X_train = X(~idx, :);
y_train = y(~idx, :);

% Test set
X_test = X(idx, :);
y_test = y(idx, :);

X_train_fft = applyFFT(X_train);
X_test_fft = applyFFT(X_test);

X_train_dwt = applyDWT(X_train, 'haar');
X_test_dwt = applyDWT(X_test, 'haar');

X_train_combined = [X_train_fft, X_train_dwt];
X_test_combined = [X_test_fft, X_test_dwt];

% Apply PCA to Raw Data
[coeff_raw, score_raw, ~, ~, explained_raw] = pca(X_train);

% Retaining 95% variance
cumulativeVariance = cumsum(explained_raw);
numComponentsToRetain = find(cumulativeVariance >= 95, 1, 'first');

X_train_pca_raw = score_raw(:, 1:numComponentsToRetain);
X_test_pca_raw = X_test * coeff_raw(:, 1:numComponentsToRetain);


% Apply PCA to FFT Data
[coeff_fft, score_fft, ~, ~, explained_fft] = pca(X_train_fft);

% Determine the number of components to retain
cumulativeVariance_fft = cumsum(explained_fft);
numComponentsToRetain_fft = find(cumulativeVariance_fft >= 95, 1, 'first');

X_train_pca_fft = score_fft(:, 1:numComponentsToRetain_fft);
X_test_pca_fft = X_test_fft * coeff_fft(:, 1:numComponentsToRetain_fft);


% Apply PCA to DWT Data
[coeff_dwt, score_dwt, ~, ~, explained_dwt] = pca(X_train_dwt);

% Determine the number of components to retain 90% variance
cumulativeVariance_dwt = cumsum(explained_dwt);
numComponentsToRetain_dwt = find(cumulativeVariance_dwt >= 95, 1, 'first');

X_train_pca_dwt = score_dwt(:, 1:numComponentsToRetain_dwt);
X_test_pca_dwt = X_test_dwt * coeff_dwt(:, 1:numComponentsToRetain_dwt);


% Apply PCA to Combined Data
[coeff_combined, score_combined, ~, ~, explained_combined] = pca(X_train_combined);

% Determine the number of components to retain for a specified amount of variance
cumulativeVariance_combined = cumsum(explained_combined);
numComponentsToRetain_combined = find(cumulativeVariance_combined >= 95, 1, 'first');

X_train_pca_combined = score_combined(:, 1:numComponentsToRetain_combined);
X_test_pca_combined = X_test_combined * coeff_combined(:, 1:numComponentsToRetain_combined);




%% 

cumulativeVariance = cumsum(explained_fft);
figure;
plot(cumulativeVariance);
xlabel('Number of Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA Cumulative Explained Variance');
grid on;
%% 
% Number of folds for cross-validation
K = 7;

% Perform K-fold cross-validation
cvLDA = fitcdiscr(X_train_pca_combined, y_train, 'DiscrimType', 'linear', 'SaveMemory', 'on', 'FillCoeffs', 'off');
cvmodel = crossval(cvLDA, 'KFold', K);

% Calculate the fold-wise accuracies
foldAcc = 1 - kfoldLoss(cvmodel, 'LossFun', 'ClassifError', 'Mode', 'individual');

% Plot fold-wise accuracies
figure;
plot(1:K, foldAcc, '-o');
xlabel('Fold Number');
ylabel('Accuracy');
title('Fold-wise Cross-Validation Accuracy for LDA');
grid on;

% Calculate the ROC curve data using kfoldPredict to get the scores
[~, scores] = kfoldPredict(cvmodel);
[~,~,~,AUC] = perfcurve(y_train, scores(:,2), 1); % Assume positive class is labeled as '1'

% Plot the ROC curve
[Xroc, Yroc, ~, AUC] = perfcurve(y_train, scores(:,2), 1);

figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC for LDA (AUC = %.2f)', AUC));


%% 
%Building LDA model with raw data 
[accuracyRaw, errorRaw] = performCVLDA(X_train_pca_raw, y_train, 5);
disp(['CV Accuracy for Raw Data: ', num2str(accuracyRaw)]);
disp(['CV Error for Raw Data: ', num2str(errorRaw)]);

%FFT data
[accuracyFFT, errorFFT] = performCVLDA(X_train_fft, y_train, 5);
disp(['CV Accuracy for FFT Data: ', num2str(accuracyFFT)]);
disp(['CV Error for FFT Data: ', num2str(errorFFT)]);

% DWT Data
[accuracyDWT, errorDWT] = performCVLDA(X_train_pca_dwt, y_train, 5);
disp(['CV Accuracy for DWT Data: ', num2str(accuracyDWT)]);
disp(['CV Error for DWT Data: ', num2str(errorDWT)])

[accuracyCombined, errorCombined] = performCVLDA(X_train_combined, y_train, 5);
disp(['CV Accuracy for Combined Data: ', num2str(accuracyCombined)]);
disp(['CV Error for Combined Data: ', num2str(errorCombined)]);

%% 
[accuracySVMRaw, errorSVMRaw] = performCVSVM(X_train_pca_raw, y_train, 5);
disp(['CV Accuracy for SVM (Raw Data): ', num2str(accuracySVMRaw)]);
disp(['CV Error for SVM (Raw Data): ', num2str(errorSVMRaw)]);

[accuracySVMFFT, errorSVMFFT] = performCVSVM(X_train_pca_fft, y_train, 5);
disp(['CV Accuracy for SVM (FFT Data): ', num2str(accuracySVMFFT)]);
disp(['CV Error for SVM (FFT Data): ', num2str(errorSVMFFT)]);

[accuracySVMDWT, errorSVMDWT] = performCVSVM(X_train_pca_dwt, y_train, 5);
 
disp(['CV Accuracy for SVM (DWT Data): ', num2str(accuracySVMDWT)]);
disp(['CV Error for SVM (DWT Data): ', num2str(errorSVMDWT)]);

[accuracySVMCombined, errorSVMCombined] = performCVSVM(X_train_combined_pca, y_train, 5);
disp(['CV Accuracy for SVM (Combined Data): ', num2str(accuracySVMCombined)]);
disp(['CV Error for SVM (Combined Data): ', num2str(errorSVMCombined)]);
%% 
% Assuming X_train_combined and y_train are your training data and labels
% Prepare a partition for cross-validation
K = 5; % Number of folds
c = cvpartition(y_train, 'KFold', K);

% Preallocate arrays to store fold-wise accuracies and ROC curve data
foldAcc = zeros(K, 1);
tprAll = cell(K, 1);
fprAll = cell(K, 1);
aucAll = zeros(K, 1);

% Loop over folds
for i = 1:K
    % Training/testing indices for this fold
    trainIdx = c.training(i);
    testIdx = c.test(i);
    
    % Train the SVM model
    svmModel = fitcsvm(X_train_combined(trainIdx, :), y_train(trainIdx), ...
        'Standardize', true, 'KernelFunction', 'rbf', 'BoxConstraint', 1);
    
    % Perform predictions and calculate accuracies
    [label, score] = predict(svmModel, X_train_combined(testIdx, :));
    foldAcc(i) = sum(label == y_train(testIdx)) / length(y_train(testIdx));
    
    % Compute ROC curve data for this fold
    [Xroc, Yroc, ~, AUC] = perfcurve(y_train(testIdx), score(:, 2), 1);
    tprAll{i} = Yroc;
    fprAll{i} = Xroc;
    aucAll(i) = AUC;
end

% Plot fold-wise accuracies
figure;
plot(1:K, foldAcc, '-o');
xlabel('Fold Number');
ylabel('Accuracy');
title('Fold-wise Cross-Validation Accuracy for SVM');
grid on;

% Plot ROC curve (average across folds)
figure;
meanFPR = linspace(0, 1, 100);
meanTPR = zeros(size(meanFPR));
for i = 1:K
    meanTPR = meanTPR + interp1(fprAll{i}, tprAll{i}, meanFPR);
end
meanTPR = meanTPR / K;
plot(meanFPR, meanTPR);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Average ROC for SVM (Mean AUC = %.2f)', mean(aucAll)));
%% 
% Determine the size of the input layer
inputSize = size(X_train_pca_combined, 2);
hiddenLayerSize = 10; % Example size, adjust as needed

% Create and train the ANN
net = createDeepANN(inputSize, hiddenLayerSize);

% Convert data to correct format
X_train_mat = X_train_pca_combined'; % Transpose data for MATLAB format
y_train_mat = y_train'; % Transpose labels for MATLAB format

% Train the network
[net, tr] = train(net, X_train_mat, y_train_mat);

% Test the Network
outputs = net(X_test_pca_combined');
errors = gsubtract(y_test', outputs);
performance = perform(net, y_test', outputs);

% Assuming binary classification
predictions = outputs > 0.5;  % Convert outputs to binary predictions
% Ensure y_test is a column vector (needed for comparison)
y_test_col = y_test';

% Calculate the number of correct predictions
numCorrectPredictions = sum(predictions == y_test_col);

% Calculate the total number of predictions
totalPredictions = numel(y_test_col);

% Compute the accuracy
accuracy = numCorrectPredictions / totalPredictions;

% Display the accuracy
disp(['Test Accuracy of ANN: ', num2str(accuracy)]);

%Training performance over epochs
figure;
plot(tr.epoch, tr.perf);
title('ANN Training Performance Over Epochs');
xlabel('Epochs');
ylabel('Performance (MSE)');
grid on;
%% 
function [images, labels] = loadImagesFromFolder(folder, label)
    imgFiles = dir(fullfile(folder, '*.png')); % Adjust the file extension as needed
    numImages = length(imgFiles);
    images = zeros(numImages, 100*100); % Adjusted for 100x100 images
    labels = zeros(numImages, 1);

    for i = 1:numImages
        img = imread(fullfile(folder, imgFiles(i).name));
        if size(img, 3) == 3
            img = rgb2gray(img); % Convert to grayscale if the image is in color
        end
        img = imresize(img, [100, 100]); % Resize image
        images(i, :) = img(:)';
        labels(i) = label;
    end
end

function fftFeatures = applyFFT(images)
    numImages = size(images, 1);
    imageSize = sqrt(size(images, 2)); % Assuming square images
    fftFeatures = zeros(numImages, imageSize * imageSize);

    for i = 1:numImages
        img = reshape(images(i, :), [imageSize, imageSize]);
        fftImg = fftshift(fft2(double(img)));
        fftFeatures(i, :) = abs(fftImg(:))';
    end
end

function dwtFeatures = applyDWT(images, waveletName)
    numImages = size(images, 1);
    imageSize = sqrt(size(images, 2)); % Assuming square images
    [cA,~,~,~] = dwt2(zeros(imageSize, imageSize), waveletName); % Sample transform for size
    dwtSize = length(cA(:));
    dwtFeatures = zeros(numImages, dwtSize);

    for i = 1:numImages
        img = reshape(images(i, :), [imageSize, imageSize]);
        [cA,~,~,~] = dwt2(double(img), waveletName);
        dwtFeatures(i, :) = cA(:)';
    end
end

function [meanAccuracy, meanError] = performCVLDA(X, y, k)
    ldaModel = fitcdiscr(X, y, 'DiscrimType', 'linear');
    cvModel = crossval(ldaModel, 'KFold', k);
    lossValues = kfoldLoss(cvModel, 'LossFun', 'ClassifError');
    meanError = mean(lossValues);
    meanAccuracy = 1 - meanError;
end

function [meanAccuracy, meanError] = performCVSVM(X, y, k)
    svmModel = fitcsvm(X, y, 'KernelFunction', 'rbf', 'Standardize', true);
    cvModel = crossval(svmModel, 'KFold', k);
    lossValues = kfoldLoss(cvModel, 'LossFun', 'ClassifError');
    meanError = mean(lossValues);
    meanAccuracy = 1 - meanError;
end

function net = createDeepANN(inputSize, hiddenLayerSize)
    % Define the number of neurons in each of the 8 hidden layers
    hiddenLayers = repmat(hiddenLayerSize, 1, 8);

    % Create a feedforward neural network with 8 hidden layers
    net = feedforwardnet(hiddenLayers);

    % Configure the inputs and outputs
    net = configure(net, rand(inputSize, 1), rand(1, 1));
    net.trainParam.epochs = 1000;        % Set a high number of epochs
    net.trainParam.max_fail = 20;        % Increase the number of validation checks
    net.trainParam.goal = 1e-6;        

    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
end