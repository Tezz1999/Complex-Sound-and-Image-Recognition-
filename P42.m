% Parameters
n = 12 ; % Number of files to process
time = 10; % Duration to consider for each audio file
nfft = 1024; % Size of FFT
k = 5; % Number of folds for cross-validation

% Paths
path_country = 'C:\Users\user\Downloads\Proj4\Proj4\Country';
path_didgeridoo = 'C:\Users\user\Downloads\Proj4\Proj4\Didgeridoo_Sounds';

% Import audio files
csounds = dir(fullfile(path_country, '*.wav'));
dsounds = dir(fullfile(path_didgeridoo, '*.wav'));

% Initialize matrices
country = [];
didgeridoo = [];
samplingRate = [];

% Process Country Sounds
for i = 1:min(n, length(csounds))
    [audioData, fs] = audioread(fullfile(csounds(i).folder, csounds(i).name));
    if size(audioData, 2) > 1
        audioData = mean(audioData, 2); 
    end
    country(:, i) = audioData(1:min(time*fs, length(audioData))); % Trim or pad to 'time' seconds
    samplingRate = fs; % Assuming all files have the same fs
end

% Process Didgeridoo Sounds
for i = 1:min(n, length(dsounds))
    [audioData, fs] = audioread(fullfile(dsounds(i).folder, dsounds(i).name));
    if size(audioData, 2) > 1
        audioData = mean(audioData, 2); % Convert to mono if stereo
    end
    didgeridoo(:, i) = audioData(1:min(time*fs, length(audioData))); % Trim or pad to 'time' seconds
end

% Feature Extraction using FFT
features = [];
labels = [];

for i = 1:size(country, 2)
    fftFeatures = abs(fft(country(:,i), nfft));
    fftFeatures = fftFeatures(1:nfft/2); % Take only the first half
    features = [features; fftFeatures'];
    labels = [labels; 1]; % Label 1 for country
end

for i = 1:size(didgeridoo, 2)
    fftFeatures = abs(fft(didgeridoo(:,i), nfft));
    fftFeatures = fftFeatures(1:nfft/2); % Take only the first half
    features = [features; fftFeatures'];
    labels = [labels; 2]; % Label 2 for didgeridoo
end

% Dimensionality Reduction using PCA
[coeff, score, ~, ~, explained] = pca(features);
varianceThreshold = 95;
cumulativeVariance = cumsum(explained);
numComponents = find(cumulativeVariance >= varianceThreshold, 1, 'first');
reducedFeatures = score(:, 1:numComponents);

% k-Fold Cross-Validation
cv = cvpartition(labels, 'KFold', k);
accuracy = zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    XTrain = reducedFeatures(trainIdx, :);
    YTrain = labels(trainIdx);
    XTest = reducedFeatures(testIdx, :);
    YTest = labels(testIdx);
    ldaModel = fitcdiscr(XTrain, YTrain);
    YPred = predict(ldaModel, XTest);

    % Evaluate the model
    confMat = confusionmat(YTest, YPred);
    accuracy(i) = sum(diag(confMat)) / sum(confMat(:));
    % Confusion Matrix
    confMat = confusionmat(YTest, YPred);
    figure;
    confusionchart(confMat);
    title(['Confusion Matrix for Fold ', num2str(i)]);
end

% Average Accuracy
avgAccuracy = mean(accuracy) * 100; % Multiply by 100 for percentage
fprintf('Average Classification Accuracy: %.2f%%\n', avgAccuracy);

% Plot first country waveform
figure;
t = linspace(0, length(country(:,1))/samplingRate, length(country(:,1)));
plot(t, country(:,1));
xlabel('Time (Seconds)');
ylabel('Amplitude');
title('Waveform - Country');

% Plot first didgeridoo waveform
figure;
t = linspace(0, length(didgeridoo(:,1))/samplingRate, length(didgeridoo(:,1)));
plot(t, didgeridoo(:,1));
xlabel('Time (Seconds)');
ylabel('Amplitude');
title('Waveform - Didgeridoo');


% Plot Cross-Validation Accuracy per Fold
figure;
plot(accuracy * 100, '-o');
xlabel('Fold Number');
ylabel('Accuracy (%)');
title('Cross-Validation Accuracy per Fold');
ylim([0 100]); % Set Y-axis from 0 to 100%


% Feature Importance (using the coefficients of LDA)
if exist('ldaModel', 'var')
    figure;
    [~, idx] = sort(abs(ldaModel.Coeffs(1,2).Linear), 'descend');
    bar(ldaModel.Coeffs(1,2).Linear(idx));
    title('Feature Importance in LDA Model');
    xlabel('Feature Index');
    ylabel('Coefficient Value');
end
