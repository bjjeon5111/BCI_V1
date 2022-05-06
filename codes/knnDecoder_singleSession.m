function [acc,confusionMat,decoder_config] = knnDecoder_singleSession(DataMatrix,stim_ori,numNeurons,numTrials,chosenK)
% this function computes the accuracies of orientation decoding using KNN
% classification.  The stimulus orientation spacing is assumed to be
% consistent at 15 degrees (12 orientations).  If not true, you must edit
% the orientation spacing in 'default param' section

% Input Variables: 
%   DataMatrix - [Nx1] cell array containing neural responses to stimuli, where each cell is the response data for one neuron.
%       Within each cell of this array, is a nested cell array of either [axb] or [cx1] responses to
%       each stimulus.  if in the form [axb] a is the number of orientations, and b is the number of spatial frequencies(in our case this is the number of
%       orientations. if in the form [cx1] c is the total number of stimuli.  Finally, each stimulus of the nested array contains [dx1] vector, where each entry in the vector is 
%       a single trial response value.  For an example of this matrix, see exampleDataMatrix.mat 
%   stim_ori - [axb] or [cx1] double array.  The stimulus
%       orientation and stimulus spatial frequency that matches the format of
%       the DataMatrix.
%   numNeurons - scalar, limit for the number of neurons to use in decoding. Number of neurons cannot be larger than the total number of neurons in the DataMatrix 
%   numTrials - scalar, number of trials to use for decoding. cannot be smaller than the minimum number of trials available for the presented stimuli
%   chosenK - scalar, number of folds for cross validation, 0 cross validates using leave one out.

% Output 
%   acc - scalar, orientation decoding accuracy
%   confusionMat - cell array containing the confusion matrix.  The confusion matrix is [axa] array where y is the predicted stim and x
%   is the true stimulus. The entry in each position of the matrix is the
%   frequency of occurence in decoding.
%   decoder_config - struct, configuration of the decoder
%% default param
OriSpacing = 15;

%%
% select neurons that will be used in decoding
numNeurons_all = size(DataMatrix,1);

% random sampling of neurons
selectedNeurons = randperm(numNeurons_all,numNeurons);
selectedNeuronID = sort(selectedNeurons,'ascend');
decoder_config.selectedNeuronID = selectedNeuronID;

% extract trials to be used
numtrials_per_stim = numTrials;
numFolds = numtrials_per_stim;
decoder_config.numTrialsUsed = numtrials_per_stim;
decoder_config.OriSpacing = OriSpacing;
%% create the dataset to use for decoding
stimOrientations = stim_ori(:);% stim orientations
numStim = length(stimOrientations);
% create a matrix of size Neurons x Stim x Trial
fullDatasetMat = nan(length(selectedNeuronID),numStim,numtrials_per_stim);
pointsAvailable = cell(numStim,1);
for i = 1:length(selectedNeuronID)
    tempData = DataMatrix{selectedNeuronID(i)}(:);
    for j = 1:numStim
        if i == 1
            pointsAvailable_temp = find(~isnan(tempData{j}));
            % Randomly sample trials 
            selectedPoints = sort(randperm(length(pointsAvailable_temp),numtrials_per_stim));
            pointsAvailable{j} = pointsAvailable_temp(selectedPoints);
        end
        fullDatasetMat(i,j,:) = tempData{j}(pointsAvailable{j});
    end
end
%%
numStim_ori = length(unique(stimOrientations));
decoder_config.selectedTrials = pointsAvailable;
%% Decode using the selected Validation method
numberOfNeighbors = chosenK;% options for K
numTrialsPerFold = numtrials_per_stim/numFolds;
[ConfusionMatFold_dist_ori,ConfusionMatFold_corr_ori] = deal(cell(numFolds,length(numberOfNeighbors)));
% ConfusionMatFold_corr = cell(numFolds,length(numberOfNeighbors));
decoder_config.DataSet = fullDatasetMat;
for i = 1:numFolds
%     disp(['Computing Fold ',num2str(i)])
    %% Shape the data for decoding
    % create training and test data set for the fold
    trainingDataInd = ones(numtrials_per_stim,1);
    testDataInd = zeros(numtrials_per_stim,1);
    a = ((i-1)*numTrialsPerFold+1):(i*numTrialsPerFold);
    trainingDataInd(a) = 0;
    testDataInd(a) = 1;
    trainingSet = fullDatasetMat(:,:,logical(trainingDataInd));
    testSet = fullDatasetMat(:,:,logical(testDataInd));

    % reshape the training and test set
    trainingSet = reshape(trainingSet,numNeurons,[]);
    trainingSetLabel = repmat((1:numStim),1,sum(trainingDataInd));
    trainingSetLabel_ori = repmat(stimOrientations,1,sum(trainingDataInd));
    testSetLabel = repmat((1:numStim),1,sum(testDataInd));
    testSetLabel_ori = repmat(stimOrientations,1,sum(testDataInd));
    testSet = reshape(testSet,numNeurons,[]);
    numPts2Test = size(testSet,2);
    
    %% now the actual classification part
    for j = 1:numPts2Test
        % compute the difference as normalized distance
        normDist = zeros(length(trainingSetLabel),1);
%         corrDist = zeros(length(trainingSetLabel),1);
        for b = 1:length(trainingSetLabel)
            % get the euclidean distance
            normDist(b) = norm((testSet(:,j)-trainingSet(:,b)));% difference in
            % also computed the correlation distance
            % first normalize the vectors
        end
        
        % sort by minimum distances
        [~,sortOrder_dist] = sort(normDist);
        
        %% loop for different K
        for k = 1:length(numberOfNeighbors)
            if j==1
                ConfusionMatFold_dist_ori{i,k} = zeros(numStim_ori,numStim_ori);
                ConfusionMatFold_corr_ori{i,k} = zeros(numStim_ori,numStim_ori);
            end
            % get the k nearest neighbors
            nn_labels_dist_ori = trainingSetLabel_ori(sortOrder_dist(1:numberOfNeighbors(k)));

            % orientation decoding
            nn_labels_dist_unique_ori = unique(nn_labels_dist_ori);
            if length(nn_labels_dist_unique_ori) == length(nn_labels_dist_ori)
                predictedStim_dist_ori = nn_labels_dist_ori(randi(length(nn_labels_dist_ori)));
            else
                %otherwise, select the mode
                predictedStim_dist_ori = mode(nn_labels_dist_ori);
            end
            predictedStim_dist_ori = round(predictedStim_dist_ori/OriSpacing)+1;
            trueStim_ori = floor(testSetLabel_ori(j)/OriSpacing)+1;
            ConfusionMatFold_dist_ori{i,k}(predictedStim_dist_ori,trueStim_ori) = ConfusionMatFold_dist_ori{i,k}(predictedStim_dist_ori,trueStim_ori)+1;       
        end
        
        
    end
end

%%
for i = 1:size(ConfusionMatFold_dist_ori,1)
    for j = 1:size(ConfusionMatFold_dist_ori,2)
        acc_dist_ori(i,j) = sum(diag(ConfusionMatFold_dist_ori{i,j}))/sum(ConfusionMatFold_dist_ori{i,j}(:));
    end
end
% save(saveName);
confusionMatTotal_ori = cell(1,size(ConfusionMatFold_dist_ori,2));
for j = 1:size(ConfusionMatFold_dist_ori,2)
    for i = 1:size(ConfusionMatFold_dist_ori,1)
        if i == 1
            confusionMatTotal_ori{j} = zeros(numStim_ori,numStim_ori);
        end
        confusionMatTotal_ori{j} = confusionMatTotal_ori{j} + ConfusionMatFold_dist_ori{i,j};
    end
end
for j = 1:length(confusionMatTotal_ori)
    acc_total(j) = sum(diag(confusionMatTotal_ori{j}))/sum(confusionMatTotal_ori{j}(:));
end
%%
confusionMat = confusionMatTotal_ori;
acc = mean(acc_dist_ori);