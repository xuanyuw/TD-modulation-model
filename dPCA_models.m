%% 
clc
clear

addpath('F:\Github\dPCA\matlab')

% modelTypes = {'full', 'noFeedback', 'cutSpec', 'cutNonspec'};
modelTypes = {'cutNonspec'};

dataDir = 'F:\Github\TD-modulation-model\dPCA_allTrial_data';
figDir = 'F:\Github\TD-modulation-model\dPCA_allTrial_plots';
processedDataDir = fullfile(figDir, 'processedData');


totalRep = 50;
useCorrect = false;
rerunReg = false;
sepStim = false;

if sepStim
    processedDataDir = fullfile(processedDataDir, 'sepStim');
else
    processedDataDir = fullfile(processedDataDir, 'sepSac');
end

if sepStim
    figDir = fullfile(figDir, 'sepStim');
else
    figDir = fullfile(figDir, 'sepSac');
end

if ~isfolder(figDir)
    mkdir(figDir)
end
if ~isfolder(processedDataDir)
    mkdir(processedDataDir)
end

%%

for i = 1:length(modelTypes)
    mType = modelTypes{i};
    for rep = 1:totalRep
%     for rep=10:totalRep
        disp('-----------------------------------------')
        disp(['Running model ' mType ' rep' num2str(rep)])
        if sepStim
            data = load(fullfile(dataDir,mType, 'sepStim', ['rep' num2str(rep-1) '.mat']), 'data');
        else
            data = load(fullfile(dataDir,mType, 'sepSac', ['rep' num2str(rep-1) '.mat']), 'data');
        end
        data = data.data;
        run_dPCA(data, mType, rep, processedDataDir, figDir, useCorrect, rerunReg, sepStim)
    end
end

%%

function run_dPCA(data, mType, rep, processedDataDir, figDir, useCorrect, rerunReg, sepStim)
%Define parameter grouping
combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}};
if sepStim
    margNames = {'StimulusDirection', 'TargetArrangement', 'Condition-independent', 'Interaction'};
else
    margNames = {'SaccadeDirection', 'TargetArrangement', 'Condition-independent', 'Interaction'};
end

% Time events of interest (e.g. stimulus onset/offset, cues etc.)
% They are marked on the plots with vertical lines
timeEvents = [25, 45];

% trialNum = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4));
trialNum = zeros(size(data, 1), size(data, 2), size(data, 3));
for n = 1:size(data, 1)
    for s = 1:size(data, 2)
        for d = 1:size(data, 3)
            trialNum(n, s, d) = sum(~isnan(data(n, s, d, 1, :)));
%             for a = 1:size(data, 4)
%                 trialNum(n, s, d, a) = sum(~isnan(data(n, s, d, a, 1, :)));
%             end
        end
    end
end

% check consistency between trialNum and avgX
for n = 1:size(data, 1)
    for s = 1:size(data, 2)
        for d = 1:size(data, 3)
            assert(isempty(find(isnan(data(n, s, d, :, 1:trialNum(n, s, d))), 1)), 'Something is wrong!')
%             for a=1:size(data, 4)
%                 assert(isempty(find(isnan(data(n, s, d, a, :, 1:trialNum(n, s, d, a))), 1)), 'Something is wrong!')
%             end
        end
    end
end


avgX = nanmean(data, length(size(data)));
disp('running dPCA...')

if useCorrect
    fn = fullfile(processedDataDir, mType, ['optimalLambdas_CombStim_rep', num2str(rep) '_correctOnly.mat']);
else
    fn = fullfile(processedDataDir, mType, ['optimalLambdas_CombStim_rep', num2str(rep) '.mat']);
end

if ~isfolder(fullfile(processedDataDir, mType))
    mkdir(fullfile(processedDataDir, mType))
end
if ~isfolder(fullfile(figDir, mType))
    mkdir(fullfile(figDir, mType))
end

if isfile(fn) && ~rerunReg
    load(fn)
else
    optimalLambda = dpca_optimizeLambda(avgX, data, trialNum, ...
        'combinedParams', combinedParams, ...
        'simultaneous', false, ...
        'numRep', 10, ... % increase this number to ~10 for better accuracy
        'filename', fn);
end

% Cnoise treats noise covariance matrices from different parameter
% combinations as equally important --> deal with unbalanced data
Cnoise = dpca_getNoiseCovariance(avgX, ...
data, trialNum, 'simultaneous', false);

[W, V, whichMarg] = dpca(avgX, 20, ...
    'combinedParams', combinedParams, ...
    'lambda', optimalLambda, ...
    'Cnoise', Cnoise);


explVar = dpca_explainedVariance(avgX, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(avgX, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'whichMarg', whichMarg, ...
    'timeEvents', timeEvents);

if useCorrect
    save(fullfile(figDir, mType, ['rep', num2str(rep),'_dPCA_expVar_correctOnly.mat']), 'explVar')
    savefig(fullfile(figDir, mType, ['rep', num2str(rep),'_dPCA_correctOnly.fig']));
else
    save(fullfile(figDir, mType, ['rep', num2str(rep),'_dPCA_expVar_allTrials.mat']), 'explVar')
    savefig(fullfile(figDir, mType, ['rep', num2str(rep),'_dPCA_allTrials.fig']));
end
close all
end

