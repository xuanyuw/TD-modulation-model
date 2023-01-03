clear
clc

f_dir = 'F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model\pCorr_data';
total_rep = 50;

f = waitbar(0, 'Please wait...');
for rep=1:total_rep
    load(fullfile(f_dir, ['pCorr_data_rep' num2str(rep-1) '_m1.mat']))
    [ipsi_pCorr_stim_m1, ipsi_pCorr_choice_m1] = calculate_pCorr(h, ipsi_stim_arr, ipsi_choice_arr);
    [contra_pCorr_stim_m1, contra_pCorr_choice_m1] = calculate_pCorr(h, contra_stim_arr, contra_choice_arr);
    load(fullfile(f_dir, ['pCorr_data_rep' num2str(rep-1) '_m2.mat']))
    [ipsi_pCorr_stim_m2, ipsi_pCorr_choice_m2] = calculate_pCorr(h, ipsi_stim_arr, ipsi_choice_arr);
    [contra_pCorr_stim_m2, contra_pCorr_choice_m2] = calculate_pCorr(h, contra_stim_arr, contra_choice_arr);
    
    ipsi_pCorr_stim = cat(2, ipsi_pCorr_stim_m1, ipsi_pCorr_stim_m2);
    contra_pCorr_stim = cat(2, contra_pCorr_stim_m1, contra_pCorr_stim_m2);
    ipsi_pCorr_choice = cat(2, ipsi_pCorr_choice_m1, ipsi_pCorr_choice_m2);
    contra_pCorr_choice = cat(2, contra_pCorr_choice_m1, contra_pCorr_choice_m2);
    
    save(fullfile(f_dir, ['pCorr_result_rep', num2str(rep-1) '.mat']), ....
        'ipsi_pCorr_stim', 'contra_pCorr_stim', 'ipsi_pCorr_choice', 'contra_pCorr_choice')
    waitbar(rep/total_rep, f, 'Calculating partial correlation');
end
close(f)


%%
function [pCorr_stim, pCorr_choice] = calculate_pCorr(h, stim_arr, choice_arr)

pCorr_stim = zeros(size(h, 1), size(h, 3));
pCorr_choice = zeros(size(h, 1), size(h, 3));
for t=1:size(h, 1)
    for c=1:size(h,3)
        pCorr_stim(t, c) = partialcorr(h(t, :, c)', stim_arr(:, c), choice_arr(:, c));
        pCorr_choice(t, c) = partialcorr(h(t, :, c)', choice_arr(:, c), stim_arr(:, c));
    end
end
end
