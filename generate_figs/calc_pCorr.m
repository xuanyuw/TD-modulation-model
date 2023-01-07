clear
clc

f_dir = 'F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model\pCorr_data';
total_rep = 50;
calc_single = true;

if calc_single
    n = 'single_pCorr_data_rep';
    cat_d = 1;
else
    n = 'pCorr_data_rep';
    cat_d = 2;
end

f = waitbar(0, 'Please wait...');
for rep=1:total_rep
    load(fullfile(f_dir, [n num2str(rep-1) '_m1.mat']))
    [ipsi_pCorr_stim_m1, ipsi_pCorr_choice_m1] = calculate_pCorr(h, ipsi_stim_arr, ipsi_choice_arr, calc_single);
    [contra_pCorr_stim_m1, contra_pCorr_choice_m1] = calculate_pCorr(h, contra_stim_arr, contra_choice_arr, calc_single);
    load(fullfile(f_dir, [n num2str(rep-1) '_m2.mat']))
    [ipsi_pCorr_stim_m2, ipsi_pCorr_choice_m2] = calculate_pCorr(h, ipsi_stim_arr, ipsi_choice_arr, calc_single);
    [contra_pCorr_stim_m2, contra_pCorr_choice_m2] = calculate_pCorr(h, contra_stim_arr, contra_choice_arr, calc_single);
    
    ipsi_pCorr_stim = cat(cat_d, ipsi_pCorr_stim_m1, ipsi_pCorr_stim_m2);
    contra_pCorr_stim = cat(cat_d, contra_pCorr_stim_m1, contra_pCorr_stim_m2);
    ipsi_pCorr_choice = cat(cat_d, ipsi_pCorr_choice_m1, ipsi_pCorr_choice_m2);
    contra_pCorr_choice = cat(cat_d, contra_pCorr_choice_m1, contra_pCorr_choice_m2);
    if calc_single
         save(fullfile(f_dir, ['single_pCorr_result_rep', num2str(rep-1) '.mat']), ....
            'ipsi_pCorr_stim', 'contra_pCorr_stim', 'ipsi_pCorr_choice', 'contra_pCorr_choice')
    else
        save(fullfile(f_dir, ['pCorr_result_rep', num2str(rep-1) '.mat']), ....
            'ipsi_pCorr_stim', 'contra_pCorr_stim', 'ipsi_pCorr_choice', 'contra_pCorr_choice')
    end
    waitbar(rep/total_rep, f, 'Calculating partial correlation');
end
close(f)


%%
function [pCorr_stim, pCorr_choice] = calculate_pCorr(h, stim_arr, choice_arr, calc_single)

if calc_single
    pCorr_stim = zeros(size(h, 2), 1);
    pCorr_choice = zeros(size(h, 2), 1);
    for c=1:size(h,2)
        pCorr_stim(c, 1) = partialcorr(h(:, c), stim_arr(:, c), choice_arr(:, c));
        pCorr_choice(c, 1) = partialcorr(h(:, c), choice_arr(:, c), stim_arr(:, c));
    end
else
    pCorr_stim = zeros(size(h, 1), size(h, 3));
    pCorr_choice = zeros(size(h, 1), size(h, 3));
    for t=1:size(h, 1)
        for c=1:size(h,3)
            pCorr_stim(t, c) = partialcorr(h(t, :, c)', stim_arr(:, c), choice_arr(:, c));
            pCorr_choice(t, c) = partialcorr(h(t, :, c)', choice_arr(:, c), stim_arr(:, c));
        end
    end
end
end
