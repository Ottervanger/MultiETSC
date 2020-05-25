%%%% User Inputs %%%%%%%%
dataset = 'MedicalImages';
disp(['Loading ' dataset ' data']);

constraint_type = 'boxco';  %'boxco', 'Naive', or 'Cheby'
pred_type = 'Corr'; %Use 'Corr' for correlated multidimensional Gaussian prediction and use 'Cond' for uncorrelated class-conditional GMM prediction

use_LDG = 1;  %0 for no LDG dimensionality reduction, 1 for LDG dimensionality reduction

%%%%%%%%%%%%%%%%%%%%%%%%%%

restoredefaultpath;
addpath('Reliable_Early_Classification')
addpath(genpath('Utilities'))

tau_percent = [1e-30 1e-10 1e-5 .001 .1 .25 .9];

results.tau_percent = tau_percent;
results.meets_final = zeros(length(tau_percent),1);
results.acc_early = results.meets_final;
results.acc_early_sct = results.meets_final;
results.final_dims = results.meets_final;
results.training_time = results.meets_final;
results.testing_time = results.meets_final;
results.final_acc = results.meets_final;
results.avg_early = results.meets_final;
results.LDG_necessary = results.meets_final;
results.all_early_times = [];
results.all_early_l = [];
results.all_early_l_sct = [];
results.all_final_l = [];
min_d = [];

for counter = 1:length(tau_percent)
    
    disp(['Reliability value ' num2str(counter) ' of ' num2str(length(tau_percent))]);

    [early_l, early_l_sct, early_t, final_l, all_l, ts_l, training_time, testing_time, ChebyThresh, final_dims, LDG_necessary] = multi_class_incomplete_classification( dataset, tau_percent(counter), constraint_type, pred_type, use_LDG, min_d);

    results.meets_final(counter) = sum(final_l == early_l)/length(final_l)*100;
    results.acc_early(counter)  = sum(ts_l == early_l)/length(ts_l)*100;
    results.acc_early_sct(counter)  = sum(ts_l == early_l_sct)/length(ts_l)*100;
    results.final_acc(counter)  = sum(final_l == ts_l)/length(ts_l)*100;
    results.avg_early(counter)  = mean(early_t);
    results.final_dims(counter) = final_dims;
    results.training_time(counter) = training_time;
    results.testing_time(counter) = testing_time;
    results.LDG_necessary(counter) = LDG_necessary;
    results.all_early_times(:,counter) = early_t;
    results.all_early_l(:,counter) = early_l;
    results.all_final_l = final_l;
    results.all_early_l_sct(:,counter) = early_l_sct;
    min_d = early_t;

end;

save(['Results/' dataset '_Reliable_Early_Results_' constraint_type '_' pred_type], 'results');

