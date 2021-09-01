%% STDP Network Different Track Analysis Pipeline
%-------------------------------------------------------------------------%
%   This script runs all the computational spiking neural network
%   experiments used in "A neuronal code for space in hippocampal 
%   coactivity dynamics independent of place fields". In particular, it 
%   examines the effect(s) of different types of plasticity on the coding 
%   of different environments. 
%
%   Written by WTR 10/07/2019 // Last updated by WTR 09/01/2021 
%-------------------------------------------------------------------------%
%% Globals 
eta = 0.0005;                   % learning rate
n_laps = 100;                    % number of laps each experiment is run for
n_pos_A = 10;                   % number of positions for track A
n_pos_B = 10;                   % number of positions for track B
prob_pf_center = 0.20;          % probability of an input neuron having an place field (pf) center at any given position on the track

%% Running the analysis
% Here we create two tracks (A & B). We train a naive network on track A
% and record the performance of the learned network. We then this "trained"
% network both to itself again and to a new track (track B). We look at how
% differing types of plasticity affects the coding of the new track and old
% original track. 
n_init = 10;                    % number of random initializations

% Vector of different plasticity conditions. 1 means plasticity is present
% and 0 means it is absent
EE_plast = [1, 1, 0, 0, 1];    
EI_plast = [1, 0, 1, 0, 1];     
IE_plast = [0, 1, 1, 0, 1];    

% Experiment names for saving
exp_names = cell(1, length(EE_plast)); 
exp_names{1} = 'No IE'; exp_names{2} = 'No EI'; exp_names{3} = 'No EE'; 
exp_names{4} = 'No plasticity'; exp_names{5} = 'All plasticity';

% Path to save experiments data to 
save_path = 'C:\Users\redman\Documents\EI Network\Results\Basic network properties\';
save_flag = 0;                  % set to 1 to save all metrics

% Plotting
plotting_flag = 0;              % set to 1 to plot metrics

% Matrices to save the correlation between activity in each track
pearson_corr_AA = zeros(length(EE_plast), n_init);
pearson_corr_AB = zeros(length(EE_plast), n_init); 

% Matrices to save the similarity metric for the initial "learning"
% experiment
ab_sim = zeros(length(EE_plast), n_init, n_laps/10); 
nn_sim = zeros(length(EE_plast), n_init, n_laps/10); 
half_sim = zeros(length(EE_plast), n_init, n_laps/10); 

% Matrix to save the decoding error for the initial "learning" experiment
d_error = zeros(length(EE_plast), n_init, n_laps/10); 

% Matrices to save the correlation between the rate maps in each
% environment
AB_corr = zeros(length(EE_plast), n_init);
AAp_corr = zeros(length(EE_plast), n_init); 

% Experiments
for pp = 1:length(EE_plast)
    pp
    for ii = 1:n_init
        ii
        close all
        cd(strcat(save_path, exp_names{pp})); 
        
        % Training naive network on track A - the output is referred to as
        % the "trained network".
        [ W_EE, W_EI, W_IE, W_input, pf_cell_A, ~, ab_sim(pp,ii, :), nn_sim(pp, ii, :), half_sim(pp, ii, :), d_error(pp, ii, :)] = ...
            EI_STDP_NETWORK_OG(EE_plast(pp), EI_plast(pp), IE_plast(pp), eta, n_laps, n_pos_A, prob_pf_center, save_flag, plotting_flag);
        if save_flag == 1
            save('absolute_similarity.mat', 'ab_sim'); 
            save('nearest_neighbor_similarity.mat', 'nn_sim'); 
            save('half_similarity.mat', 'half_sim'); 
        end
        
        % Running the trained network on two instances of track A 
        %(potentially with plasticity) to compare against the exposure to
        %track B.
        [activity_A, W_EE_A, W_EI_A, W_IE_A, ratemap_A ] = ...
            EI_STDP_NETWORK(EE_plast(pp), EI_plast(pp), IE_plast(pp), eta, ...
            n_laps, W_EE, W_EI, W_IE, W_input, pf_cell_A, n_pos_A, save_flag, plotting_flag);
        
        [activity_Ap, W_EE_Ap, W_EI_Ap, W_IE_Ap, ratemap_Ap ] = ...
            EI_STDP_NETWORK(EE_plast(pp), EI_plast(pp), IE_plast(pp), eta, ...
            n_laps, W_EE, W_EI, W_IE, W_input, pf_cell_A, n_pos_A, save_flag, plotting_flag);

        % Exposing the trained network to track B (potentially with 
        % plasticity). 
        [pf_cell_B] = GET_TRACK_PFs(n_pos_B, prob_pf_center, n_input);  
        
        [activity_B, W_EE_B, W_EI_B, W_IE_B, ratemap_B ] = ...
            EI_STDP_NETWORK(EE_plast(pp), EI_plast(pp), IE_plast(pp), eta, ...
            n_laps, W_EE, W_EI, W_IE, W_input, pf_cell_B, n_pos_B, save_flag, plotting_flag); 
        
        % Producing and correlating the rate maps
        [AAp_corr(pp, ii), AB_corr(pp, ii)] = FIXED_ORDER_RATE_MAP(ratemap_A, ...
            ratemap_Ap, ratemap_B, ii, save_flag, plotting_flag);

        % Cell pair correlations
        activity_Ac = activity_A(:, (floor(size(activity_A, 2)/2)):end);      
        activity_Apc = activity_Ap(:, (floor(size(activity_A, 2)/2)):end);    
        activity_Bc = activity_B(:, (floor(size(activity_B, 2)/2)):end);
        
        kendall_corr_A = corr(activity_Ac', activity_Ac', 'Type', 'Kendall', 'Rows', 'Complete');
        kendall_corr_Aprime = corr(activity_Apc', activity_Apc', 'Type', 'Kendall', 'Rows', 'Complete');
        kendall_corr_B = corr(activity_Bc', activity_Bc', 'Type', 'Kendall', 'Rows', 'Complete');
        
        pearson_corr_AA(pp, ii) = corr(kendall_corr_A(:), kendall_corr_Aprime(:), 'Rows', 'complete');
        pearson_corr_AB(pp, ii) = corr(kendall_corr_A(:), kendall_corr_B(:),'Rows', 'complete');
        
        % Saving activity 
        if save_flag 
            save(strcat('Activity_A', num2str(ii), '.mat'), 'activity_A'); 
            save(strcat('Activity_Ap', num2str(ii), '.mat'), 'activity_Ap'); 
            save(strcat('Activity_B', num2str(ii), '.mat'), 'activity_B'); 
        end
    end
end

%% Plotting 
% Pearson correlation of Kendall correlation
if n_init > 1 && plotting_flag == 1
    figure
    bar(1:2:(2 * 5 - 1), mean(pearson_corr_AA, 2), 0.3, 'b'); hold on 
    bar(2:2:(2 * 5), mean(pearson_corr_AB, 2), 0.3,  'r'); hold on 
    for ii = 1:5
        plot([2 * ii - 1, 2 * ii - 1], [mean(pearson_corr_AA(ii, :)) + std(pearson_corr_AA(ii, :)) / sqrt(n_init), ...
            mean(pearson_corr_AA(ii, :)) - std(pearson_corr_AA(ii, :)) / sqrt(n_init)], 'k-'); 

        plot([2 * ii, 2 * ii], [mean(pearson_corr_AB(ii, :)) + std(pearson_corr_AB(ii, :)) / sqrt(n_init), ...
            mean(pearson_corr_AB(ii, :)) - std(pearson_corr_AB(ii, :)) / sqrt(n_init)], 'k-'); 
    end
    ylabel('Pearson correlation'); 
    legend('A-A prime', 'A-B'); 
    title('Pearson correlation of Kendall correlations'); 
end

% Pearson correlation of rate maps 
if n_init > 1 && plotting_flag == 1
    figure
    bar(1:2:(2 * 5 - 1), mean(AAp_corr, 2), 0.3, 'b'); hold on 
    bar(2:2:(2 * 5), mean(AB_corr, 2), 0.3,  'r'); hold on 
    for ii = 1:5
        plot([2 * ii - 1, 2 * ii - 1], [mean(AAp_corr(ii, :)) + std(AAp_corr(ii, :)) / sqrt(n_init), ...
            mean(AAp_corr(ii, :)) - std(AAp_corr(ii, :)) / sqrt(n_init)], 'k-'); 

        plot([2 * ii, 2 * ii], [mean(AB_corr(ii, :)) + std(AB_corr(ii, :)) / sqrt(n_init), ...
            mean(AB_corr(ii, :)) - std(AB_corr(ii, :)) / sqrt(n_init)], 'k-'); 
    end
    ylabel('Pearson correlation of rate maps'); 
    legend('A-A prime', 'A-B'); 
    title('Pearson correlation of rate maps'); 
end




