function [W_EE, W_EI, W_IE, W_inputE, pf_cell, activity, ab_sim, nn_sim, half_sim, decoding_error ] ...
    = EI_STDP_NETWORK_OG(EE_plast, EI_plast, IE_plast, eta, n_laps, n_pos, prob_pf_center, save_flag, plotting_flag)
%% EI STDP NETWORK OG
%------------------------------------------------------------------------%
%   This script creates and runs a "naive" neural network on track A.
%
%   Written by WTR 10/08/2019 // Last updated by WTR 07/05/2021
%-------------------------------------------------------------------------%
%% Parameters 
n_input = 1000;                             % number of tuned input neurons                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
n_excit = 500;                              % number of excitatory neurons
n_inhib = 50;                               % number of inhibititory neurons 
prob_E_recurrent_connectivity = 0.25;       % probability of any given excitatory neuron will connect to another excitatory neuron
prob_I_E_connectivity = 0.30;               % probability of any inhibitory neuron connecting to an excitatory neuron
prob_E_I_connectivity = 0.175;              % probability of any excitatory neuron connecting to an inhibitory neuron                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ; %probability of any excitatory neuron connecting to an inhibitory neuron
I_F_thresh = 1.0;                           % integrate and fire threshold 
test_time = 10;                             % number of laps used to compute metrics

n_steps = n_laps * n_pos;                   % number of steps on the track 
input_refract_length = 0;                   % input neuron refractory length 
excit_refract_length = 0.02;                % excit refractory length
inhib_refract_length = 0.02;                % inhibitory refractory length
initial_weight_max_input = 0.05;            % initial weight maximum input to E
initial_weight_max_EE = 0.05 / prob_E_recurrent_connectivity; % initial weight maximum for E to E
initial_weight_max_EI = 0.05 / prob_I_E_connectivity;  % initial weight maximum for E to I
initial_weight_max_IE = 0.05 / prob_E_I_connectivity; % initial weight maximum for I to E
W_upper_limit = 0.5;                        % maximum weight we allow the E to I weights to take
alpha_P = 0.05;                             % pink noise strength
pf_rate = 8;                                % input neuron rate (in Hz) at center of pf
pf_width = 10;                              % controls the Gaussian pf width (note it's actually the inverse) 

corr_window_size = 0.25;                    % size of window used for smoothing activity for correlation analysis

%% Initiliazing the input neurons 
% Setting up the pfs of the input neurons (continuous)
pf_cell = GET_TRACK_PFs(n_pos, prob_pf_center, n_input);

% Setting up the weight matrices
% Note: W_inputE is not plastic
W_inputE = rand(n_input, n_excit) * initial_weight_max_input;

W_EE = rand(n_excit, n_excit); 
W_EI = rand(n_excit, n_inhib); 
W_IE = rand(n_inhib, n_excit);

W_EE(W_EE > prob_E_recurrent_connectivity) = 0;
W_EE(diag(ones(1,n_excit)) == 1) = 0;       % no neurons can connect recurrently to themselves
W_EE = W_EE * initial_weight_max_EE;

W_EI(W_EI > prob_E_I_connectivity) = 0;
W_EI = W_EI * initial_weight_max_EI; 

W_IE(W_IE > prob_I_E_connectivity) = 0;
W_IE = W_IE * initial_weight_max_IE;
  
%% Running the network 
% Determining the speed of the animal running through the track
mean_dt = 0.007; sigma_dt = 0;              % time scale of updating activity
mean_v = 1; sigma_v = 0;                    % mean and std speed of animal

dt_vec = normrnd(mean_dt, sigma_dt, 1, ceil(n_steps/mean_dt));
dt_vec(dt_vec < 0 ) = 0;
total_time_vec = cumsum(dt_vec);
v_vec = normrnd(mean_v, sigma_v, 1, ceil(n_steps/mean_dt)); 
v_vec = [0, v_vec];
v_vec(v_vec < 0) = 0;

% Determining the path of the animal running through the track
positions = 0.5 * (v_vec(1:end-1) + v_vec(2:end)) .* dt_vec; % generating the positions at each time step from dt and the difference in velocities (trying to make it smooth)
positions = mod(cumsum(positions), n_pos);
integer_pos = floor(positions) + 1;         % the integer values of all the positions 
time_pos_switch = find((integer_pos(2:end) - integer_pos(1:(end - 1))) ~= 0); % time right before the position switches from one integer block to another
time_lap_switch = find((integer_pos(2:end) - integer_pos(1:(end - 1))) < 0); % time right before the position switches from one lap to another
time_lap_switch = [0, time_lap_switch, length(positions)]; % time right before the position switches from one lap to another
integer_traj = integer_pos(time_pos_switch); % the integer values of the trajectory (this and the above are used in PLOTTING_EI_STDP_NETWORK.m)
test_laps = time_lap_switch(1:test_time:end); % laps where the metrics will be computed on

% Initializing matrices to store weight change
IE_average_weight_change = zeros(1, length(time_pos_switch));
EI_average_weight_change = zeros(1, length(time_pos_switch));
EE_average_weight_change = zeros(1, length(time_pos_switch));
W_EE_old = W_EE; 
W_EI_old = W_EI;
W_IE_old = W_IE; 

% Initializing matrices & vectors to store various information about the
% dynamics
spike_mat_excit = zeros(n_excit, length(total_time_vec)); % stores excitatory spikes 
spike_mat_inhib = zeros(n_inhib, length(total_time_vec));  
excit_spikes = zeros(1, n_excit);                       % stores the spiking at each time step of the excitatory neurons 
inhib_spikes = zeros(1, n_inhib);                       
excit_spikes_2 = zeros(1, n_excit);                     % buffer that stores the t+1 activity so we can still keep the t activity
inhib_spikes_2 = zeros(1, n_inhib); 
excit_cum_input = zeros(1, n_excit);                    % stores the voltage of the excitatory neurons
inhib_cum_input = zeros(1, n_inhib); 
x_excit = zeros(1, n_excit);                            % sotres the exponentials used in calculating the STDP changes
x_inhib = zeros(1, n_inhib);
input_most_recent_fire_times_vec = zeros(1, n_input) - 100; % stores the most recent firing times of the input neurons 
excit_most_recent_fire_times_vec = zeros(1, n_excit) - 100;
inhib_most_recent_fire_times_vec = zeros(1, n_inhib) - 100;

% Initializing matrices that store the similarity values
ab_sim = zeros(1, n_laps / test_time);           % absolute similarity
nn_sim = zeros(1, n_laps / test_time);           % nearest neighbor similarity
half_sim = zeros(1, n_laps / test_time);         % halfway around track similarity

% Initializing matrix that stores the decoding performance 
decoding_error = zeros(1, n_laps / test_time); 

% Running the network on the linear track 
for tt = 1:length(total_time_vec)  
    % Getting the position and time step
    pos = positions(tt);
    dt = dt_vec(tt);
    
    % Finding how far away the animal is from all of the rf centers and
    % calculating the probability of each of the input neurons firing
    input_prob_firing_vec = zeros(1, n_input); 
    for ii = 1:n_input
        rfs = pf_cell{ii}; 
        possible_distance_vec = [abs(pos - rfs); (n_pos) - abs(pos - rfs)];
        distance_vec = min(possible_distance_vec); 
        input_prob_firing_vec(ii) = sum(exp(-distance_vec.^2 * pf_width)); 
    end
    input_prob_firing_vec = input_prob_firing_vec + alpha_P * pinknoise(n_input);
    input_prob_firing_vec(input_prob_firing_vec < 0) = 0; 
    input_prob_firing_vec = input_prob_firing_vec * pf_rate * dt; 
    
    % Finding which input neurons fired 
    coin_flips = rand(1, n_input);
    input_neurons_fired = find(coin_flips < input_prob_firing_vec & ...
        (total_time_vec(tt) - input_most_recent_fire_times_vec) > input_refract_length);
    input_most_recent_fire_times_vec(input_neurons_fired) = total_time_vec(tt);
    input_spikes = zeros(1, n_input); 
    input_spikes(input_neurons_fired) = 1; 
    
    % Updating excitatory cumulative input 
    excit_cum_input = excit_cum_input * exp(- dt) + input_spikes * W_inputE + ...
        excit_spikes * W_EE - inhib_spikes * W_IE + alpha_P * pinknoise(n_excit);     
    excit_cum_input((total_time_vec(tt) - excit_most_recent_fire_times_vec) < excit_refract_length) = 0; % setting voltage equal to 0 for any neurons still in their absolute refractory period
    
    % Updating excitatory spikes 
    excit_most_recent_fire_times_vec(excit_cum_input >= I_F_thresh) = total_time_vec(tt);
    excit_spikes_2(excit_cum_input >= I_F_thresh) = 1;  % storing current timestep spikes
    excit_spikes_2(excit_cum_input < I_F_thresh) = 0;
    excit_cum_input(excit_cum_input >= I_F_thresh) = 0; % setting the input back to 0 for the neurons that spiked        
    spike_mat_excit(:, tt) = excit_spikes_2;
       
    % Updating inhibitory cumulative input 
    inhib_cum_input = inhib_cum_input * exp(-dt) + excit_spikes * W_EI + alpha_P * pinknoise(n_inhib);     
    inhib_cum_input((total_time_vec(tt) - inhib_most_recent_fire_times_vec) < inhib_refract_length) = 0;
    
    % Updating inhibitory spikes
    inhib_most_recent_fire_times_vec(inhib_cum_input >= I_F_thresh) = total_time_vec(tt);
    inhib_spikes_2(inhib_cum_input >= I_F_thresh) = 1;
    inhib_spikes_2(inhib_cum_input < I_F_thresh) = 0;
    inhib_cum_input(inhib_cum_input >= I_F_thresh) = 0;       
    spike_mat_inhib(:, tt) = inhib_spikes_2;
    
    % Updating the spikes for the next times step
    inhib_spikes = inhib_spikes_2; 
    excit_spikes = excit_spikes_2;
        
    % Updating the inhibitory/excitatory exponentials for STDP
    x_excit = exp(-dt) * x_excit;
    x_excit(excit_spikes == 1) = 1 + x_excit(excit_spikes == 1); 
    x_inhib = exp(-dt) * x_inhib;
    x_inhib(inhib_spikes == 1) = 1 + x_inhib(inhib_spikes == 1);
             
    % Updating weights of the network. Remember that the input weights are 
    % not plastic but the other weights are 
    d_W_EI = (inhib_spikes' * x_excit)' - excit_spikes' * x_inhib;
    d_W_EI(d_W_EI > 1) = 1; d_W_EI(d_W_EI < -1) = -1; % making the max change in weight be eta
    d_W_IE = (excit_spikes' * x_inhib)' + inhib_spikes' * x_excit - 0.01 * ones(size(W_IE)); % shifting the curve by 1%
    d_W_IE(d_W_IE > 1) = 1; d_W_IE(d_W_IE < -1) = -1;
    d_W_EE = x_excit' * excit_spikes - excit_spikes' * x_excit; 
    d_W_EE(d_W_EE > 1) = 1; d_W_EE(d_W_EE < -1) = -1;
     
    W_EI = W_EI + EI_plast * eta * d_W_EI;    
    W_IE = W_IE + IE_plast * eta * d_W_IE; 
    W_EE = W_EE + EE_plast * eta * d_W_EE; 

    % Enforcing weight boundaries 
    W_EE(W_EE < 0 ) = 0; 
    W_EI(W_EI < 0) = 0; 
    W_IE(W_IE < 0) = 0;
    W_EI(W_EI > W_upper_limit) = W_upper_limit; 
    W_IE(W_IE > W_upper_limit) = W_upper_limit; 
    W_EE(W_EE > W_upper_limit) = W_upper_limit; 
    
    % Evaluating weight changes across spatial positions 
    if ~isempty(intersect(tt, time_pos_switch))
        id = find(time_pos_switch == tt); 
        IE_average_weight_change(id) = mean(abs(W_IE - W_IE_old), 'all');
        EI_average_weight_change(id) = mean(abs(W_EI - W_EI_old), 'all');
        EE_average_weight_change(id) = mean(abs(W_EE - W_EE_old), 'all');
        W_IE_old = W_IE; 
        W_EI_old = W_EI;
        W_EE_old = W_EE;        
    end
    
    % Evaluating similarity and decoding
    if ~isempty(intersect(tt, time_lap_switch(1:test_time:end)))
        test_lap_id = find(test_laps == tt);
        window = (test_laps(test_lap_id - 1) + 1):test_laps(test_lap_id);
        E = spike_mat_excit(:, window); 
        p = integer_pos(window); 
        
        [decoding_error(test_lap_id - 1)] = BAYESIAN_DECODER(E, p, n_pos);
        
        A = zeros(n_excit, n_pos); 
        
        for ii = 1:(n_pos)
            times_ii = find(p == ii);
            for jj = 1:n_excit
                E_jj = E(jj,:);
                A(jj, ii) = sum(E_jj(times_ii)) / length(times_ii); 
            end
        end
        
        [max_val, max_ids] = max(A, [], 2); 
        A(max_val == 0, :) = []; max_ids(max_val == 0) = [];
        ordered_A = sort(A, 2); 
        
        ab_diff = ordered_A(:, end) - ordered_A(:, (end - 1)); ab_sum = ordered_A(:, end) + ordered_A(:, (end - 1)); 
        ab_sim(test_lap_id - 1) = 1 - nanmean(ab_diff ./ ab_sum);       
        
        nn_sim_vec = zeros(1, n_excit);
        half_sim_vec = zeros(1, n_excit); 
        
        ids_shift_left = max_ids - 1; ids_shift_left(ids_shift_left == 0) = n_pos; 
        ids_shift_right = max_ids + 1; ids_shift_right(ids_shift_right > n_pos) = 1;
        halfway_ids = max_ids - n_pos/2; 
        halfway_ids(halfway_ids < 1) = halfway_ids(halfway_ids < 1) + n_pos; 
        
        for nn = 1:size(A, 1)
            nn_diff_1 = A(nn, max_ids(nn)) - A(nn, ids_shift_left(nn)); nn_sum_1 = A(nn, max_ids(nn)) + A(nn, ids_shift_left(nn)); 
            nn_diff_2 = A(nn, max_ids(nn)) - A(nn, ids_shift_right(nn)); nn_sum_2 = A(nn, max_ids(nn)) + A(nn, ids_shift_right(nn));
            nn_sim_vec(nn) = 1 - 0.5 * (nanmean(nn_diff_1./nn_sum_1 + nn_diff_2./nn_sum_2));
            
            half_diff = A(nn, max_ids(nn)) - A(nn, halfway_ids(nn)); half_sum = A(nn, max_ids(nn)) + A(nn, halfway_ids(nn)); 
            half_sim_vec(nn) = 1 - half_diff ./ half_sum; 
        end
        
        nn_sim(test_lap_id - 1) = nanmean(nn_sim_vec); 
        half_sim(test_lap_id - 1) = nanmean(half_sim_vec); 
                     
    end
end

%% Smoothing spike matrix for correlation analysis
activity_window = round(corr_window_size ./ mean_dt);
activity = zeros(n_excit, floor(length(total_time_vec)/activity_window));

for ii = 1:floor(length(total_time_vec)/activity_window)
    activity(:, ii) = mean(spike_mat_excit(:, (1 + (ii - 1) * activity_window):(ii * activity_window)), 2); 
end

% %% Decoding 
% window = round(size(spike_mat_excit, 2)/2):size(spike_mat_excit, 2); 
% E = spike_mat_excit(:, window);
% p = integer_pos(window); 
% [decoding_error] = BAYESIAN_DECODER(E, p, n_pos);

%% Plotting 
if plotting_flag
    PLOTTING_EI_STDP_NETWORK 
    
    % Plotting similarity 
    figure
    plot(ab_sim, 'ko-'); hold on
    plot(nn_sim, 'bo-');
    plot(half_sim, 'ro-');
    title('Similarity through time');
    legend('Absolute', 'Nearest neighbor', 'Halfway');
    if save_flag
        savefig('absolute_sim.fig');
    end

end

end



