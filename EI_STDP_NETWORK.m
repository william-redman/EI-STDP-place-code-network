function [ activity, W_EE, W_EI, W_IE, A ] = EI_STDP_NETWORK(EE_plast, EI_plast, ...
    IE_plast, eta, n_laps, W_EE, W_EI, W_IE, W_inputE, rf_cell, n_pos)
%% EI STDP NETWORK NOVEL
%------------------------------------------------------------------------%
%   This script is for the different track analysis. It is exactly like
%   EI_STDP_NETWORK_OG.m, but it loads in a pre-determined track and 
%   weights. For more details, see the comments in EI_STDP_NETWORK_OG.m.
%
%   Written by WTR 10/08/2019 // Last updated by WTR 06/30/2021 
%-------------------------------------------------------------------------%
%% Parameters 
n_input = 1000;                             % number of tuned input neurons                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
n_excit = 500;                              % number of excitatory neurons
n_inhib = 50;                               % number of inhibititory neurons                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ; %probability of any excitatory neuron connecting to an inhibitory neuron
I_F_thresh = 1.0;                           % integrate and fire threshold 

n_steps = n_laps * n_pos;                   % number of steps on the track 
input_refract_length = 0;                   % input refractory length 
excit_refract_length = 0.02;                % excit refractory length
inhib_refract_length = 0.02;                % inhibitory refractory length
W_upper_limit = 0.5;                        % maximum weight we allow the E to I weights to take
alpha_P = 0.05;                             % noise strength
rf_rate = 8;                                % input neuron rate (in Hz) at rf 
rf_width = 10;                              % controls the Gaussian rf width (note it's actually the inverse) 

corr_window_size = 0.25;                    % size of window used for smoothing activity for correlation analysis

%% Initiliazing the network
% Setting up the rfs of the input neurons (continuous)
% Here we've already loaded the necessary structure (the weights and the
% rf_mat) from the input.
  
%% Running the network 
% Determining the speed of the animal running through the track
mean_dt = 0.007; sigma_dt = 0;
mean_v = 1; sigma_v = 0;

dt_vec = normrnd(mean_dt, sigma_dt, 1, ceil(n_steps/mean_dt));
dt_vec(dt_vec < 0 ) = 0;
total_time_vec = cumsum(dt_vec);
v_vec = normrnd(mean_v, sigma_v, 1, ceil(n_steps/mean_dt)); 
v_vec = [0, v_vec];
v_vec(v_vec < 0) = 0;

% Determining the path of the animal running through the track
positions = 0.5 * (v_vec(1:end-1) + v_vec(2:end)) .* dt_vec; 
positions = mod(cumsum(positions), n_pos);
integer_pos = floor(positions) + 1; 
time_pos_switch = find((integer_pos(2:end) - integer_pos(1:(end - 1))) ~= 0); 
IE_average_weight_change = zeros(1, length(time_pos_switch));
EI_average_weight_change = zeros(1, length(time_pos_switch));
EE_average_weight_change = zeros(1, length(time_pos_switch));
W_EE_old = W_EE; 
W_EI_old = W_EI;
W_IE_old = W_IE; 
integer_traj = integer_pos(time_pos_switch); 

% Initializing matrices & vectors to store various information about the
% dynamics
spike_mat_excit = zeros(n_excit, length(total_time_vec)); 
spike_mat_inhib = zeros(n_inhib, length(total_time_vec)); 
excit_spikes = zeros(1, n_excit); 
inhib_spikes = zeros(1, n_inhib); 
excit_spikes_2 = zeros(1, n_excit); 
inhib_spikes_2 = zeros(1, n_inhib); 
excit_cum_input = zeros(1, n_excit); 
inhib_cum_input = zeros(1, n_inhib); 
x_excit = zeros(1, n_excit); 
x_inhib = zeros(1, n_inhib);
input_most_recent_fire_times_vec = zeros(1, n_input) - 100; 
excit_most_recent_fire_times_vec = zeros(1, n_excit) - 100;
inhib_most_recent_fire_times_vec = zeros(1, n_inhib) - 100;

% Running the network on the linear track 
for tt = 1:length(total_time_vec)  
    % Getting the position and time step
    pos = positions(tt);
    dt = dt_vec(tt);
    
    % Finding how far away the animal is from all of the rf centers and
    % calculating the probability of each of the input neurons firing
    input_prob_firing_vec = zeros(1, n_input); 
    for ii = 1:n_input
        rfs = rf_cell{ii}; 
        possible_distance_vec = [abs(pos - rfs); (n_pos) - abs(pos - rfs)];
        distance_vec = min(possible_distance_vec); 
        input_prob_firing_vec(ii) = sum(exp(-distance_vec.^2 * rf_width)); 
    end
    input_prob_firing_vec = input_prob_firing_vec + alpha_P * pinknoise(n_input);
    input_prob_firing_vec(input_prob_firing_vec < 0) = 0; 
    input_prob_firing_vec = input_prob_firing_vec * rf_rate * dt; 
    
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
    excit_cum_input((total_time_vec(tt) - excit_most_recent_fire_times_vec) < excit_refract_length) = 0; 
    
    % Updating excitatory spikes 
    excit_most_recent_fire_times_vec(excit_cum_input >= I_F_thresh) = total_time_vec(tt);
    excit_spikes_2(excit_cum_input >= I_F_thresh) = 1; 
    excit_spikes_2(excit_cum_input < I_F_thresh) = 0;
    excit_cum_input(excit_cum_input >= I_F_thresh) = 0;       
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
    d_W_EI(d_W_EI > 1) = 1; d_W_EI(d_W_EI < -1) = -1;
    d_W_IE = (excit_spikes' * x_inhib)' + inhib_spikes' * x_excit - 0.01 * ones(size(W_IE)); 
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
    
end

%% Smoothing spike matrix
activity_window = round(corr_window_size ./ mean_dt);
activity = zeros(n_excit, floor(length(total_time_vec)/activity_window));

for ii = 1:floor(length(total_time_vec)/activity_window)
    activity(:, ii) = mean(spike_mat_excit(:, (1 + (ii - 1) * activity_window):(ii * activity_window)), 2); 
end

%% Computing the rate map 
window = round(size(spike_mat_excit, 2)/2):size(spike_mat_excit, 2); 
E = spike_mat_excit(:, window);
A = zeros(n_excit, n_pos);

for ii = 1:(n_pos)
    times_ii = find(integer_pos(window) == ii);
    for jj = 1:n_excit
        E_jj = E(jj,:);
        A(jj, ii) = sum(E_jj(times_ii)) / length(times_ii);
    end
end 

%% Plotting 
if plotting_flag
    PLOTTING_EI_STDP_NETWORK 
end

end



