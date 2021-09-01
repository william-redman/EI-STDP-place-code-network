%% PLOTTING EI STDP NETWORK
%-------------------------------------------------------------------------%
%   All the metrics that we might want to look at after running the spiking
%   neural network. 
%
%   Written by WTR 09/29/2019 // Last updated by WTR 06/30/2021
%-------------------------------------------------------------------------%
%% Plotting
% Excitatory activity raster plot
figure
plot(total_time_vec, [0:1/(n_excit-1):1]' .* spike_mat_excit, 'k.');

%Inhibitory activity raster plot 
figure
plot(total_time_vec, [0:1/(n_inhib-1):1]' .* spike_mat_inhib, 'k.');

% W_EE distribution
figure
histogram(W_EE(:), 20); 
title('W_{EE}');

% W_IE distribution
figure 
histogram(W_IE(:), 20);
title('W_{IE}');

% W_EI distribution
figure
histogram(W_EI(:), 20);
title('W_{EI}');

% Rate maps
% Here we specificially look at the last 25% of the experimental time. 
av_fire_rate_mat = zeros(n_excit, n_pos);

for jj = 1:(n_pos)
    times_at_pos = time_pos_switch(floor(3*n_steps/4 - 1)) + find(integer_pos((time_pos_switch(floor(3*n_steps/4 - 1)) + 1):end) == jj);  
    
    for kk = 1:n_excit
        output_neuron_spike_train = spike_mat_excit(kk,:);  
        av_fire_rate_mat(kk, jj) = sum(output_neuron_spike_train(times_at_pos)) ...
            / length(times_at_pos);    
    end
end

% Normalizing by the maximum activity
[maxval, m] = max(av_fire_rate_mat, [], 2);
av_fire_rate_mat(find(maxval == 0),:) = []; m(find(maxval == 0)) = []; % removing any neurons that did not spike in the last 25% of time
[~, id] = sort(m);

ordered_av_rates = av_fire_rate_mat(id, :);

figure
imagesc(ordered_av_rates ./ max(ordered_av_rates, [], 2));
title('Ordered normalized rates');
xlabel('Position'); ylabel('Excitatory neuron');
if save_flag
    savefig('ordered_normalized_rates.fig'); 
end

% Average weight matrix change
figure
plot(1:length(time_pos_switch), IE_average_weight_change, 'k-'); 
title('Average W_{IE} changes at each position'); 
xlabel('Time'); 
ylabel('|\Delta W_{IE}|'); 

figure
plot(1:length(time_pos_switch), EI_average_weight_change, 'k-'); 
title('Average W_{EI} changes at each position'); 
xlabel('Time'); 
ylabel('|\Delta W_{EI}|');

figure
plot(1:length(time_pos_switch), EE_average_weight_change, 'k-'); 
title('Average W_{EE} changes at each position'); 
xlabel('Time'); 
ylabel('|\Delta W_{EE}|');

% Plotting average activity 
conv_window = 100;                          % window for smoothing the rate
E_av_activity = mean(spike_mat_excit); 
E_av_activity = conv(E_av_activity, ones(1, conv_window)); 
E_av_activity((end - conv_window + 2):end) = [];

figure
plot(1:length(E_av_activity), E_av_activity, 'k-'); 
xlabel('Time')
ylabel('Average activity'); 
title('Average excitatory activity'); 

I_av_activity = mean(spike_mat_inhib); 
I_av_activity = conv(I_av_activity, ones(1, conv_window)); 
I_av_activity((end - conv_window + 2):end) = [];

figure
plot(1:length(I_av_activity), I_av_activity, 'k-'); 
xlabel('Time')
ylabel('Average activity'); 
title('Average inhibitory activity'); 







