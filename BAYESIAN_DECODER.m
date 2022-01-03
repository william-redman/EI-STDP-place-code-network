function [ e ] = BAYESIAN_DECODER(E, p, n_pos)
%-------------------------------------------------------------------------%
%   This function performs bayesian decoding on activity from a certain
%   number of laps. It builds the decoder on n-1 laps of the data and then
%   test on the remaining lap. The average error, computed over holding 
%   each lap out, is output. 
%
%   Written by WTR 07/05/2021 // Last updated by WTR 07/05/2021
%-------------------------------------------------------------------------%
%% Removing any extraneous data points
if p(1) ~= 1
    start_id = min(find(p == 1)); 
    p(1:(start_id-1)) = [];
    E(:, 1:(start_id-1)) = [];
end

if p(end) ~= 10
    end_id = max(find(p == 10)); 
    if (end_id + 1) < size(E, 2)
        p((end_id + 1):end) = [];
        E(:, (end_id + 1):end) = []; 
    end
end

%% Finding the start to each lap 
laps = find((p(2:end) - p(1:(end - 1))) < 0);
laps = [0, laps, length(p)];

%% Computing average activity in each position per lap
n_laps = length(laps) - 1; 
n_neurons = size(E, 1); 
X = zeros(n_neurons, n_pos, n_laps);

for ii = 1:n_laps
    lap_ii_E = E(:, (laps(ii) + 1):laps(ii + 1)); 
    lap_ii_p = p(:, (laps(ii) + 1):laps(ii + 1));
    
    for jj = 1:n_pos 
        X(:, jj, ii) = mean(lap_ii_E(:, lap_ii_p == jj), 2); 
    end
end

%% Cross-validated Bayesian decoding 
e_vec = zeros(1, n_laps);  
n_bins = 10; 

for ii = 1:n_laps
    X_test = X(:, :, ii); 
    X_train = X; X_train(:, :, ii) = []; 

    % Computing probabilities of rates for each neuron at each position
    P_BA = cell(n_neurons, n_pos); 
    P_B = cell(n_neurons); 
    for jj = 1:n_pos 
        for kk = 1:n_neurons
            histo_edges = linspace(min(X_train(kk, jj, :)), max(X_train(kk, jj, :)), n_bins + 1); 
            h = histogram(X_train(kk, jj, :), 'BinEdges', histo_edges, 'Normalization', 'probability'); 
            P_BA{jj, kk} = {h.Values, (histo_edges(1:(end - 1)) + histo_edges(2:end))/2}; 
        end
    end
    
    % Computing probabilities of each neuron having a certain rate
    for kk = 1:n_neurons
        X_train_kk = X_train(kk, :, :); 
        histo_edges = linspace(min(X_train_kk(:)), max(X_train_kk(:)), n_bins + 1); 
        h = histogram(X_train_kk(:), 'BinEdges', histo_edges, 'Normalization', 'probability');   
        P_B{kk} = {h.Values, (histo_edges(1:(end - 1)) + histo_edges(2:end))/2}; 
    end
    
    % Decoding position from test activity
    decoded_pos = zeros(1, n_pos); 
    
    for pp = 1:n_pos
        bayes_prob = NaN(n_neurons, n_pos); 
        for kk = 1:n_neurons
            kk_rate = X_test(kk, pp);
            if kk_rate > 0
                [~, B_bin_id] = min(abs(kk_rate - P_B{kk}{2}));
                prob_B = P_B{kk}{1}(B_bin_id);
                
                if prob_B < 0.001
                    prob_B = 0.001;
                end
                
                for jj = 1:n_pos
                    [~, BA_bin_id] = min(abs(kk_rate - P_BA{jj, kk}{2}));
                    prob_BA = P_BA{jj, kk}{1}(BA_bin_id);
                    
                    if prob_BA < 0.001
                        prob_BA = 0.001;
                    end
                    
                    bayes_prob(kk, jj) = prob_BA * prob_B; % kkk_rate % / (prob_B * n_pos);
                end
            end
        end
        
        [~, decoded_pos(pp)] = max(nansum(log(bayes_prob)));         
    end
    
    error = min([abs(decoded_pos - (1:n_pos)); (n_pos - abs(decoded_pos - (1:n_pos)))]);
    e_vec(ii) = mean(error); 
    
end

e = mean(e_vec); 
