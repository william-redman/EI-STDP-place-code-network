function [pf_cell] = GET_TRACK_PFs(n_pos, prob_pf_center, n_input)         
%% GET_TRACK_PFs.mat
%-------------------------------------------------------------------------%
%   This function generates the place fields of the input neurons used in
%   the spiking neural network. 
%
%   Written by WTR 05/16/2021 // Last updated by WTR 06/30/2021
%-------------------------------------------------------------------------%
%% Setting up the pfs of the input neurons
pf_cell = cell(1, n_input); 

for ii = 1:n_input 
    num_pfs = length(find(rand(1, n_pos) <= prob_pf_center)); 
    if num_pfs == 0
        pf_cell{ii} = rand() * n_pos; % making sure all input neurons have at least one PF
    else
        pf_cell{ii} = rand(1, num_pfs) * n_pos; 
    end
end

end

