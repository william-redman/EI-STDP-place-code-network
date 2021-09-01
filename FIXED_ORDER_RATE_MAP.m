function [ cAAp, cAB ] = FIXED_ORDER_RATE_MAP(A, Ap, B, counter, save_flag, plotting_flag)
%-------------------------------------------------------------------------%
%   This function plots the activity of Ap and B in the order given by A. 
%
%   Written by WTR 06/28/2021 // Last updated by WTR 06/30/2021 
%-------------------------------------------------------------------------%
%%
A = A ./ max(A, [], 2); 
Ap = Ap ./ max(Ap, [], 2); 
B = B ./ max(B, [], 2); 

if save_flag
    save(strcat('A_ratemap', num2str(counter),'.mat'), 'A'); 
    save(strcat('Ap_ratemap.mat', num2str(counter), '.mat'), 'Ap');
    save(strcat('B_ratemap', num2str(counter),'.mat'), 'B'); 
end

[~, mA] = max(A, [], 2); 
[~, idA] = sort(mA); 

cAAp = corr(A(:), Ap(:), 'rows', 'complete');
cAB = corr(A(:), B(:), 'rows', 'complete');

if plotting_flag
    figure
    imagesc(Ap(idA, :));
    title('A prime rate map ordered by A');
    savefig('Ap_ordered_by_A.fig');
    
    figure
    imagesc(B(idA, :));
    title('B prime rate map ordered by A');
    savefig('B_ordered_by_A.fig');
end

end

