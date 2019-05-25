% Loss_euclidean    Compute Euclidean distance from the prediction and the
%                   ground truth
%
%   [L, dLdy] = Loss_euclidean(y_tilde, y) computes the euclidean distance
%   given prediction y_tilde and ground truth y.
%
%   Input:
%       y_tilde = mx1 prediction vector
%       y       = mx1 ground truth vector
%
%   Output:
%       L       = loss
%       dLdy    = 1xm loss derivative w.r.t the prediction y_tilde

function [L, dLdy] = Loss_euclidean(y_tilde, y)
    L = sum((y - y_tilde) .^ 2);
    dLdy = transpose(-2 * (y - y_tilde));
end