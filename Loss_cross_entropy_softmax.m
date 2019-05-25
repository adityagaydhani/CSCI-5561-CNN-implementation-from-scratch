% Loss_cross_entropy_softmax    Compute Euclidean distance from the input
%                               and the ground truth
%
%   [L, dLdy] = Loss_cross_entropy_softmax(x, y) computes the euclidean
%   distance given input to the softmax layer x and ground truth y.
%
%   Input:
%       x   = mx1 input to the softmax layer
%       y   = mx1 ground truth vector
%
%   Output:
%       L       = loss
%       dLdy    = 1xm loss derivative w.r.t the input x

function [L, dLdy] = Loss_cross_entropy_softmax(x, y)
    y_tilde = exp(x) / sum(exp(x));
    L = -sum(y .* log(y_tilde));
    dLdy = transpose(y_tilde - y);
end