% ReLu  Activate input using Rectified Linear Unit
%
%   [y] = ReLu(x) activates the input x using ReLu and sets it to y.
%
%   Input:
%       x   = a general tensor, matrix or vector
%
%   Output:
%       y   = ReLu of x with same input size

function [y] = ReLu(x)
    y = max(0, x);
end