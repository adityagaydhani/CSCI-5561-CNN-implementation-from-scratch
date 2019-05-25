% ReLu_backward     Compute loss derivative w.r.t the given input
%
%   Input:
%       dLdy    = loss derivative w.r.t output y
%       x       = input tensor, matrix, or vector
%       y       = output tensor, matrix, or vector
%
%   Output:
%       dLdx    = loss derivative w.r.t input x

function [dLdx] = ReLu_backward(dLdy, x, y)
    dydx = zeros(size(dLdy));
    
    dydx(x > 0) = 1;
    dydx(x < 0) = 0;
    dydx(x == 0) = 0.5;
    
    dLdx = dLdy .* dydx;
end