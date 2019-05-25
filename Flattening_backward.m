% Flattening_backward    Compute loss derivative w.r.t input x
%
%   [dLdx] = Flattening_backward(dLdy, x, y) computes loss derivative w.r.t
%   input x to the flatten layer.
%
%   Input:
%       dLdy    = 1 x (h x w x c) loss derivative w.r.t output y to the
%       flatten layer
%       x       = h x w x c input to the flatten layer
%       y       = (h x w x c) x 1 output of the flatten layer
%
%   Output:
%       dLdx    = h x w x c loss derivative w.r.t input x

function [dLdx] = Flattening_backward(dLdy, x, y)
    dLdx = reshape(dLdy, size(x));
end