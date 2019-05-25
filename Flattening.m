% Flattening    Flatten a given tensor in column major format
%
%   [y] = Flattening(x) flattens the input tensor x to the output vector y.
%
%   Input:
%       x   = h x w x c input tensor
%
%   Output:
%       y   = (h x w x c) x 1 output vector

function [y] = Flattening(x)
    [h, w, c] = size(x);
    y = reshape(x, [h * w * c, 1]);
end