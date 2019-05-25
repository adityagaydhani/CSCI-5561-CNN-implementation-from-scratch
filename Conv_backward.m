% Conv_backward Compute loss derivative w.r.t. weights and bias
%
%   [dLdw, dLdb] = Conv_backward(dLdy, x, w_conv, b_conv, y) computes the
%   loss derivative w.r.t the given weight w_conv and bias b_conv
%
%   Input:
%       dLdy    = h x w x c2 loss derivative w.r.t y
%       x       = h x w x c1 input to the convolution operation
%       w_conv  = f x f x c1 x c2 weight tensor
%       b_conv  = c2 x 1 bias vector
%       y       = h x w x c2 output of the convolution operation
%
%   Output:
%       dLdw    = 1 x (f x f x c1 x c2) loss derivative w.r.t weight w
%       dLdb    = 1 x (c2 x 1) loss derivative w.r.t bias b

function [dLdw, dLdb] = Conv_backward(dLdy, x, w_conv, b_conv, y)
    [height, width, ~] = size(x); % height = 14, width = 14
    [f, ~, ~, c2] = size(w_conv); % f = 3, c2 = 3
    x_reshaped = reshape(x, [height width]); % Dim: 14 x 14
    
    lr_pad = zeros(height, 1); tb_pad = zeros(1, width+2);
    x_padded = [tb_pad; [lr_pad x_reshaped lr_pad]; tb_pad]; % Dim: 16 x 16   
    x_row_version = im2col(x_padded, [f f]); % Dim: (3x3) x (14x14)
    
    % Dim: 3 x (14x14)
    dLdy_reshaped = transpose(reshape(dLdy, [height*width c2]));
    
    dLdw = dLdy_reshaped * transpose(x_row_version); % Dim: 3 x (3x3)
    dLdw = reshape(dLdw, [1 c2 * f * f]); % Dim: 1 x 27
    
    dLdb = transpose(sum(dLdy_reshaped, 2)); % Dim.: 1 x 3
end