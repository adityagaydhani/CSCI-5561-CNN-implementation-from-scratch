% Conv  Perform convolution operation
%
%   [y] = Conv(x, w_conv, b_conv) performs convolution operation
%   using given weight and bias.
%
%   Input:
%       x       = h x w x c1 input to the convolution operation
%       w_conv  = f x f x c1 x c2 weight tensor
%       b_conv  = c2 x 1 bias vector
%
%   Output:
%       y   = h x w x c2 output of convolution


function [y] = Conv(x, w_conv, b_conv)
    [height, width, ~] = size(x); % height = 14, width = 14
    [f, ~, ~, c2] = size(w_conv); % f = 3, c2 = 3
    x_reshaped = reshape(x, [height width]); % Dim: 14 x 14
    
    lr_pad = zeros(height, 1); tb_pad = zeros(1, width+2);
    x_padded = [tb_pad; [lr_pad x_reshaped lr_pad]; tb_pad]; % Dim: 16 x 16
    
    x_row_version = im2col(x_padded, [f f]); % Dim: (3x3) x (14x14)
    
    w_reshaped = reshape(w_conv, [f*f c2]); % Dim: (3x3) x 3
    
    % Dim: (14x14) x 3
    y = transpose(x_row_version) * w_reshaped + transpose(b_conv);
    
    y = reshape(y, [height width c2]);
end