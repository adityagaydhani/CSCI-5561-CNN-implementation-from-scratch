% Pool2x2_backward  Compute loss derivative w.r.t input to the pooling
%                   layer
%
%   [dLdx] = Pool2x2_backward(dLdy, x, y) computes the loss derivative
%   w.r.t the input x using dLdy
%
%   Input:
%       dLdy    = h/2 x w/2 x c loss derivative w.r.t output y
%       x       = h x w x c input to the pooling layer
%       y       = h/2 x w/2 x c output of the pooling layer
%
%   Output:
%       dLdx    = zzzh x w x c loss derivative w.r.t the input x

function [dLdx_rv] = Pool2x2_backward(dLdy, x, y)
    [height, width, c] = size(x);
    
    x_row_version = zeros(4, height/2 * width/2, c);
    dLdy_flattened_xy = zeros(1, height/2 * width/2, c);
    
    for iDepth = 1:c
        % Dim: 4 x (h/2 x w/2) x 1
        x_row_version(:, :, iDepth) = im2col(x(:, :, iDepth), [2 2],...
            'distinct');
        
        % Dim: 1 x (h/2 x w/2) x 1
        dLdy_flattened_xy(1, :, iDepth)...
            = reshape(dLdy(:, :, iDepth), [1 size(dLdy, 1)*size(dLdy, 2)]);
    end
    
    [~, maxIdx] = max(x_row_version); % Dim.: 1 x (h/2 x w/2) x c
    
    dLdx_rv = zeros(size(x_row_version)); % Dim.: 4 x (h/2 x w/2) x c
    for iColNo = 1:size(dLdx_rv, 2)
        for iDepth = 1:c
            dLdx_rv(maxIdx(1, iColNo, iDepth), iColNo, iDepth)...
                = dLdy_flattened_xy(1, iColNo, iDepth);
        end
    end
    
    dLdx = zeros(size(x)); % Dim.: h x w x c
    for iDepth = 1:c
        dLdx(:, :, iDepth) = col2im(...
            dLdx_rv(:, :, iDepth), [2 2], [height width], 'distinct');
    end
end