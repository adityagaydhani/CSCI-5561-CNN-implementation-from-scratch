% Pool2x2   Perform 2x2 max pooling with stride 2
%
%   [y] = Pool2x2(x) performs 2x2 max pooling on x with stride 2.
%
%   Input:
%       x = a general h x w x c tensor and matrix
%
%   Output:
%       y = h/2 x w/2 x c output of the max pooling operation

function [y] = Pool2x2(x)
    [height, width, c] = size(x);
    
    x_row_version = zeros(4, height/2 * width/2, c);
    y_row_version = zeros(1, height/2 * width/2, c);
    y = zeros(floor(height/2), floor(width/2), c);

    for iDepth = 1:c
        % Dim: 4 x (h/2 x w/2) x 1
        x_row_version(:, :, iDepth) = im2col(x(:, :, iDepth), [2 2],...
            'distinct');
        
        % Dim: 1 x (h/2 x w/2) x 1
        y_row_version(:, :, iDepth) = max(x_row_version(:, :, iDepth));
        
        % Dim: h/2 x w/2 x 1
        y(:, :, iDepth) = reshape(y_row_version(:, :, iDepth),...
            [height/2 width/2 1]);
    end

end