% TrainSLP_linear   Train the linear single layer perceptron model
%
%   [w, b] = TrainSLP_linear(mini_batch_x, mini_batch_y) computes the
%   weight w and bias b by performing stochastic mini batch gradient
%   descent.
%
%   Input:
%       mini_batch_x    = kx1 array of cells where each cell is a batch of
%                         images
%       mini_batch_y    = kx1 array of cells where each cell is a batch of
%                         labels
%
%   Output:
%       w   = 10x196 weight matrix
%       b   = 10x1 bias vector

function [w, b] = TrainSLP_linear(mini_batch_x, mini_batch_y)
    nBatches = size(mini_batch_x, 1);
    gamma = 0.03; % learning rate
    lambda = 0.8; % decay rate
    w = normrnd(0, 1, [10 196]); b = zeros(10, 1);
    kBatchNo = 1;
    loss = zeros(10000, 1);
    
    for iIter = 1:10000
        % At every 1000th iteration, update the learning rate using the
        % decay rate.
        if mod(iIter, 1000) == 0
            gamma = lambda * gamma;
        end
        
        dLdw = zeros(10, 196); dLdb = zeros(10, 1);
        
        currBatch = mini_batch_x{kBatchNo};
        currLabels = mini_batch_y{kBatchNo};
        currBatchSize = size(currBatch, 2);
        
        for jImage = 1:currBatchSize
           x = currBatch(:, jImage); y = currLabels(:, jImage);
           y_tilde = FC(x, w, b);
           [L, dldy] = Loss_euclidean(y_tilde, y);
           loss(iIter) = loss(iIter) + L;
           [~, dldw, dldb] = FC_backward(dldy, x, w, b, y_tilde);
           dLdw = dLdw + reshape(dldw, [size(dLdw, 1) size(dLdw, 2)]);
           dLdb = dLdb + transpose(dldb);
        end
        
        w = w - gamma/currBatchSize * dLdw;
        b = b - gamma/currBatchSize * dLdb;
        
        kBatchNo = kBatchNo + 1;
        if kBatchNo > nBatches
            kBatchNo = 1;
        end
        
    end
    f = figure(2);
    f.Name = 'SLP_linear';
    f.NumberTitle = 'off';
    plot([1:10000], loss);
end