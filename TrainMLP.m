% TrainMLP  Train the multi layer perceptron model
%
%   [w1, b1, w2, b2] = TrainMLP(mini_batch_x, mini_batch_y) computes the
%   weights w1, w2 and bias b1, b2 by performing stochastic mini batch
%   gradient descent.
%
%   Input:
%       mini_batch_x    = kx1 array of cells where each cell is a batch of
%                         images
%       mini_batch_y    = kx1 array of cells where each cell is a batch of
%                         labels
%
%   Output:
%       w1  = 30x196 weight matrix for the first layer
%       b1  = 30x1 bias vector for the first layer
%       w2  = 10x30 weight matrix for the second layer
%       b2  = 10x1 bias vector for the second layer

function [w1, b1, w2, b2] = TrainMLP(mini_batch_x, mini_batch_y)
    nBatches = size(mini_batch_x, 1);
    gamma = 1.0;
    lambda = 0.75;
    w1 = normrnd(0, 1, [30 196]); b1 = zeros(30, 1);
    w2 = normrnd(0, 1, [10 30]); b2 = zeros(10, 1);
    kBatchNo = 1;
    loss = zeros(10000, 1);
    
    for iIter = 1:10000
        if mod(iIter, 1000) == 0
            gamma = lambda * gamma;
        end
        
        dLdw1 = zeros(30, 196); dLdb1 = zeros(30, 1);
        dLdw2 = zeros(10, 30); dLdb2 = zeros(10, 1);
        
        currBatch = mini_batch_x{kBatchNo};
        currLabels = mini_batch_y{kBatchNo};
        currBatchSize = size(currBatch, 2);
        
        for jImage = 1:currBatchSize
           x = currBatch(:, jImage); y = currLabels(:, jImage);
           
           a1 = FC(x, w1, b1);
           f1 = ReLu(a1);
           a2 = FC(f1, w2, b2);
           
           [l, dlda2] = Loss_cross_entropy_softmax(a2, y);
           loss(iIter) = loss(iIter)+l;
           
           [dldf1, dldw2, dldb2] = FC_backward(dlda2, f1, w2, b2, a2);
           dlda1 = ReLu_backward(dldf1, a1, f1);
           [~, dldw1, dldb1] = FC_backward(dlda1, x, w1, b1, a1);
           
           dLdw2 = dLdw2 + reshape(dldw2, [size(dLdw2, 1) size(dLdw2, 2)]);
           dLdb2 = dLdb2 + transpose(dldb2);
           dLdw1 = dLdw1 + reshape(dldw1, [size(dLdw1, 1) size(dLdw1, 2)]);
           dLdb1 = dLdb1 + transpose(dldb1);
        end
        
        w2 = w2 - gamma/currBatchSize*dLdw2;
        b2 = b2 - gamma/currBatchSize*dLdb2;
        w1 = w1 - gamma/currBatchSize*dLdw1;
        b1 = b1 - gamma/currBatchSize*dLdb1;
        
        kBatchNo = kBatchNo+1;
        if kBatchNo > nBatches
            kBatchNo = 1;
        end
    end
    
    f = figure(2); pause(0.1);
    f.Name = 'MLP';
    f.NumberTitle = 'off';
    plot([1:10000], loss);
end