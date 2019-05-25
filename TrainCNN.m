% TrainCNN  Train the Convolutional Neural Network
%
%   [w_conv, b_conv, w_fc, b_fc] = TrainCNN(mini_batch_x, mini_batch_y)
%   trains the CNN using batches of training data
%
%   Input:
%       mini_batch_x    = kx1 array of cells where each cell is a batch of
%                         images
%       mini_batch_y    = kx1 array of cells where each cell is a batch of
%                         labels
%
%   Output:
%       w_conv  = f x f x c1 x c2 weight tensor
%       b_conv  = c2 x 1 bias vector
%       w_fc    = 10 x (h/2 x w/2 x 3) weight matrix for fully connected
%                 layer
%       b_fc    = 10x1 bias for fully connected layer

function [w_conv, b_conv, w_fc, b_fc] = TrainCNN(mini_batch_x,...
    mini_batch_y)
    nBatches = size(mini_batch_x, 1);
    gamma = 0.03;
    lambda = 1;
    
    
    w_conv = normrnd(0, 1, [3 3 1 3]); b_conv = zeros(3, 1);
    w_fc = normrnd(0, 1, [10 147]); b_fc = zeros(10, 1);
    [~, f, c1, c2] = size(w_conv);
    
    kBatchNo = 1;
    loss = zeros(10000, 1);
    
    for iIter = 1:10000
        if mod(iIter, 1000) == 0
            gamma = lambda * gamma;
        end
        
        dLdw_conv = zeros(3, 3, 1, 3); dLdb_conv = zeros(3, 1);
        dLdw = zeros(10, 147); dLdb = zeros(10, 1);
        
        currBatch = mini_batch_x{kBatchNo};
        currLabels = mini_batch_y{kBatchNo};
        currBatchSize = size(currBatch, 2);
        
        for jImage = 1:currBatchSize
           x = reshape(currBatch(:, jImage), [14 14 1]);
           y = currLabels(:, jImage);
           
           pred1 = Conv(x, w_conv, b_conv);
           pred2 = ReLu(pred1);
           pred3 = Pool2x2(pred2);
           pred4 = Flattening(pred3);
           pred5 = FC(pred4, w_fc, b_fc);
           
           [l, dldy] = Loss_cross_entropy_softmax(pred5, y);
           loss(iIter) = loss(iIter)+l;
           
           [dldx_fc, dldw, dldb] = FC_backward(...
               dldy, pred4, w_fc, b_fc, pred5);
           [dldx_flat] = Flattening_backward(dldx_fc, pred3, pred4);
           [dldx_pool] = Pool2x2_backward(dldx_flat, pred2, pred3);
           [dldx_relu] = ReLu_backward(dldx_pool, pred1, pred2);
           [dldw_conv, dldb_conv] = Conv_backward(dldx_relu, x, w_conv,...
               b_conv, pred1);
           
           dLdw = dLdw + reshape(dldw, [size(dLdw, 2) size(dLdw, 1)])';
           dLdb = dLdb + transpose(dldb);
           
           dLdw_conv = dLdw_conv + reshape(dldw_conv, [f f c1 c2]);
           dLdb_conv = dLdb_conv + transpose(dldb_conv);
        end
        
        w_conv = w_conv - gamma/currBatchSize*dLdw_conv;
        b_conv = b_conv - gamma/currBatchSize*dLdb_conv;
        w_fc = w_fc - gamma/currBatchSize*dLdw;
        b_fc = b_fc - gamma/currBatchSize*dLdb;
        
        kBatchNo = kBatchNo+1;
        if kBatchNo > nBatches
            kBatchNo = 1;
        end
    end
    
    f = figure(2);
    f.Name = 'CNN';
    f.NumberTitle = 'off';
    plot([1:10000], loss);
end