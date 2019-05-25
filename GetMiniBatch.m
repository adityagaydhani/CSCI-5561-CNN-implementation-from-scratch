% GetMiniBatch  Segment data into mini batches
%
%   [mini_batch_x, mini_batch_y] = GetMiniBatch(im_train, label_train,
%   batch_size) divides the training data into small batches.
%
%   Input:
%       im_train    = 196 x n_train vectorized training images
%       label_train = 1 x n_train corresponding labels for im_train
%       batch_size  = size of mini batch for SGD
%
%   Output:
%       mini_batch_x    = Cells containing batches for images. Each
%       batch is of dim (196 x batch_size)
%       mini_batch_y    = Cells containing batches for labels. Each
%       batch is of dim (10 * batch_size)

function [mini_batch_x, mini_batch_y] = GetMiniBatch(im_train,...
    label_train, batch_size)

    nTrain = size(label_train, 2);
    nCells = ceil(nTrain/batch_size);
    mini_batch_x = cell(nCells, 1);
    mini_batch_y = cell(nCells, 1); 
    
    % One-hot encoding
    I = eye(10);
    label_train_enc = I(:, label_train+1); % Dim = 10 x n_train
    
    % Shuffle
    perm = randperm(nTrain);
    im_train = im_train(:, perm);
    label_train_enc = label_train_enc(:, perm);
    
    idx = 0;
    for iCellNo = 1:nCells
        if idx + 30 > nTrain
            mini_batch_x{iCellNo} = im_train(:, idx+1:end);
            mini_batch_y{iCellNo} = label_train_enc(:, idx+1:end);
        else
            mini_batch_x{iCellNo} = im_train(:, idx+1:idx+batch_size);
            mini_batch_y{iCellNo} = label_train_enc(...
                :, idx+1:idx+batch_size);
        end
        idx = idx + batch_size;
    end
    
end