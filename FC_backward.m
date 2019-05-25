% FC_backward   Compute partial derivatives w.r.t inputs, weights and bias
%
%   [dLdx dLdw dLdb] = FC_backward(dLdy, x, w, b, y) computes partial
%   derivatives w.r.t input x, weight w and bias b.
%     
%   Input:
%       dLdy    = 1xn loss derivative w.r.t output y
%       x       = mx1 input to the fully connected layer
%       w       = nxm weight matrix
%       b       = nx1 bias
%       y       = nx1 output
%         
%   Output:
%       dLdx    = 1xm loss derivative w.r.t x, which is propogated
%                 backwards
%       dLdw    = 1x(nxm) loss derivative w.r.t w, which is used to update
%                 weights
%       dLdb    = 1xn loss derivative w.r.t b, which is used to update
%                 the bias

function [dLdx, dLdw, dLdb] = FC_backward(dLdy, x, w, b, y)
    [n, m] = size(w);
   
    X = kron(eye(n), transpose(x)); % Block diagonal matrix of x
    dLdx = dLdy * w;
    dLdw = dLdy * X;
    
    dLdb = dLdy;
end
