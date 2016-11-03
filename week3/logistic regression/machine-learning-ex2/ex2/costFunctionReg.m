function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

d = 0;
penal_cost = 0;

for i=2:length(theta),
  penal_cost = penal_cost + (theta(i)^2);
end
penal_cost = (lambda/(2*m)) * penal_cost;

for i=1:m,
	d = d + ((y(i,:)*log(sigmoid(theta'*X(i,:)'))) + ((1-y(i,:))*log(1-sigmoid(theta'*X(i,:)'))));
end

J = (-(1/m))*d;

J = J + penal_cost;

g = zeros(size(theta));

for j=1:size(theta),
  if (j == 1)
    for i=1:m,
	    g(j) = g(j) + (sigmoid(theta'*X(i,:)') - y(i,:)) * X(i,j);
    end
  else
    for i=1:m,
	    g(j) = g(j) + (sigmoid(theta'*X(i,:)') - y(i,:)) * X(i,j);
    end
    g(j) = g(j) + ((lambda)* theta(j));
  endif
end

grad = (1/m)*g;




% =============================================================

end
