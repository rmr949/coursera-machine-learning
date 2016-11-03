function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    temp_theta = theta;
    
    d = 0;
    for i=1:m,
	    d = d + ((theta' * X(i,:)') - y(i,:));
    end
    theta0_cost = (1/m)*d;
    
    temp_theta(1,1) = theta(1,1) - (alpha * theta0_cost);
    
    d = 0;
    for i=1:m,
	    d = d + ((theta' * X(i,:)') - y(i,:)) * X(i,2);
    end
    theta1_cost = (1/m)*d;
       
    temp_theta(2,1) = theta(2,1) - (alpha * theta1_cost);

    theta(1,1) = temp_theta(1,1);
    theta(2,1) = temp_theta(2,1);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
disp(J_history);
end
