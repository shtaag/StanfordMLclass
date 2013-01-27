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

theta_from2 = theta(2:size(theta,1),:);
theta_zero = theta(1,:);
theta_reg = [0; theta_from2];
X_from2 = X(:,2:size(X,2));

predictions = sigmoid(X * theta);
predictions_from2 = sigmoid(X_from2 * theta_from2);
predictions_zero = sigmoid(X(:,1) * theta_zero);
errors = -y .* log(predictions) - (1 - y) .* log( 1 - predictions);
regulateParam = lambda / (2*m) * (theta_from2' * theta_from2);

J = 1/m * sum(errors) + regulateParam ;
grad = (((predictions - y)' * X)' + lambda*theta_reg) / m;
%grad_zero = ((predictions_zero - y)' * X(:,1))' / m;
%grad_from2 = (((predictions_from2 - y)' * X_from2)' + lambda.*theta_from2)  / m;
%grad = [grad_zero ; grad_from2];


% =============================================================

end
