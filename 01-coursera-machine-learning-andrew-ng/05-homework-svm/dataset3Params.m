function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error_results = zeros(3, 1);

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

for i = 1:length(C_values),
	current_C = C_values(i);

	for j = 1:length(sigma_values),
		current_sigma = sigma_values(j);

		model= svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma)); 
		predictions = svmPredict(model, Xval);
		error_value = mean(double(predictions ~= yval));

		% append new column to the error_results matrix with the current calculations
		error_results = [error_results [current_C; current_sigma; error_value]];

	end
end

% get rid of the first 'dummy' column with zeros
error_results = error_results(:, 2:end);

[min_error, min_error_index] = min(error_results(3, :));

C = error_results(1, min_error_index);
sigma = error_results(2, min_error_index);






% =========================================================================

end
