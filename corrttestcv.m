function [t,p] = corrttestcv(yhat1,yhat2,c)
% [t,p] = corrttestML(yhat1,yhat2,c)
%
% Corrected repeated k-fold cross-validation test. c is the matlab
% cvpartition object you used to generate predictions in yhat1 and yhat2
% using two different models. Rows of yhat1 and yhat2 are the sample
% predictions, and columns are the repeats. We convert t-values to p-values
% using the statistics toolbox so make sure you have that installed.
%
% It is well documented that t-tests have a high type 1 error and low
% replicability when comparing model performance across repeated k-fold cv.
% Various proposals have tired to correct t-tests for type 1 errors. The
% corrected repeated k-fold cv test implemented here does this while also
% improving replicability.
%
% See: Bouckaert, R. R., & Frank, E. (2004, May). Evaluating the
% replicability of significance tests for comparing learning algorithms. In
% Pacific-Asia conference on knowledge discovery and data mining (pp.
% 3-12). Springer, Berlin, Heidelberg.
%
% alex.teghipco@sc.edu

k = c{1}.NumTestSets;
r = size(yhat1,2);
d = zeros(k, r); % Preallocate matrix for differences

for i = 1:k
    for j = 1:r
        test_indices = test(c{j}, i); % Indices of test samples for fold i
        differences = yhat1(test_indices, j) - yhat2(test_indices, j); % Differences for all samples in the fold
        d(i, j) = mean(differences); % Mean difference for fold i and run j
    end
end

df = k*r - 1; % Degrees of freedom
n = mean(d(:)); % Mean of differences
s = var(d(:), 1); % Variance of differences, 1 for N normalization

% Corrected test statistic
corrected_variance = ((1/(k*r)) + (mean(c{j}.TestSize)/mean(c{j}.TrainSize))) * s;
t = n / sqrt(corrected_variance);

% P-value calculation
p = 1 - tcdf(t, df);