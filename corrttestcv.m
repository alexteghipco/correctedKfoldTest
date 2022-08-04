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

k = c.NumTestSets;
r = size(yhat1,2);

for i = 1:k
    for j = 1:r
        d(i,j) = yhat1(test(c,i),j)-yhat2(test(c,i),j);
    end
end
df = k*r;
n = (1/(df))*sum(d(:)); % this amounts to the mean...
s = (1/(df-1))*sum((d(:)-n).^2); % variance...
t = sqrt(((1/(k*r))+(mean(c.TrainSize)/mean(c.TestSize)))*s);
%p = 1-tcdf(t,df-1);
ttop = @(t,df) (1-betainc(df/(df+t^2),df/2,0.5));
p = 1-ttop(t,df);