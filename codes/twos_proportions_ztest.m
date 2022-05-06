function z_pval = twos_proportions_ztest(p1,n1,p2,n2)
% two-sample proportions Z-test,
% test whether proportion of one sample is statistically different from
% another proportion of another sample.  
% inputs:
%   p1 = porportion of sample 1
%   n1 = sample 1 size
%   p2 = proportion of sample 2
%   n2 = sample 2 size

    % calculate the combined proportion
    pc = (p1*n1+p2*n2)/(n1+n2);
    % then calculate the total SE in the combined population
    seTotal = sqrt(pc*(1-pc)*(1/n1+1/n2));
    % now we calculate the the difference in the two proportions and
    % transform to Z-statistic
    zstat = (p2-p1)/seTotal;
    % now we get our p-value
    z_pval = (1 - normcdf(abs(zstat)))*2;
 end