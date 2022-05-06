# BCI_V1
Codes used in the V1 BCI manuscript
Includes code used to perform k-nearest neighbor (KNN) decoding, representational-drift decoding, permutation test for testing the 2D Gaussian fit, and 2-sample proportions z-test.

--List of standard Matlab statistical test functions used within the paper--

Linear regression:
  [b,bint,r,rint,stats] = regress(y,X);

Two-sample K-S test:
  [h,p] = kstest2(x,y);
  
One-sample K-S test (K-S1): 
  [h,p] = kstest(x);
  
Wilcoxon rank-sum test:
  [p,h] = ranksum(x,y);
  
Wilcoxon signed-rank test:
  [p,h] = signrank(x);
  
One-way ANOVA:
  [p,tbl,stats] = anova1(y,group);
