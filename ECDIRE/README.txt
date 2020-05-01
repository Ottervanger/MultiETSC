########################################################
#################AUTHORS AND CONTACT INFORMATION########
########################################################

Authors: Usue Mori, Alexander Mendiburu and Jose A. Lozano

Contact email: usue.mori@ehu.es

##########################################################
###################INITIAL COMMENTS#######################
##########################################################

The implementation of the GP classification models in the RelGP method is largely based on the vbmp package of R: 

Lama N and Girolami M. vbmp: Variational Bayesian Multinomial Probit Regression. R package version 1.34.0, 
http://bioinformatics.oxfordjournals.org/cgi/content/short/btm535v1. 

This package is only directed to multinomial classification and, in order to enable binary classification, 
the following functions have been modified:

-vbmp.R
-predClass.R
-predictCPP.R

Additionally, some functions have been added for this same purpose:

-genCPP.binary.R
-tmean.binary.R


This has been done following the formulae shown in the following paper:

Girolami, M., Rogers, S., 2006. Variational Bayesian Multinomial Probit Regression with Gaussian Process Priors. 
Neural Computation 18, 1790–1817.


These modified and additional files are available in the vbmp-modified/modified folder. 

The rest of the files in the vbmp package are included in the vbmp-modified/others folder, in order to be 
able to call to internal functions from outside the package. However, these functions have not been modified 
and are identical to those extracted from the vbmp package.

###################################################################################APPLICATION OF THE CODE#################
##############################################################

Next we describe how to use the available code and apply the RelGP method:

REQUIREMENTS

-R must be installed (preferably the last version)

-The TSdist package from CRAN must be installed (version 1.2 downloadable from 
http://cran.r-project.org/src/contrib/Archive/TSdist).

-The time series database of interest must be saved in folder databases as a .RData object. 
	It is defined as a list of two elements: 

1.-Database[[1]]: The values of the time series are saved in a dataframe. Each line corresponds to one time serie. 
	The last column is a variable that takes a value of 0 if the series is in the training set and a value of 
	1 if it is in the testing set.

2.-Database[[2]]: The class values of the series are saved.


INPUT INFORMATION THAT MUST BE MODIFIED BY THE USER

In order to make the code work, some parameter must be modified in the main relGP.R function:

-path: the path to the ./RelGP folder.

-databasename: The name of the database surrounded by ""-s.

-cvprocess: logical value that indicates if the cross validation process in step 2b of Section 4.1 must 
be performed or not. For the UCR-4 database (CBF), the results from the cross validation are already 
included in the results/accuracies and results/probabilities folders, so crossvalidation=FALSE should be chosen. 
For other databases, crossvalidation=TRUE. However, it these latter cases, it highly recommended that the 
crossvalidation.R function is executed separately in a paralel environment because the computation may be 
very time consuming. 

-estimatehyp: logical parameter which indicates if the hyperparameter of the kernel function used when 
training the GP models must be estimated or is fixed to a vector of 1-s.

-accuracythresholds: the accuracy that is desired as a percentage of the accuracy using the whole series. (set to 100%)

-distance: A number that indicates which distance to use. At this time, only Euclidean distance (=1) is supported.

-kernel: numeric value that defines kernel function to be used in the GP models. 
Takes values of 1 for inner product kernel, 2 for Gaussian kernel, 3 for Cauchy kernel and 4 for Laplace kernel.


EXECUTION OF THE CODE

Once this is done, the relGP method can be executed by sourcing the code/full_method.R file. 
The results will be saved in the results/finalresults file.


