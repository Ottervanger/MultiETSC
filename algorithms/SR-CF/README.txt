
########################################################

Authors: Usue Mori, Alexander Mendiburu, Sanjoy Dasgupta and Jose A. Lozano

Contact email: usue.mori@ehu.eus

##########################################################
###################INITIAL COMMENTS#######################
##########################################################

The implementation of the GP classification models in are largely based on the vbmp package of R: 

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


These modified and additional files are available together with the code. 

The rest of the files in the vbmp package are also included, in order to be 
able to call to internal functions from outside the package. However, these functions have not been modified 
and are identical to those extracted from the vbmp package.

####################REQUIREMENTS###################
###################################################

Next we describe how to use the available code:

REQUIREMENTS

-R must be installed (preferably the last version)

-The TSdist and GA packages from CRAN must be installed.

-The time series database of interest must be saved in folder databases as a .RData object. 
	It is defined as a list of two elements: 

1.-Database[[1]]: The values of the time series are saved in a dataframe. Each line corresponds to one time serie. 
	The last column is a variable that takes a value of 0 if the series is in the training set and a value of 
	1 if it is in the testing set.

2.-Database[[2]]: The class values of the series are saved.


####################EXECUTION###################
###################################################

1.- Obtain probabilities for training series: 

	EXECUTE: ./Code/Train.probabilities/posterior.probabilities.R
	NOTE1: The working directory must be Train.probabilities
	NOTE2: This process may be long, and can be paralelized. 
	NOTE3: Parameters, can be modified, they are defined at the beginning of the source files.
	 

2.- Obtain probabilities for testing series: 

	Execute ./Code/Test.probabilities/posterior.probabilities.R
	NOTE1: The working directory must be Test.probabilities
	NOTE2: This process may be long, and can be paralelized. 
	NOTE3: Parameters, can be modified, they are defined at the beginning of the source files.


3.- Optimization: 

	For SR1-CF1 execute:
		./Code./Optimization/SR1-CF1/optimization.simple.R

		NOTE1: The working directory must be Optimization/SR1-CF1 
		NOTE2: This process may be long, and can be paralelized. 
		NOTE3: Parameters, must be modified, they are defined at the beginning of the source files.

	For other combinations:
		./Code/Optimization/Others/optimization.simple.R
		
		NOTE1: The working directory must be Optimization/Others 
		NOTE2: This process may be long, and can be paralelized. 
		NOTE3: Parameters, must be modified, they are defined at the beginning of the source files.

4.- Prediction: 

	For SR1-CF1 execute:
		./Code/Prediction/SR1-CF1/prediction.R
		NOTE1: The working directory must be Prediction/SR1-CF1 
		NOTE2: Parameters, must be modified, they are defined at the beginning of the source files.

	For other combinations:
		./Prediction/Others/prediction.R
		NOTE1: The working directory must be Prediction/Others 
		NOTE2: Parameters, must be modified, they are defined at the beginning of the source files.