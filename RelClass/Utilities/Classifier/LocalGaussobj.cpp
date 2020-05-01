/*
 * This code returns the local QDA discriminant score for the input data.
 * Input
 * X - [dim x n_t] matrix of training data (each column is a feature vector)
 * locmeans - [dim x (n_t * n_c)] matrix of local means.  The first n_t columns correspond to class 1, the next n_t columns to class 2, etc.
 * locvars - [dim x (n_t * n_c)] matrix of local variances.  The first n_t columns correspond to class 1, the next n_t columns to class 2, etc.
 * priors - [n_c x 1] vector of class priors
 * Num_Classes_d - [n_c x 1] vector of class labels
 * Classes - [n_c x 1] - vector of possible class labels
 *
 * Output
 * expscore - [n_t x n_c] matrix of QDA discriminant scores, each row corresponds to a test sample, and each column corresponds to a class
 */

#include <string.h>
#include <math.h>
#include <mex.h>

void 
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // check number of arguments
    if (nlhs < 1) {
        mexErrMsgTxt("Too few output arguments.");
    }
    if (nlhs >= 2) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (nrhs < 5) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nrhs >= 6) {
        mexErrMsgTxt("Too many input arguments.");
    }

    // get arguments
    double* X = mxGetPr(prhs[0]);
    int numtest = mxGetN(prhs[0]);
    int numfea = mxGetM(prhs[0]);
    double* locmeans = mxGetPr(prhs[1]);
    double* locvars = mxGetPr(prhs[2]);
	double* priors = mxGetPr(prhs[3]);
	double* Num_Classes_d = mxGetPr(prhs[4]);
	int Num_Classes = static_cast<int>(Num_Classes_d[0]);

   // printf("pts=%d ",N);
   // printf("classes=%d ",K);
   // printf("indim=%d ",ID);
   // printf("outdim=%d \n",OD);

    ////// set up output arguments
    plhs[0] = mxCreateDoubleMatrix(numtest,Num_Classes,mxREAL); 
	double* expscore = mxGetPr(plhs[0]);
    
	for (int i = 0; i < numtest; i++)
	{
        for (int j = 0; j < Num_Classes; j++)
		{
			expscore[j*numtest + i] = - 2*log( priors[j] );
			for (int k = 0; k < numfea; k++)
			{
				expscore[j*numtest + i] = expscore[j*numtest + i] + log( locvars[j*numfea*numtest + i*numfea + k] ) + pow( X[i*numfea + k]- locmeans[j*numfea*numtest + i*numfea + k], 2)/locvars[j*numfea*numtest + i*numfea + k];
			}
		}
	}
}



