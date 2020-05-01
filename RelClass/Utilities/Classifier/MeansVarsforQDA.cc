/*
 * This code computes the means and variances for the local QDA classifier.
 * Input
 * X - [dim x n_t] matrix of training data (each column is a feature vector)
 * y_s - [n_t x 1] vector of training labels associated with the feature vectors in X
 * IX_d - [n_t -1 x n_t] matrix of distances.  The (i,j) entry gives the index of the
 *   j^th closest training data point to the i^th training point.
 * Num_Neighbors_d - integer that gives the number of local neighbors to use
 * Classes - [n_c x 1] - vector of possible class labels
 *
 * Output
 * means - [dim x (n_t * n_c)] matrix of local means.  The first n_t columns correspond to class 1, the next n_t columns to class 2, etc.
 * vars - [dim x (n_t *n_c)] matrix of local variances.  The first n_t columns correspond to class 1, the next n_t columns to class 2, etc.
 */

#include <string.h>
#include <math.h>
#include <mex.h>

void 
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // check number of arguments
    if (nlhs < 2) {
        mexErrMsgTxt("Too few output arguments.");
    }
    if (nlhs >= 3) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (nrhs < 5) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nrhs >= 6) {
        mexErrMsgTxt("Too many input arguments.");
    }

    // get the input arguments
    double* X = mxGetPr(prhs[0]);
    int Fea_Dim = mxGetM(prhs[0]);
	double* y_s = mxGetPr(prhs[1]);
    double* IX_d = mxGetPr(prhs[2]);
	int Num_X = mxGetM(prhs[2]);
	int Num_y_t = mxGetN(prhs[2]);
	double* Num_Neighbors_d = mxGetPr(prhs[3]);
	double* Classes = mxGetPr(prhs[4]);
	int Num_Classes = mxGetM(prhs[4]);

	int Num_Neighbors = static_cast<int>(Num_Neighbors_d[0]);

    // Set up the output arguments
	plhs[0] = mxCreateDoubleMatrix(Fea_Dim,Num_y_t*(Num_Classes),mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(Fea_Dim,Num_y_t*(Num_Classes),mxREAL);
    double* means = mxGetPr(plhs[0]);
    double* vars = mxGetPr(plhs[1]);

	//means and variances from the nearest neighbors to each point
	int class_idx = 0;
	int num_so_far = 0;
    for (int i = 0; i < Num_y_t; i++)  //loop over the columns of IX, one column represents one element of y_s
    {
		class_idx = 0;
		for (int class_now = 0; class_now < Num_Classes; class_now++)
		{
			//mean of the ith point
			num_so_far = 0;
			for (int j = 0; j < Num_X; j++)  //loop over the rows of IX, one column represents one element of X
			{
				if ( static_cast<int>(y_s[ static_cast<int>(IX_d[i*Num_X + j]) - 1 ]) == static_cast<int>(Classes[class_now]))
				{
					for (int k = 0; k < Fea_Dim; k++)
						means[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = means[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] + X[(static_cast<int>(IX_d[i*Num_X + j]) - 1)*Fea_Dim + k];
					num_so_far++;
				}
				if (num_so_far == Num_Neighbors)
					break;
			}
			for (int k = 0; k < Fea_Dim; k++)
				means[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = means[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k]/num_so_far;
			
			//variance of the ith point
			if (num_so_far > 1)
			{
				num_so_far = 0;
				for (int j = 0; j < Num_X; j++)  //loop over the rows of IX, one column represents one element of X
				{
					if ( static_cast<int>(y_s[ static_cast<int>(IX_d[i*Num_X + j]) - 1 ]) == static_cast<int>(Classes[class_now]))
					{
						for (int k = 0; k < Fea_Dim; k++)
							vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] + pow(X[(static_cast<int>(IX_d[i*Num_X + j]) - 1)*Fea_Dim + k] - means[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k],2);
						num_so_far++;
					}
					if (num_so_far == Num_Neighbors)
						break;
				}
				for (int k = 0; k < Fea_Dim; k++)
				{
					if (vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] == 0)
						vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = 1e-4;
					else
						vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k]/(num_so_far-1);
						if (Num_Neighbors < 16)
							vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k]+1e-4;
				}
			}
			else
			{
				for (int k = 0; k < Fea_Dim; k++)
					vars[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = 1;
			}
			class_idx++;
		}
	}
 
}



