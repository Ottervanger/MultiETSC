/*
 * This code computes the local means and local variances for local discriminative Gaussian dimensionality reduction.
 * The local means and local variances can be computed for target data from source data, if desired (see the transfer learning code)
 * or can simply be computed from the target data for the target data (in the standard LDG case).
 *
 * Input
 * X - [dim x n_s] matrix of source training data (each column is a feature vector) - in the standard LDG case, this is just the training data
 * y_t - [n_t x 1] vector of target domain training labels
 * y_s - [n_s x 1] vector of source training labels associated with the feature vectors in X  - in the standard case, this should be set equal to y_t
 * IX_d - [n_s x n_t] matrix of distances.  The (i,j) entry gives the index of the
 *   j^th closest source domain point data point to the i^th target point.  If the source and target training sets are equal, then this should be of size [n_t -1 x n_t]
 * Num_Neighbors_d - integer that gives the number of local neighbors to use
 * Num_Classes_d - integer giving the number of classes
 * Classes - [n_c x 1] - vector of possible class labels
 *
 * Output
 * means - [dim x n_t] matrix of same-class local means. 
 * vars - [dim x n_c] matrix of same-class local variances.  
 * means_disc - [dim x (n_t * n_c - 1)] matrix of different class local means.
 * vars_disc - [dim x (n_t * n_c - 1)] matrix of different class local variances.
 */

#include <string.h>
#include <math.h>
#include <mex.h>

void 
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // check number of arguments
    if (nlhs < 4) {
        mexErrMsgTxt("Too few output arguments.");
    }
    if (nlhs >= 5) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (nrhs < 7) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nrhs >= 8) {
        mexErrMsgTxt("Too many input arguments.");
    }

    // get arguments
    double* X = mxGetPr(prhs[0]);
    int Fea_Dim = mxGetM(prhs[0]);
    double* y_t = mxGetPr(prhs[1]);
    int Num_y_t = mxGetM(prhs[1]);
	double* y_s = mxGetPr(prhs[2]);
    double* IX_d = mxGetPr(prhs[3]);
	int Num_X = mxGetM(prhs[3]);
	double* Num_Neighbors_d = mxGetPr(prhs[4]);
	double* Num_Classes_d = mxGetPr(prhs[5]);
    double* Classes = mxGetPr(prhs[6]);

	int Num_Neighbors = static_cast<int>(Num_Neighbors_d[0]);
	int Num_Classes = static_cast<int>(Num_Classes_d[0]);

    // set up output arguments
    plhs[0] = mxCreateDoubleMatrix(Fea_Dim,Num_y_t,mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(Fea_Dim,Num_y_t,mxREAL); 
	plhs[2] = mxCreateDoubleMatrix(Fea_Dim,Num_y_t*(Num_Classes-1),mxREAL); 
    plhs[3] = mxCreateDoubleMatrix(Fea_Dim,Num_y_t*(Num_Classes-1),mxREAL);
    double* means = mxGetPr(plhs[0]);
    double* vars = mxGetPr(plhs[1]);
	double* means_disc = mxGetPr(plhs[2]);
	double* vars_disc = mxGetPr(plhs[3]);

    //means and variances from the nearest neighbors to each point
	int num_so_far = 0;
    for (int i = 0; i < Num_y_t; i++)  //loop over the columns of IX, one column represents one element of y
    {

		//mean of the ith point
		num_so_far = 0;
		for (int j = 0; j < Num_X; j++)  //loop over the rows of IX, one column represents one element of X
		{
			if ( static_cast<int>(y_s[ static_cast<int>(IX_d[i*Num_X + j]) - 1 ]) == static_cast<int>(y_t[i]))
			{
				//if (i == 0)
					//printf("%d ", static_cast<int>(IX_d[i*Num_X + j]));
				for (int k = 0; k < Fea_Dim; k++)
				{
					means[i*Fea_Dim + k] = means[i*Fea_Dim + k] + X[(static_cast<int>(IX_d[i*Num_X + j]) - 1)*Fea_Dim + k];
					//if (i == 0)
						//printf("%g ", means[i*Fea_Dim + k]);
				}
				num_so_far++;
			}
			if (num_so_far == Num_Neighbors)
				break;
		}
		for (int k = 0; k < Fea_Dim; k++)
			means[i*Fea_Dim + k] = means[i*Fea_Dim + k]/num_so_far;
			
		//variance of the ith point
		if (num_so_far > 1)
		{
			num_so_far = 0;
			for (int j = 0; j < Num_X; j++)  //loop over the rows of IX, one column represents one element of X
			{
				if ( static_cast<int>(y_s[ static_cast<int>(IX_d[i*Num_X + j]) - 1 ]) == static_cast<int>(y_t[i]))
				{
					for (int k = 0; k < Fea_Dim; k++)
						vars[i*Fea_Dim + k] = vars[i*Fea_Dim + k] + pow(X[(static_cast<int>(IX_d[i*Num_X + j]) - 1)*Fea_Dim + k] - means[i*Fea_Dim + k],2);
					num_so_far++;
				}
				if (num_so_far == Num_Neighbors)
					break;
			}
			for (int k = 0; k < Fea_Dim; k++)
			{
				if (vars[i*Fea_Dim + k] == 0)
					vars[i*Fea_Dim + k] = 1e-4;
				else
					vars[i*Fea_Dim + k] = vars[i*Fea_Dim + k]/(num_so_far-1);
                    if (Num_Neighbors < 16)
                        vars[i*Fea_Dim + k] = vars[i*Fea_Dim + k] + 1e-4; //Regularization
			}
		}
		else
		{
			for (int k = 0; k < Fea_Dim; k++)
					vars[i*Fea_Dim + k] = 1;
		}

    }

	//discriminative means and variances from the nearest neighbors to each point
	int class_idx = 0;
	num_so_far = 0;
    for (int i = 0; i < Num_y_t; i++)  //loop over the columns of IX, one column represents one element of y
    {
		class_idx = 0;
		for (int class_now = 0; class_now < Num_Classes; class_now++)
		{
			if ( static_cast<int>(Classes[class_now]) != static_cast<int>(y_t[i]))
			{
				//mean of the ith point
				num_so_far = 0;
				for (int j = 0; j < Num_X; j++)  //loop over the rows of IX, one column represents one element of X
				{
					if ( static_cast<int>(y_s[ static_cast<int>(IX_d[i*Num_X + j]) - 1 ]) == static_cast<int>(Classes[class_now]))
					{
						for (int k = 0; k < Fea_Dim; k++)
							means_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = means_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] + X[(static_cast<int>(IX_d[i*Num_X + j]) - 1)*Fea_Dim + k];
						num_so_far++;
					}
					if (num_so_far == Num_Neighbors)
						break;
				}
				for (int k = 0; k < Fea_Dim; k++)
					means_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = means_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k]/num_so_far;
			
				//variance of the ith point
				if (num_so_far > 1)
				{
					num_so_far = 0;
					for (int j = 0; j < Num_X; j++)  //loop over the rows of IX, one column represents one element of X
					{
						if ( static_cast<int>(y_s[ static_cast<int>(IX_d[i*Num_X + j]) - 1 ]) == static_cast<int>(Classes[class_now]))
						{
							for (int k = 0; k < Fea_Dim; k++)
								vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] + pow(X[(static_cast<int>(IX_d[i*Num_X + j]) - 1)*Fea_Dim + k] - means_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k],2);
							num_so_far++;
						}
						if (num_so_far == Num_Neighbors)
							break;
					}
					for (int k = 0; k < Fea_Dim; k++)
					{
						if (vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] == 0)
							vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = 1e-4;
						else
							vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k]/(num_so_far-1);
                            if (Num_Neighbors < 16) //Regularization
                                vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] + 1e-4;
					}
				}
				else
				{
					for (int k = 0; k < Fea_Dim; k++)
						vars_disc[class_idx*(Fea_Dim*Num_y_t)+i*Fea_Dim + k] = 1;
				}
				class_idx++;
			}

		}
	}
 
}



