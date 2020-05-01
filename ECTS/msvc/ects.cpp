// mex file to do IndexBuilding and hierarchical NN classification

/* This next bit gets around yvals.h conflict of char16_t between 
 Visual C++ 2010's yvals.h and MATLAB's mex.h on Windows*/
#ifdef _MSC_VER
#include <yvals.h>
#if (_MSC_VER >= 1600)
#define __STDC_UTF_16__
#endif
#endif

#include "mex.h" // for Matlab
#include <iterator>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <string>
#include <stdlib.h> // for exit
#include "find.h"
#include "Euclidean.h"
#include "minValue.h"
#include "math.h"

// global variables that used to be in DataSetInformation.h
int DIMENSION; // length of time series
int ROWTRAINING; // size of training data
int ROWTESTING; // size of testing data
int NofClasses;
int *classLabel;

double **training; // training dataset: training[ROWTRAINING][DIMENSION];
double *labelTraining; // training labels: labelTraining[ROWTRAINING]={0};
double **testing; // test dataset: testing [ROWTESTING][DIMENSION]={0};
double *labelTesting; // test labels: labelTesting[ROWTESTING]={0};
int **trainingIndex; // 1NN for each space, no ranking tie: TrainingIndex[ROWTRAINING][DIMENSION]={0};
double **distanceArray; // the pairwise distance array of full length: distanceArray[ROWTRAINING][ROWTRAINING]={0};
double **minDistance; // minDistance[ROWTRAINING][DIMENSION]={0};

// Algorithm parameters: minimal support
double minimalSupport = 0;
int strictversion = 1; // 1 strict version , 0 loose version

double *predictedLabel; // predicted label by the classifier: predictedLabel[ROWTESTING]
int *predictedLength;// predicted length by the classifier: predictedLength[ROWTESTING]
int *classDistri; // class distribution: classDistri[NofClasses]
double *classSupport;// classSupport[NofClasses]={0};
int *FullLengthClassificationStatus; // FullLengthClassificationStatus[ROWTRAINING];// 0 can not be classified correctly, 1 can be classified correctly
int *PredictionPrefix; // PredictionPrefix[ROWTRAINING];
std::vector<std::set<int, std::less<int> > > NNSetList;// store the MNCS;
std::vector<int> NNSetListLength;
std::vector<int> NNSetListClassMap;// store the class label of the MNCS
int correct = 0; // the unlabeled sequence which is correctly classified.

// functions in the same file
void hierarchical();

//void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[], int len  );
void getClassDis();
//void LoadDisArray(const char * fileName, double Data[ROWTRAINING][ROWTRAINING]  );
//void LoadTrIndex(const char * fileName, int Data[ROWTRAINING][DIMENSION]  );
std::set<int, std::less<int> > SetRNN(int l, std::set<int, std::less<int> >& s);
std::set<int, std::less<int> > SetNN(int l, std::set<int, std::less<int> >& s);
int NNConsistent(int l, std::set<int, std::less<int> >& s);
//void testUnion();
int getMPL(std::set<int, std::less<int> >& s); // loose version
int getMPL(std::set<int, std::less<int> >& s, int la, int lb); // loose version
int getMPL2(std::set<int, std::less<int> >& setA,
        std::set<int, std::less<int> >& setB, int la, int lb);
//void getMNCS(int onenode,double label, std::set<int,std::less<int> >& PreMNCS);
int updateMPL(std::set<int, std::less<int> > s);
//bool sharePrefix(   std::set<int, std::less<int> >& s1, std::set<int, std::less<int> >& s2, std::set<int, std::less<int> >& s3, int level        );
int findMin(int data[], int len);
void classification();
void fullClassification(double*);
double mean(int data[], int len);
int getLabelIndex(int label, int* classLabel, int NofClasses);
double getSetDistance(std::set<int, std::less<int> > A,
        std::set<int, std::less<int> > B);
std::vector<int> RankingArray(
        std::vector<std::set<int, std::less<int> > > & List);
int
GetMN(std::vector<std::set<int, std::less<int> > > & List, int ClusterIndex); // get the NN of cluster, if there is no MN, return -1, others , return the value;

int getMPLStrict(std::set<int, std::less<int> >& s, int la, int lb);
int getMPLStrict(std::set<int, std::less<int> >& s);
int updateMPLStrict(std::set<int, std::less<int> > s);

int getMPLTest(std::set<int, std::less<int> >& s, int la, int lb);
int getMPLTest(std::set<int, std::less<int> >& s);
int updateMPLTest(std::set<int, std::less<int> > s);

void BuildIndex();

template<class T>
T** newArray(int rows, int cols) {
    /*
     Trick: call "new" only twice, and allow 2D array to be contiguous in memory.
     */
    T ** array = new T*[rows];
    array[0] = new T[rows * cols];
    for (int m = 1; m < rows; m++)
        array[m] = array[0] + m * cols;

    return array;
}

template<class T>
void deleteArray(T** array) {
    delete[] array[0];
    delete[] array;
}

int doubleToInt(double value){
    return (int)(value + (value < 0 ? -0.5 : 0.5));
}

// MATLAB calling syntax:
//   [labelTesting,predictedLengthTesting,fullTestLabel] = ects( training, labelTraining, testing, classLabels );

// the main mex function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    double *tmp;
    int m, n;

    /* check for proper number of arguments */
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("ects:nrhs", "Four inputs required.");
    }
    if (nlhs > 3) {
        mexErrMsgIdAndTxt("ects:nlhs", "Too many outputs.");
    }

    DIMENSION = (int)mxGetN(prhs[0]);
    ROWTRAINING = (int)mxGetM(prhs[0]);
    ROWTESTING = (int)mxGetM(prhs[2]);
    NofClasses = (int)mxGetNumberOfElements(prhs[3]);

    if (mxGetM(prhs[1]) != ROWTRAINING) {
        // # of training labels should match # of training data points
        mexErrMsgIdAndTxt("ects:row",
                "training labels rows mismatch training data.");
    }

    if (mxGetN(prhs[2]) != DIMENSION) {
        // # test and training dimension should match
        mexErrMsgIdAndTxt("ects:dim",
                "training and test data dimension mismatch.");
    }

    // Allocate arrays
    training = newArray<double> (ROWTRAINING, DIMENSION);
    testing = newArray<double> (ROWTESTING, DIMENSION);
    FullLengthClassificationStatus = new int[ROWTRAINING]();
    PredictionPrefix = new int[ROWTRAINING]();

    classLabel = new int[NofClasses];
    classDistri = new int[NofClasses]();
    classSupport = new double[NofClasses]();
    predictedLength = new int[ROWTESTING]();

    trainingIndex = newArray<int> (ROWTRAINING, DIMENSION);
    distanceArray = newArray<double> (ROWTRAINING, ROWTRAINING);
    minDistance = newArray<double> (ROWTRAINING, DIMENSION);

    // 1st arg: Copy training matrix
    tmp = (double *) mxGetPr(prhs[0]);
    for (n = 0; n < DIMENSION; n++) {
        for (m = 0; m < ROWTRAINING; m++) {
            training[m][n] = tmp[m + n * ROWTRAINING];
        }
    }

    // 2nd arg: Set training labels
    labelTraining = (double *) mxGetPr(prhs[1]);

    // 3rd arg: Copy testing matrix
    tmp = (double *) mxGetPr(prhs[2]);
    for (n = 0; n < DIMENSION; n++) {
        for (m = 0; m < ROWTESTING; m++) {
            testing[m][n] = tmp[m + n * ROWTESTING];
        }
    }

    // 4th arg: Copy class ids
    tmp = (double *) mxGetPr(prhs[3]);
    for (n = 0; n < NofClasses; n++) {
        // double to int conversion
        classLabel[n] = doubleToInt(tmp[n]);
    }

    // Allocate data for output labels
    plhs[0] = mxCreateDoubleMatrix(ROWTESTING, 1, mxREAL);
    predictedLabel = (double *) mxGetPr(plhs[0]);

    // Run Algorithm
    BuildIndex(); // build TrainingIndex and distanceArray

    hierarchical();

    classification();

    // Set output
    if (nlhs >= 2) {
        // Copy predictedLength to output 2 if it exists
        plhs[1] = mxCreateDoubleMatrix(ROWTESTING, 1, mxREAL);
        double*len = (double *) mxGetPr(plhs[1]);
        for (n = 0; n < ROWTESTING; ++n) {
            len[n] = predictedLength[n];
        }
    }
    // Set output for full labels
    if (nlhs >= 3) {
        // Copy predictedLength to output 2 if it exists
        plhs[2] = mxCreateDoubleMatrix(ROWTESTING, DIMENSION, mxREAL);
        double* label = (double *) mxGetPr(plhs[2]);
        fullClassification(label);
    }

    deleteArray<int> (trainingIndex);
    deleteArray<double> (distanceArray);
    deleteArray<double> (minDistance);
    deleteArray<double> (training);
    deleteArray<double> (testing);
    delete[] FullLengthClassificationStatus;
    delete[] PredictionPrefix;
    delete[] predictedLength;
    delete[] classLabel;
    delete[] classDistri;
    delete[] classSupport;
}

void BuildIndex() {
    int LargeNumber = 1000000;
    for (int row = 0; row < ROWTRAINING; row++) {
        for (int col = 0; col < DIMENSION; col++) {
            minDistance[row][col] = LargeNumber;
        }
    }

    // initial the distance between itself as -1;
    for (int row = 0; row < ROWTRAINING; row++) {
        distanceArray[row][row] = -1;
    }

    for (int i = 0; i < ROWTRAINING; i++) {
        for (int j = i + 1; j < ROWTRAINING; j++) {
            double prefixEuclidean = 0;
            for (int l = 0; l < DIMENSION; l++) {
                prefixEuclidean = prefixEuclidean + (training[i][l]
                        - training[j][l]) * (training[i][l] - training[j][l]);
                if (prefixEuclidean < minDistance[i][l]) {
                    minDistance[i][l] = prefixEuclidean;
                    trainingIndex[i][l] = j;
                }

                if (prefixEuclidean < minDistance[j][l]) {
                    minDistance[j][l] = prefixEuclidean;
                    trainingIndex[j][l] = i;
                }

                if (l == DIMENSION - 1) {
                    distanceArray[i][j] = sqrt(prefixEuclidean);
                    distanceArray[j][i] = distanceArray[i][j];
                }
            } // end of for l
        } // end of for j
    } // end of for i
}// end of building index


void hierarchical() {

    getClassDis();

    for (int i = 0; i < ROWTRAINING; i++) {

        int NNofi = trainingIndex[i][DIMENSION - 1];

        if (labelTraining[i] == labelTraining[NNofi]) {
            FullLengthClassificationStatus[i] = 1;
        }
    }

    // Initialize the prediction prefix
    for (int i = 0; i < ROWTRAINING; i++) {
        PredictionPrefix[i] = DIMENSION;

    }

    // simple RNN method
    for (int i = 0; i < ROWTRAINING; i++) {

        std::set<int, std::less<int> > s;
        s.insert(i);

        if (strictversion == 0) {
            PredictionPrefix[i] = getMPL(s);
        } else {

            PredictionPrefix[i] = getMPLStrict(s);

        }

        // PredictionPrefix[i]=getMPLTest(s);

    }

    int latticeLevel = 1;
    //cout<<"bi-roots level\n";

    std::vector<std::set<int, std::less<int> > > biRoots; // is a vector of sets
    std::vector<int> biRootsLength;// the length of the node
    std::vector<double> SetLabel;// -,1,2,3,4,5,6,   0 means no label
    // initial the length to L
    std::vector<int> CurrentNew;
    std::vector<int> NextNew;

    bool *UsedInBiRoots = new bool[ROWTRAINING]; // 0 not used, 1 used
    for (int i = 0; i < ROWTRAINING; i++)
        UsedInBiRoots[i] = 0;

    for (int i = 0; i < ROWTRAINING; i++) {
        if (UsedInBiRoots[i] == 0) {
            int NNofi = trainingIndex[i][DIMENSION - 1];
            // get bi-root
            if (trainingIndex[NNofi][DIMENSION - 1] == i) {
                std::set<int, std::less<int> > temp;
                temp.insert(i);
                temp.insert(NNofi);

                if (labelTraining[i] == labelTraining[NNofi])

                {
                    int tempL;
                    if (strictversion == 0) {
                        tempL = updateMPL(temp);
                    } else {
                        tempL = updateMPLStrict(temp);

                    }

                    //  tempL=updateMPLTest(temp);

                    biRootsLength.push_back(tempL);
                    SetLabel.push_back(labelTraining[NNofi]);
                    biRoots.push_back(temp);
                    CurrentNew.push_back((int)biRoots.size() - 1);

                } else // unpure group
                {

                    biRootsLength.push_back(0);

                    SetLabel.push_back(0);
                    biRoots.push_back(temp);
                    CurrentNew.push_back((int)biRoots.size() - 1);

                }

                UsedInBiRoots[i] = 1;
                UsedInBiRoots[NNofi] = 1;
            } else // single element set
            {
                std::set<int, std::less<int> > temp;
                temp.insert(i);
                biRootsLength.push_back(PredictionPrefix[i]);
                SetLabel.push_back(labelTraining[i]);
                biRoots.push_back(temp);
            }
        }

    } // end for

    delete[] UsedInBiRoots;

    // Heuristic algorithm
    latticeLevel = 2;
    int mergedPair = 0;

    std::vector<std::set<int, std::less<int> > > NNSetListCurrent = biRoots;
    std::vector<int> NNSetListCurrentLength = biRootsLength;
    std::vector<double> NNSetListCurrentLabel = SetLabel;

    std::vector<std::set<int, std::less<int> > > NNSetListNext;
    std::vector<int> NNSetListNextLength;
    std::vector<double> NNSetListNextLabel;

    //
    while (NNSetListCurrent.size() > 1) {
        //cout<<latticeLevel<<endl;
        //cout<<"number "<<NNSetListCurrent.size()<<endl;
        mergedPair = 0;
        NNSetListNext.clear();
        NNSetListNextLength.clear();
        NNSetListNextLabel.clear();
        NextNew.clear();

        std::vector<int> status;
        for (unsigned int i = 0; i < NNSetListCurrent.size(); i++) {
            status.push_back(0);
        }

        // std::vector<int> CurrentRankingArray= RankingArray(NNSetListCurrent);

        for (unsigned int i = 0; i < CurrentNew.size(); i++) {
            int tempi = CurrentNew[i];

            if (status[tempi] == 0) {
                std::vector<int> pair;
                int MNofi = GetMN(NNSetListCurrent, tempi); // get the NN of cluster, if there is no MN, return -1, others , return the value
                if (MNofi != -1) {
                    pair.push_back(tempi);
                    pair.push_back(MNofi);

                    // test if they are of the same label
                    double label1 = NNSetListCurrentLabel[pair[0]];
                    double label2 = NNSetListCurrentLabel[pair[1]];

                    std::set<int, std::less<int> > tempSet;
                    std::set<int, std::less<int> > setA =
                            NNSetListCurrent[pair[0]];
                    std::set<int, std::less<int> > setB =
                            NNSetListCurrent[pair[1]];

                    // merge two set
                    std::set<int, std::less<int> >::iterator a;
                    for (a = setA.begin(); a != setA.end(); a++) {
                        tempSet.insert(*a);
                    }

                    std::set<int, std::less<int> >::iterator b;
                    for (b = setB.begin(); b != setB.end(); b++) {
                        tempSet.insert(*b);
                    }

                    if (label1 != 0 && label2 != 0 && label1 == label2) {
                        int tempLength;

                        if (setA.size() > 1 && setB.size() > 1) {
                            if (strictversion == 0) {
                                tempLength = getMPL(tempSet,
                                        NNSetListCurrentLength[pair[0]],
                                        NNSetListCurrentLength[pair[1]]);
                            } else {
                                tempLength = getMPLStrict(tempSet,
                                        NNSetListCurrentLength[pair[0]],
                                        NNSetListCurrentLength[pair[1]]);
                            }
                            // tempLength=getMPLTest(tempSet,NNSetListCurrentLength[pair[0]],NNSetListCurrentLength[pair[1]]);
                        } else {
                            if (strictversion == 0) {
                                tempLength = getMPL(tempSet);
                            } else {
                                tempLength = getMPLStrict(tempSet);
                            }
                        }
                        // update the length

                        std::set<int, std::less<int> >::iterator jj;
                        for (jj = tempSet.begin(); jj != tempSet.end(); jj++) {
                            if (PredictionPrefix[*jj] > tempLength) {
                                PredictionPrefix[*jj] = tempLength;
                            }
                        }

                        NNSetListNextLength.push_back(tempLength);
                        NNSetListNext.push_back(tempSet);
                        NextNew.push_back((int)NNSetListNext.size() - 1);

                        NNSetListNextLabel.push_back(label1);
                        mergedPair++;
                        status[pair[0]] = 1;
                        status[pair[1]] = 1;
                    } else { // the pair is not pure
                        NNSetListNextLength.push_back(0);
                        NNSetListNext.push_back(tempSet);
                        NextNew.push_back((int)NNSetListNext.size() - 1);

                        NNSetListNextLabel.push_back(0);
                        status[pair[0]] = 1;
                        status[pair[1]] = 1;
                    }
                }
            }// end if

        }// end for

        if (mergedPair > 0) {
            //cout<<"Merged pair"<<mergedPair<<endl;
            for (unsigned int k = 0; k < NNSetListCurrent.size(); k++) {
                if (status[k] == 0) {
                    NNSetListNext.push_back(NNSetListCurrent[k]);
                    NNSetListNextLength.push_back(NNSetListCurrentLength[k]);
                    NNSetListNextLabel.push_back(NNSetListCurrentLabel[k]);
                }
            }// end for

            NNSetListCurrent.clear();
            for (unsigned int kk = 0; kk < NNSetListNext.size(); kk++) {
                NNSetListCurrent.push_back(NNSetListNext[kk]);
            }
            NNSetListNext.clear();
            NNSetListCurrentLength.clear();
            for (unsigned int kk = 0; kk < NNSetListNextLength.size(); kk++) {
                NNSetListCurrentLength.push_back(NNSetListNextLength[kk]);
            }
            NNSetListNextLength.clear();

            NNSetListCurrentLabel.clear();
            for (unsigned int kk = 0; kk < NNSetListNextLabel.size(); kk++) {
                NNSetListCurrentLabel.push_back(NNSetListNextLabel[kk]);
            }
            NNSetListNextLabel.clear();

            // update CurrentNew and NextNew
            CurrentNew.clear();
            for (unsigned int kk = 0; kk < NextNew.size(); kk++) {
                CurrentNew.push_back(NextNew[kk]);
            }
            NextNew.clear();

            latticeLevel++;
        } else {
            break;
        }
    }// end while
    // end of hueristic algorithm

    classification();
} // end hierarchical main function


void getClassDis() {
    for (int i = 0; i < NofClasses; i++) {
        classDistri[i] = find(labelTraining, classLabel[i], ROWTRAINING);
        //      cout<<"\n"<<"Class "<<i<<": "<< classDistri[i]<<"\n";
        classSupport[i] = classDistri[i] * minimalSupport;
    }
}

std::set<int, std::less<int> > SetRNN(int l, std::set<int, std::less<int> >& s) // find a set's RNN on prefix l
{
    std::set<int, std::less<int> > index;

    std::set<int, std::less<int> >::iterator i;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        for (int j = 0; j < ROWTRAINING; j++) {
            if (trainingIndex[j][l - 1] == element && s.count(j) == 0) {
                index.insert(j);
            }
        }
    }
    return index;
}

std::set<int, std::less<int> > SetNN(int l, std::set<int, std::less<int> >& s) {
    std::set<int, std::less<int> > index;

    std::set<int, std::less<int> >::iterator i;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        int NNofelement = trainingIndex[element][l - 1];
        index.insert(NNofelement);
    }
    return index;
}
int NNConsistent(int l, std::set<int, std::less<int> >& s)// return 1, if it is NN consistent, return 0, if not
{
    std::set<int, std::less<int> >::iterator i;
    int consistent = 1;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        int NNofelement = trainingIndex[element][l - 1]; // get the NN

        if (s.count(NNofelement) == 0) {
            consistent = 0;
            break;
        }
    }
    return consistent;
}

int getMPL(std::set<int, std::less<int> >& s) {

    int MPL = DIMENSION;

    std::set<int, std::less<int> >::iterator i;
    if (s.size() == 1) // simple RNN Method
    {
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int, std::less<int> > LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;

            if (FullLengthClassificationStatus[element] == 1) {
                LastUsefulRNN.insert(element);
            }
        }
        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastUsefulRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastUsefulRNN.size() > 0) {
                std::set<int, std::less<int> > PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {
                    PreviousRNN = SetRNN(le, s);
                    std::set<int, std::less<int> > PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;

                        if (FullLengthClassificationStatus[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }
                    if (LastUsefulRNN == PreviousUsefulRNN) {
                        // CurrentUsefulRNN=PreviousUsefulRNN;
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of for
            } else {
                MPL = DIMENSION;
            }
        } else {
            MPL = DIMENSION;
        }
    } else { // super-sequence Method
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int, std::less<int> > LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;
            if (FullLengthClassificationStatus[element] == 1) {
                LastUsefulRNN.insert(element);
            }
        }
        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastUsefulRNN.size();

        if (Support >= classSupport[labelIndex]) {
            std::set<int, std::less<int> > PreviousRNN;
            for (int le = DIMENSION - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                } else {
                    PreviousRNN = SetRNN(le, s);
                    std::set<int, std::less<int> > PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;

                        if (FullLengthClassificationStatus[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }

                    if (LastUsefulRNN == PreviousUsefulRNN) {
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of else
            }// end of for
        } else {
            MPL = DIMENSION;
        }
    }
    return MPL;
}

int getMPLStrict(std::set<int, std::less<int> >& s) {
    int MPL = DIMENSION;
    std::set<int, std::less<int> >::iterator i;
    if (s.size() == 1) // simple RNN Method
    {
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int, std::less<int> > LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;

            if (FullLengthClassificationStatus[element] == 1) {
                LastUsefulRNN.insert(element);
            }
        }
        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastUsefulRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastUsefulRNN.size() > 0) {
                std::set<int, std::less<int> > PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {
                    PreviousRNN = SetRNN(le, s);
                    std::set<int, std::less<int> > PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;

                        if (FullLengthClassificationStatus[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }
                    if (LastUsefulRNN == PreviousUsefulRNN) {
                        // CurrentUsefulRNN=PreviousUsefulRNN;
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;

                    }
                }// end of for
            } else {
                MPL = DIMENSION;
            }
        } else {
            MPL = DIMENSION;
        }
    } else // super-sequence Method
    {
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);
        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastRNN.size();

        if (Support >= classSupport[labelIndex]) {

            std::set<int, std::less<int> > PreviousRNN;
            for (int le = DIMENSION - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                } else {
                    PreviousRNN = SetRNN(le, s);
                    if (LastRNN == PreviousRNN) {
                        PreviousRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of else
            }// end of for
        } else {
            MPL = DIMENSION;
        }
    }
    return MPL;
}

int updateMPL(std::set<int, std::less<int> > s) {
    int length = getMPL(s);
    std::set<int, std::less<int> >::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (PredictionPrefix[*jj] > length) {
            PredictionPrefix[*jj] = length;
        }
    }
    return length;
}

int updateMPLStrict(std::set<int, std::less<int> > s) {
    int length = getMPLStrict(s);
    std::set<int, std::less<int> >::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (PredictionPrefix[*jj] > length) {
            PredictionPrefix[*jj] = length;
        }
    }
    return length;
}

bool sharePrefix(std::set<int, std::less<int> >& s1,
        std::set<int, std::less<int> >& s2, std::set<int, std::less<int> >& s3,
        int level) {
    std::set<int, std::less<int> >::iterator i, j;
    bool SharePrefix = 1;
    int count = 0;
    for (count = 0, i = s1.begin(), j = s2.begin(); count < level - 2; count++, i++, j++) {
        if ((*i) != (*j)) {
            SharePrefix = 0;
            break;
        }

    }
    if (SharePrefix == 1) {
        set_union(
                s1.begin(),
                s1.end(),
                s2.begin(),
                s2.end(),
                std::insert_iterator<std::set<int, std::less<int> > >(s3,
                        s3.begin()));
        /*cout<<"Merged set:\n";
         * for(i=s3.begin();i!=s3.end();i++)
         * {
         * cout<<(*i)<<" ";
         * }
         * cout<<"\n";*/
        return SharePrefix;
    } else {
        //cout<<"\n can not merge";
        return SharePrefix;
    }

}

/* find minimum */
int findMin(int data[], int len) {
    int Min = DIMENSION;
    for (int i = 0; i < len; i++) {
        if (data[i] < Min) {
            Min = data[i];
        }
    }
    return Min;
}

/* find nearest neighbor */
int findNN(int index, int len) {
    int indexOfNN = -1;

    double Mindis = 1e9;
    for (int i = 0; i < ROWTRAINING; i++) {
        double tempdis = Euclidean(testing[index], training[i], len);

        if (tempdis < Mindis) {
            Mindis = tempdis;
            indexOfNN = i;
        }

    }
    return indexOfNN;
}

/* classify the test data */
void classification() {

    /* LoadData(testingFileName, testing, labelTesting, ROWTESTING); */
    int startfrom = findMin(PredictionPrefix, ROWTRAINING);

    for (int i = 0; i < ROWTESTING; i++) {
        for (int j = startfrom; j <= DIMENSION; j++) {

            int tempNN = findNN(i, j);

            // cout<<"\n instance "<<i<<" 's NN is "<<tempNN<<" at length "<<j;

            if (PredictionPrefix[tempNN] <= j) {

                predictedLabel[i] = labelTraining[tempNN];
                predictedLength[i] = j;

                //                if (predictedLabel[i] == labelTesting[i]) {
                //                    correct++;
                //                } else {
                //                    //                    cout<<"\n instance "<<i<<" ("<<labelTesting[i]<<")  is missed classified by instance "<<tempNN+1 <<" at length "<<predictedLength[i] <<"as  "<<predictedLabel[i];
                //                }
                break; // can be classified.
            }
        }
    }
}

/* classify the test data */
void fullClassification(double *mexOutput) {
    for (int i = 0; i < ROWTESTING; i++) {
        for (int j = 0; j < DIMENSION; j++) {

            int tempNN = findNN(i, j + 1);
            int mexIdx = i + j*ROWTESTING;
            mexOutput[mexIdx] = doubleToInt(labelTraining[tempNN]);

            // cout<<"\n instance "<<i<<" 's NN is "<<tempNN<<" at length "<<j;
        }
    }
}

double mean(int data[], int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum = sum + data[i];
    }
    return sum / len;
}
/* Return the index of a particular class label
 * e.g. For classLabel={10, 11, 12, 13}, NofClasses=4
 * find the index of 12:
 * int index = getLabelIndex(12, classLabel, NofClasses);
 * // index = 2
 */
int getLabelIndex(int label, int* classLabel, int NofClasses){
    int index;
    for(index=0; index<NofClasses; index++){
        if(classLabel[index]==label){
            return index;
        }
    }
}

double getSetDistance(std::set<int, std::less<int> > A,
        std::set<int, std::less<int> > B) {
    double minimal = 10000000;
    std::set<int, std::less<int> >::iterator iA;
    std::set<int, std::less<int> >::iterator iB;

    for (iA = A.begin(); iA != A.end(); iA++) {
        for (iB = B.begin(); iB != B.end(); iB++) {
            double temp = distanceArray[*iA][*iB];

            if (temp < minimal) {
                minimal = temp;
            }
        }
    }
    return minimal;
}

std::vector<int> RankingArray(
        std::vector<std::set<int, std::less<int> > > & List) {

    std::vector<int> result;

    for (unsigned int i = 0; i < List.size(); i++) {
        double Mindis = 100000;
        int MinIndex = -1;

        for (unsigned int j = 0; j < List.size(); j++) {
            if (j != i) {
                double tempdis = getSetDistance(List[i], List[j]);

                if (tempdis < Mindis) {
                    Mindis = tempdis;
                    MinIndex = j;
                }
            }
        } // end for inside
        result.push_back(MinIndex);

    }// endfor outside

    return result;

}

int GetMN(std::vector<std::set<int, std::less<int> > > & List, int ClusterIndex) // get the NN of cluster, if there is no MN, return -1, others , return the value
{
    // find the NN of ClusterIndex
    double minDist = 100000;
    int NN = -1;
    for (unsigned int i = 0; i < List.size(); i++) {

        if ((int)i != ClusterIndex) {

            double tempdis = getSetDistance(List[ClusterIndex], List[i]);

            if (tempdis < minDist) {
                minDist = tempdis;
                NN = i;
            }
        }
    }

    // find the NN of MinIndex

    minDist = 100000;
    int minIndex = -1;
    for (unsigned int i = 0; i < List.size(); i++) {

        if ((int)i != NN) {

            double tempdis = getSetDistance(List[NN], List[i]);

            if (tempdis < minDist) {
                minDist = tempdis;
                minIndex = i;

            }
        }
    }

    if (minIndex == ClusterIndex) {
        return NN;
    } else {
        return -1;
    }

}
int getMPL(std::set<int, std::less<int> >& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = DIMENSION;

    std::set<int, std::less<int> >::iterator i;

    std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN


    int FirstElement = *(s.begin());
    int label = doubleToInt(labelTraining[FirstElement]);
    int labelIndex = getLabelIndex(label, classLabel, NofClasses);

    // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
    std::set<int, std::less<int> > LastUsefulRNN;
    for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
        int element = *i;

        if (FullLengthClassificationStatus[element] == 1) {
            LastUsefulRNN.insert(element);
        }
    }
    // compute the support of the sequence
    int Support = (int)s.size() + (int)LastUsefulRNN.size();
    std::set<int, std::less<int> > PreviousRNN;
    std::set<int, std::less<int> > PreviousUsefulRNN;
    if (Support >= classSupport[labelIndex]) {
        for (int le = startFrom - 1; le >= 1; le--) {
            if (NNConsistent(le, s) == 0) {
                MPL = le + 1;
                break;
            } else {
                PreviousRNN = SetRNN(le, s);
                for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                    int element = *i;

                    if (FullLengthClassificationStatus[element] == 1) {
                        PreviousUsefulRNN.insert(element);
                    }
                }

                if (LastUsefulRNN == PreviousUsefulRNN) {
                    PreviousRNN.clear();
                    PreviousUsefulRNN.clear();
                } else {
                    MPL = le + 1;
                    break;
                }
            }// end of else

        }// end of for

    } else {
        MPL = DIMENSION;
    }
    return MPL;
}

int getMPLStrict(std::set<int, std::less<int> >& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = DIMENSION;

    std::set<int, std::less<int> >::iterator i;

    std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN


    int FirstElement = *(s.begin());
    int label = doubleToInt(labelTraining[FirstElement]);
    int labelIndex = getLabelIndex(label, classLabel, NofClasses);

    // compute the support of the sequence
    int Support = (int)s.size() + (int)LastRNN.size();
    std::set<int, std::less<int> > PreviousRNN;

    if (Support >= classSupport[labelIndex]) {
        for (int le = startFrom - 1; le >= 1; le--) {
            if (NNConsistent(le, s) == 0) {
                MPL = le + 1;
                break;
            } else {
                PreviousRNN = SetRNN(le, s);
                if (LastRNN == PreviousRNN) {

                    PreviousRNN.clear();
                } else {
                    MPL = le + 1;
                    break;
                }
            }// end of else
        }// end of for

    } else {
        MPL = DIMENSION;
    }

    return MPL;

}

int getMPLTest(std::set<int, std::less<int> >& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = DIMENSION;

    std::set<int, std::less<int> >::iterator i;
    if (s.size() == 1) // simple RNN Method
    {
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastRNN.size() > 0)

            {
                std::set<int, std::less<int> > PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {

                    PreviousRNN = SetRNN(le, s);

                    if (LastRNN == PreviousRNN) {

                        PreviousRNN.clear();

                    } else {

                        MPL = le + 1;
                        break;
                    }
                }// end of for
            } else {
                MPL = DIMENSION;
            }
        } else {
            MPL = DIMENSION;
        }
    } else // super-sequence Method
    {
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN


        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastRNN.size();
        std::set<int, std::less<int> > PreviousRNN;

        if (Support >= classSupport[labelIndex]) {
            for (int le = startFrom - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                }
            }// end of for

        } else {
            MPL = DIMENSION;
        }
    }
    return MPL;
}

int getMPLTest(std::set<int, std::less<int> >& s) {

    int MPL = DIMENSION;

    std::set<int, std::less<int> >::iterator i;
    if (s.size() == 1) // simple RNN Method
    {
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastRNN.size() > 0) {
                std::set<int, std::less<int> > PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {

                    PreviousRNN = SetRNN(le, s);
                    if (LastRNN == PreviousRNN) {
                        PreviousRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of for
            }
        }
    } else { // super-sequence Method
        std::set<int, std::less<int> > LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        int label = doubleToInt(labelTraining[FirstElement]);
        int labelIndex = getLabelIndex(label, classLabel, NofClasses);

        // compute the support of the sequence
        int Support = (int)s.size() + (int)LastRNN.size();

        if (Support >= classSupport[labelIndex]) {
            std::set<int, std::less<int> > PreviousRNN;
            for (int le = DIMENSION - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                }
            }// end of for
        }
    }
    return MPL;
}

int updateMPLTest(std::set<int, std::less<int> > s) {
    int length = getMPLTest(s);
    std::set<int, std::less<int> >::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (PredictionPrefix[*jj] > length) {
            PredictionPrefix[*jj] = length;
        }
    }
    return length;
}
