
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <set>
#include <algorithm>
#include <vector>
#include <time.h>
#include <limits>

#include "DataSetInformation.h"
#include "Euclidean.h"
#include "minValue.h"
#include "find.h"

// ALgorithm parameters: minimal support
double MinimalSupport = 0;

int strictversion = 0; // 1 strict veresion , 0 loose version

// global variable
double training[ROWTRAINING][DIMENSION]; // training data set
double labelTraining[ROWTRAINING] = {0}; // training data class labels
double testing [ROWTESTING][DIMENSION]; //  testing data set
double labelTesting[ROWTESTING] = {0}; // testing data class labels
double predictedLabel[ROWTESTING] = {0}; // predicted label by the classifier
int predictedLength[ROWTESTING] = {0}; // predicted length by the classifier
int  TrainingIndex[ROWTRAINING][DIMENSION];//  store the 1NN for each space, no ranking tie
double DisArray[ROWTRAINING][ROWTRAINING] = {0}; //  the pairwise distance array of full length
int classDistri[NofClasses] = {0};
double classSupport[NofClasses] = {0};
int FullLengthClassificationStatus[ROWTRAINING];// 0 can not be classified correctly, 1 can be classified correctly
int PredictionPrefix[ROWTRAINING];
std::vector<std::set<int, std::less<int> >> NNSetList;// store the MNCS;
std::vector<int> NNSetListLength;
std::vector<int> NNSetListClassMap;// store the class label of the MNCS
int correct = 0; // the unlabeled sequence which is correctly classified.
double classificationTime; // classification time of one instance
double trainingTime; // classification time of one instance

// functions in the same file
void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[], int len  );
void getClassDis();
void LoadDisArray(const char * fileName, double Data[ROWTRAINING][ROWTRAINING]  );
void LoadTrIndex(const char * fileName, int Data[ROWTRAINING][DIMENSION]  );
std::set<int, std::less<int>> SetRNN(int l, std::set<int, std::less<int>>& s);
std::set<int, std::less<int>> SetNN(int l, std::set<int, std::less<int>>& s);
int NNConsistent(int l, std::set<int, std::less<int>>& s);
//void testUnion();
int getMPL(std::set<int, std::less<int>>& s); // loose version
int getMPL(std::set<int, std::less<int>>& s, int la, int lb); // loose version
int getMPL2(std::set<int, std::less<int>>& setA, std::set<int, std::less<int>>& setB, int la, int lb);
//void getMNCS(int onenode,double label, std::set<int,std::less<int>>& PreMNCS);
int updateMPL(std::set<int, std::less<int>> s);
//bool sharePrefix(   std::set<int, std::less<int> >& s1, std::set<int, std::less<int> >& s2, std::set<int, std::less<int> >& s3, int level        );
void printSet(std::set<int, std::less<int> > & s);
void printSetList(std::vector<std::set<int, std::less<int> >> & v) ;
int findMin( int data[], int len);
void classification();
double  mean(int data[], int len);
void report();
double SetDis( std::set<int, std::less<int> > A, std::set<int, std::less<int> > B );
std::vector<int> RankingArray(std::vector<std::set<int, std::less<int>>> & List);
int GetMN(std::vector<std::set<int, std::less<int>>> & List, int ClusterIndex); // get the NN of cluster, if there is no MN, return -1, others , return the value;

int getMPLStrict(std::set<int, std::less<int>>& s, int la, int lb);
int getMPLStrict(std::set<int, std::less<int>>& s);
int updateMPLStrict(std::set<int, std::less<int>> s);

int getMPLTest(std::set<int, std::less<int>>& s, int la, int lb);
int getMPLTest(std::set<int, std::less<int>>& s);
int updateMPLTest(std::set<int, std::less<int>> s);

void reportSynUCI();

int main () {
    // load training data
    LoadData(trainingFileName, training, labelTraining, ROWTRAINING);
    getClassDis();
    LoadDisArray(DisArrayFileName, DisArray  );
    LoadTrIndex(trainingIndexFileName, TrainingIndex  );
    // compute the full length classification status, 0 incorrect, 1 correct
    clock_t t3, t4;

    for (int i = 0; i < ROWTRAINING; i++) {
        int NNofi = TrainingIndex[i][DIMENSION - 1];
        if (labelTraining[i] == labelTraining[NNofi]) {
            FullLengthClassificationStatus[i] = 1;
        }
    }

    // intialize the prediction prefix
    for (int i = 0; i < ROWTRAINING; i++) {
        PredictionPrefix[i] = DIMENSION;

    }

    // simple RNN method
    t3 = clock();
    // simple RNN method
    for (int i = 0; i < ROWTRAINING; i++) {
        std::set<int, std::less<int> >  s;
        s.insert(i);

        if (strictversion == 0) {
            PredictionPrefix[i] = getMPL(s);
        } else {

            PredictionPrefix[i] = getMPLStrict(s);
        }
        // PredictionPrefix[i]=getMPLTest(s);
    }

    int latticeLevel = 1;
    //std::cout << "bi-roots level\n";

    std::vector<std::set<int, std::less<int> >> biRoots; // is a std::vector of std::sets
    std::vector<int> biRootsLength;// the length of the node
    std::vector<double> SetLabel;// -,1,2,3,4,5,6,   0 means no label
    // initial the length to L
    std::vector<int> CurrentNew;
    std::vector<int> NextNew;

    bool UsedInBiRoots[ROWTRAINING] = {0}; // 0 not used, 1 used

    for (int i = 0; i < ROWTRAINING; i++) {
        if (UsedInBiRoots[i] == 0) {
            int NNofi = TrainingIndex[i][DIMENSION - 1];
            // get bi-root
            if (TrainingIndex[NNofi][DIMENSION - 1] == i ) {
                std::set<int, std::less<int> > temp;
                temp.insert(i);
                temp.insert(NNofi);

                if (labelTraining[i] == labelTraining[NNofi]) {
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
                    CurrentNew.push_back(biRoots.size() - 1);

                } else { // unpure group
                    biRootsLength.push_back(0);

                    SetLabel.push_back(0);
                    biRoots.push_back(temp);
                    CurrentNew.push_back(biRoots.size() - 1);

                }

                UsedInBiRoots[i] = 1;
                UsedInBiRoots[NNofi] = 1;
            } else { // single element std::set
                std::set<int, std::less<int> > temp;
                temp.insert(i);
                biRootsLength.push_back(PredictionPrefix[i]);
                SetLabel.push_back(labelTraining[i]);
                biRoots.push_back(temp);
            }
        }
    } // end for

    // Heuristic algorithm
    latticeLevel = 2;
    int mergedPair = 0;

    std::vector<std::set<int, std::less<int> >> NNSetListCurrent = biRoots;
    std::vector<int> NNSetListCurrentLength = biRootsLength;
    std::vector<double> NNSetListCurrentLabel = SetLabel;

    std::vector<std::set<int, std::less<int> >> NNSetListNext;
    std::vector<int> NNSetListNextLength;
    std::vector<double> NNSetListNextLabel;

    while (NNSetListCurrent.size() > 1) {
        //std::cout << latticeLevel << std::endl;
        //std::cout << "number " << NNSetListCurrent.size()<< std::endl;
        mergedPair = 0;
        NNSetListNext.clear();
        NNSetListNextLength.clear();
        NNSetListNextLabel.clear();
        NextNew.clear();

        std::vector<int> status;
        for (int i = 0; i < NNSetListCurrent.size(); i++) {
            status.push_back(0);
        }

        // std::vector<int> CurrentRankingArray= RankingArray(NNSetListCurrent);

        for (int i = 0; i < CurrentNew.size(); i++) {

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

                    std::set<int, std::less<int>> tempSet;

                    std::set<int, std::less<int>> setA = NNSetListCurrent[pair[0]];
                    std::set<int, std::less<int>> setB = NNSetListCurrent[pair[1]];

                    // merge two std::set

                    std::set<int, std::less<int> > ::iterator a;
                    for (a = setA.begin(); a != setA.end(); a++) {
                        tempSet.insert(*a);

                    }

                    std::set<int, std::less<int> > ::iterator b;
                    for (b = setB.begin(); b != setB.end(); b++) {
                        tempSet.insert(*b);

                    }

                    if (label1 != 0 && label2 != 0 && label1 == label2 ) {
                        int tempLength;

                        if (setA.size() > 1 && setB.size() > 1) {
                            if (strictversion == 0) {
                                tempLength = getMPL(tempSet, NNSetListCurrentLength[pair[0]], NNSetListCurrentLength[pair[1]]);

                            } else {

                                tempLength = getMPLStrict(tempSet, NNSetListCurrentLength[pair[0]], NNSetListCurrentLength[pair[1]]);

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

                        std::set<int, std::less<int> > ::iterator jj;
                        for (jj = tempSet.begin(); jj != tempSet.end(); jj++) {
                            if (PredictionPrefix[*jj] > tempLength) {
                                PredictionPrefix[*jj] = tempLength;

                            }

                        }

                        NNSetListNextLength.push_back(tempLength);
                        NNSetListNext.push_back(tempSet);
                        NextNew.push_back(NNSetListNext.size() - 1);

                        NNSetListNextLabel.push_back(label1);
                        mergedPair++;
                        status[pair[0]] = 1;
                        status[pair[1]] = 1;

                    } else { // the pair is not pure

                        NNSetListNextLength.push_back(0);
                        NNSetListNext.push_back(tempSet);
                        NextNew.push_back(NNSetListNext.size() - 1);

                        NNSetListNextLabel.push_back(0);
                        status[pair[0]] = 1;
                        status[pair[1]] = 1;

                    }

                }

            }// end if

        }// end for

        if (mergedPair > 0) {

            //std::cout << "Merged pair" << mergedPair << std::endl;

            for (int k = 0; k < NNSetListCurrent.size(); k++) {

                if (status[k] == 0) {

                    NNSetListNext.push_back(NNSetListCurrent[k]);
                    NNSetListNextLength.push_back(NNSetListCurrentLength[k]);
                    NNSetListNextLabel.push_back(NNSetListCurrentLabel[k]);

                }
            }// end for

            NNSetListCurrent.clear();
            for (int kk = 0; kk < NNSetListNext.size(); kk++) {
                NNSetListCurrent.push_back(NNSetListNext[kk]);

            }
            NNSetListNext.clear();

            NNSetListCurrentLength.clear();
            for (int kk = 0; kk < NNSetListNextLength.size(); kk++) {
                NNSetListCurrentLength.push_back(NNSetListNextLength[kk]);

            }
            NNSetListNextLength.clear();

            NNSetListCurrentLabel.clear();
            for (int kk = 0; kk < NNSetListNextLabel.size(); kk++) {
                NNSetListCurrentLabel.push_back(NNSetListNextLabel[kk]);

            }
            NNSetListNextLabel.clear();

            // update CurrentNew and NextNew

            CurrentNew.clear();
            for (int kk = 0; kk < NextNew.size(); kk++) {
                CurrentNew.push_back(NextNew[kk]);

            }
            NextNew.clear();

            latticeLevel++;

        } else {
            break;
        }

    }// end while

    // end of hueristic algorithm
    t4 = clock();
    trainingTime = (double)(t4 - t3) / CLOCKS_PER_SEC  ;
    //std::cout << "\PredictionPrefix\n";

    /*
    for (int i = 0; i < ROWTRAINING; i++) {
        std::cout << "\ninstance " << i + 1 << " :" << PredictionPrefix[i] << " ";
    }
    */

    classification();
    report();
    reportSynUCI();
}// end main

void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[], int len  ) {
    std::ifstream inputFile( fileName, std::ifstream::in);
    if ( !inputFile ) {
        std::cerr << "file could not be opened" << std::endl;
        exit(1);
    }

    while ( !inputFile.eof() ) {
        for (int row = 0; row < len; row++) {
            for (int col = 0; col < DIMENSION + 1; col++) {
                if (col == 0) {
                    inputFile >> Labels[row];
                } else {
                    inputFile >> Data[row][col - 1];
                }
            }
        }
    }
    inputFile.close();
}

void LoadDisArray(const char * fileName, double Data[ROWTRAINING][ROWTRAINING]  ) {

    std::ifstream inputFile( fileName, std::ifstream::in);
    if ( !inputFile ) {
        std::cerr << "file could not be opened" << std::endl;
        exit(1);
    } // end if

    while ( !inputFile.eof() ) {
        for (int row = 0; row < ROWTRAINING; row++) {
            for (int col = 0; col < ROWTRAINING; col++) {
                inputFile >> Data[row][col];
            }
        }
    }

    inputFile.close();
}

void LoadTrIndex(const char * fileName, int Data[ROWTRAINING][DIMENSION]  ) {

    std::ifstream inputFile( fileName, std::ifstream::in);
    if ( !inputFile ) {
        std::cerr << "file could not be opened" << std::endl;
        exit(1);
    } // end if

    int row = 0;
    int col = 0;
    while ( !inputFile.eof() ) {
        for ( row = 0; row < ROWTRAINING; row++)
            for ( col = 0; col < DIMENSION; col++) {

                inputFile >> Data[row][col];

            }

    }

    inputFile.close();
}

void getClassDis() {
    for (int i = 0; i < NofClasses; i++) {

        classDistri[i] = find(labelTraining, Classes[i], ROWTRAINING );
        std::cout << "\n" << "Class " << i << ": " << classDistri[i] << "\n";
        classSupport[i] = classDistri[i] * MinimalSupport;

    }

}

std::set<int, std::less<int>> SetRNN(int l, std::set<int, std::less<int>>& s) { // find a std::set's RNN on prefix l
    std::set<int, std::less<int>> index;

    std::set<int, std::less<int> >::iterator i;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        for (int j = 0; j < ROWTRAINING; j++) {
            if (TrainingIndex[j][l - 1] == element && s.count(j) == 0) {
                index.insert(j);
            }

        }

    }

    return index;

}

std::set<int, std::less<int>> SetNN(int l, std::set<int, std::less<int>>& s) {
    std::set<int, std::less<int>> index;

    std::set<int, std::less<int> >::iterator i;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        int NNofelement = TrainingIndex[element][l - 1];
        index.insert(NNofelement);

    }

    return index;

}
int NNConsistent(int l, std::set<int, std::less<int>>& s) { // return 1, if it is NN consistent, return 0, if not
    std::set<int, std::less<int> >::iterator i;
    int consistent = 1;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        int NNofelement = TrainingIndex[element][l - 1]; // get the NN

        if (s.count(NNofelement) == 0) {
            consistent = 0;
            break;

        }

    }

    return consistent;

}

int getMPL(std::set<int, std::less<int>>& s) {

    int MPL = DIMENSION;

    std::set<int, std::less<int> > ::iterator i;
    if (s.size() == 1) { // simple RNN Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int, std::less<int>> LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;

            if ( FullLengthClassificationStatus[element] == 1) {

                LastUsefulRNN.insert(element);

            }

        }
        // compute the support of the sequence
        int Support = s.size() + LastUsefulRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastUsefulRNN.size() > 0) {
                std::set<int, std::less<int>> PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {

                    PreviousRNN = SetRNN(le, s);

                    std::set<int, std::less<int>> PreviousUsefulRNN;

                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;

                        if (FullLengthClassificationStatus[element] == 1) {

                            PreviousUsefulRNN.insert(element);

                        }

                    }

                    if (  LastUsefulRNN == PreviousUsefulRNN ) {

                        // CurrentUsefulRNN=PreviousUsefulRNN;

                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();

                    } else {
                        /*printSet(s);
                        printSet(PreviousUsefulRNN);
                        std::cout << "Support=" << support << "\n";*/

                        MPL = le + 1;
                        break;

                    }

                }// end of for

            } else {MPL = DIMENSION;}
        } else {
            MPL = DIMENSION;
        }

    } else { // super-sequence Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int, std::less<int>> LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;

            if (FullLengthClassificationStatus[element] == 1) {

                LastUsefulRNN.insert(element);

            }

        }
        // compute the support of the sequence
        int Support = s.size() + LastUsefulRNN.size();

        if (Support >= classSupport[labelIndex]) {

            std::set<int, std::less<int>> PreviousRNN;
            for (int le = DIMENSION - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                } else {

                    PreviousRNN = SetRNN(le, s);

                    std::set<int, std::less<int>> PreviousUsefulRNN;

                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;

                        if ( FullLengthClassificationStatus[element] == 1) {

                            PreviousUsefulRNN.insert(element);

                        }

                    }

                    if ( LastUsefulRNN == PreviousUsefulRNN) {

                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();

                    } else {
                        /*printSet(s);
                        printSet(PreviousUsefulRNN);
                        std::cout << "Support=" << support << "\n";*/
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

int getMPLStrict(std::set<int, std::less<int>>& s) {

    int MPL = DIMENSION;

    std::set<int, std::less<int> > ::iterator i;
    if (s.size() == 1) { // simple RNN Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int, std::less<int>> LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;

            if ( FullLengthClassificationStatus[element] == 1) {

                LastUsefulRNN.insert(element);

            }

        }
        // compute the support of the sequence
        int Support = s.size() + LastUsefulRNN.size();
        if (Support >= classSupport[labelIndex]) {
            if (LastUsefulRNN.size() > 0) {
                std::set<int, std::less<int>> PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {
                    PreviousRNN = SetRNN(le, s);
                    std::set<int, std::less<int>> PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;
                        if (FullLengthClassificationStatus[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }
                    if (  LastUsefulRNN == PreviousUsefulRNN ) {
                        // CurrentUsefulRNN=PreviousUsefulRNN;
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        /*printSet(s);
                        printSet(PreviousUsefulRNN);
                        std::cout << "Support=" << support << "\n";*/
                        MPL = le + 1;
                        break;
                    }
                }// end of for
            } else {MPL = DIMENSION;}
        } else {
            MPL = DIMENSION;
        }
    } else { // super-sequence Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN
        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}
        // compute the support of the sequence
        int Support = s.size() + LastRNN.size();
        if (Support >= classSupport[labelIndex]) {
            std::set<int, std::less<int>> PreviousRNN;
            for (int le = DIMENSION - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                } else {
                    PreviousRNN = SetRNN(le, s);
                    if ( LastRNN == PreviousRNN) {
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

int updateMPL(std::set<int, std::less<int>> s) {

    int length = getMPL(s);

    std::set<int, std::less<int> > ::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (PredictionPrefix[*jj] > length) {
            PredictionPrefix[*jj] = length;

        }

    }

    return length;

}

int updateMPLStrict(std::set<int, std::less<int>> s) {

    int length = getMPLStrict(s);

    std::set<int, std::less<int> > ::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (PredictionPrefix[*jj] > length) {
            PredictionPrefix[*jj] = length;
        }
    }
    return length;
}

bool sharePrefix(   std::set<int, std::less<int> >& s1, std::set<int, std::less<int> >& s2, std::set<int, std::less<int> >& s3, int level        ) {

    std::set<int, std::less<int> > ::iterator i, j;
    bool SharePrefix = 1;
    int count = 0;
    for ( count = 0, i = s1.begin(), j = s2.begin(); count < level - 2; count++, i++, j++) {
        if ((*i) != (*j)) {
            SharePrefix = 0;
            break;
        }

    }

    if (SharePrefix == 1) {
        std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                       std::insert_iterator<std::set<int, std::less<int> >> (s3, s3.begin()) );
        /*std::cout << "Merged std::set:\n";
        for (i=s3.begin();i!=s3.end();i++) {
            std::cout <<(*i)<< " ";
        }
        std::cout << "\n";*/
        return SharePrefix;
    } else {

        //std::cout << "\n can not merge";
        return SharePrefix;

    }

}
void printSetList(std::vector<std::set<int, std::less<int> >> & v) {
    std::cout << "\n";

    for (int i = 0; i < v.size(); i++) {
        std::cout << "Set " << i << ":";

        std::set<int, std::less<int> > temp = v[i];
        //printSet(temp);

    }

}

void printSet(std::set<int, std::less<int> > & s) {
    std::set<int, std::less<int> >::iterator i;
    for (i = s.begin(); i != s.end(); i++)
        std::cout << *i << " ";
    std::cout << std::endl;
}

int findMin( int data[], int len) {
    int Min = DIMENSION;
    for (int i = 0; i < len; i++) {
        if (data[i] < Min) {
            Min = data[i];

        }

    }
    return Min;

}

int findNN(int index, int len) {
    int indexOfNN = -1;

    double Mindis = 100000;

    for (int i = 0; i < ROWTRAINING; i++) {
        double tempdis = Euclidean( testing[index], training[i], len );

        if (tempdis < Mindis) {
            Mindis = tempdis;
            indexOfNN = i;

        }

    }

    return indexOfNN;
}

void classification() {

    LoadData(testingFileName, testing, labelTesting, ROWTESTING);
    int startfrom = findMin( PredictionPrefix, ROWTRAINING);
    std::cout << "\n Start from: " << startfrom << std::endl;

    clock_t t1, t2;

    t1 = clock();

    std::cout << (double)t1;

    for (int i = 0; i < ROWTESTING; i++) {
        for (int j = startfrom; j <= DIMENSION; j++) {

            int tempNN = findNN(i, j);

            // std::cout << "\n instance " << i << " 's NN is " << tempNN << " at length " << j;

            if (PredictionPrefix[tempNN] <= j) {

                predictedLabel[i] = labelTraining[tempNN];
                predictedLength[i] = j;

                if (predictedLabel[i] == labelTesting[i]) {correct++;}
                else {
                    // std::cout << "\n instance " << i << " (" << labelTesting[i] << ")  is missed classified by instance " << tempNN + 1 << " at length " << predictedLength[i] << "as  " << predictedLabel[i];
                }

                break; // can be classified.
            }

        }

    }

    t2 = clock();
    std::cout << "\n finish time: " << (double)t2 << std::endl;

    classificationTime = (double)(t2 - t1) / CLOCKS_PER_SEC  ;

}

double  mean(int data[], int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum = sum + data[i];
    }

    return sum / len;

}

void reportSynUCI() {
    for (int i = 0; i < NofClasses; i++) {
        double sum = 0; correct = 0;
        for (int j = i * 50; j < i * 50 + 50; j++) {

            sum = sum + predictedLength[j];
            if (predictedLabel[j] == i + 1) {
                correct++;
            }
        }
        std::cout << "Prediction length of class " << i << " is " << sum / 50 << std::endl;
        std::cout << "Accuracy of class " << i << " is " << (double)correct / 50 << std::endl;
    }
}

void report() {
    std::ostringstream ss;
    if (strictversion == 1) {
        ss << "\nStrict version\n";
    } else {
        ss << "\nLoose Version\n";
    }

    ss << "Heuristic Algorithm Report\n";
    ss << "The averaged predicted length of testing data is " << mean(predictedLength, ROWTESTING) << "\n";
    ss << "The averaged prediction prefix of training data is " << mean(PredictionPrefix, ROWTRAINING) << "\n";
    ss << "accuracy: " << double(correct) / ROWTESTING << "\n";
    ss << "mean classification time" << classificationTime / ROWTESTING << "s\n";
    ss << "training time           " << trainingTime << "s\n";

    // compute the false positive  and true positve
    int FP = 0, TP = 0, TC = 0, FC = 0;
    for (int i = 0; i < ROWTESTING; i++) {
        if (predictedLabel[i] == 1 && labelTesting[i] == -1) FP++;
        if (predictedLabel[i] == 1 && labelTesting[i] ==  1) TP++;

        if (labelTesting[i] == 1) {
            TC++;
        } else {
            FC++;
        }
    }

    double FPRate = (double)FP / FC;
    double TPRate = (double)TP / TC;

    ss << "false postive rate: " << FPRate << "\n";
    ss << "true postive rate: " << TPRate << "\n";

    std::cout << ss.str();

    std::ofstream outputFile(ResultfileName, std::ofstream::out | std::ofstream::app);
    outputFile << ss.str();
    outputFile.close();
}

double SetDis( std::set<int, std::less<int> > A, std::set<int, std::less<int> > B ) {

    double minimal = 10000000;

    std::set<int, std::less<int> > :: iterator iA;
    std::set<int, std::less<int> > :: iterator iB;

    for ( iA = A.begin(); iA != A.end(); iA++) {
        for (iB = B.begin(); iB != B.end(); iB++) {
            double temp = DisArray[*iA][*iB];
            //std::cout << "dis " << temp << std::endl;
            if (minimal > temp) {
                minimal = temp;
            }
        }
    }
    return minimal;
}

std::vector<int> RankingArray(std::vector<std::set<int, std::less<int>>> & List) {
    std::vector<int> result;
    for (int i = 0; i < List.size(); i++) {
        double Mindis = 100000;
        int MinIndex = -1;
        for (int j = 0; j < List.size(); j++) {
            if (j != i) {
                double tempdis = SetDis( List[i], List[j]);
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

int GetMN(std::vector<std::set<int, std::less<int>>> & List, int ClusterIndex) { // get the NN of cluster, if there is no MN, return -1, others , return the value
    // find the NN of ClusterIndex
    double Mindis = 100000;
    int NN = -1;
    for (int i = 0; i < List.size(); i++) {

        if (i != ClusterIndex) {

            double tempdis = SetDis( List[ClusterIndex], List[i]);

            if (tempdis < Mindis) {
                Mindis = tempdis;
                NN = i;
            }
        }
    }

    // find the NN of MinIndex
    Mindis = 100000;
    int MinIndex = -1;
    for (int i = 0; i < List.size(); i++) {
        if (i != NN) {
            double tempdis = SetDis( List[NN], List[i]);
            if (tempdis < Mindis) {
                Mindis = tempdis;
                MinIndex = i;
            }
        }
    }

    if (MinIndex == ClusterIndex) {return NN;}
    else {return -1;}

}
int getMPL(std::set<int, std::less<int>>& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = DIMENSION;

    std::set<int, std::less<int> > ::iterator i;

    std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

    int FirstElement = *(s.begin());
    double label = labelTraining[FirstElement];
    int labelIndex = 0;
    if (label == -1) {labelIndex = 1;}
    else {labelIndex = label - 1;}

    // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
    std::set<int, std::less<int>> LastUsefulRNN;
    for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
        int element = *i;

        if (FullLengthClassificationStatus[element] == 1) {

            LastUsefulRNN.insert(element);

        }

    }
    // compute the support of the sequence
    int Support = s.size() + LastUsefulRNN.size();
    std::set<int, std::less<int>> PreviousRNN;
    std::set<int, std::less<int>> PreviousUsefulRNN;
    if (Support >= classSupport[labelIndex]) {
        for (int le = startFrom - 1; le >= 1; le--) {
            if (NNConsistent(le, s) == 0) {
                MPL = le + 1;
                break;
            } else {

                PreviousRNN = SetRNN(le, s);

                for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                    int element = *i;

                    if ( FullLengthClassificationStatus[element] == 1) {

                        PreviousUsefulRNN.insert(element);

                    }

                }

                if ( LastUsefulRNN == PreviousUsefulRNN) {

                    PreviousRNN.clear();
                    PreviousUsefulRNN.clear();

                } else {
                    /*printSet(s);
                    printSet(PreviousUsefulRNN);
                    std::cout << "Support=" << support << "\n";*/
                    MPL = le + 1;
                    break;

                }

            }// end of else

        }// end of for

    } else {
        MPL = DIMENSION;
    }

    //}

    return MPL;

}

int getMPLStrict(std::set<int, std::less<int>>& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = DIMENSION;

    std::set<int, std::less<int> > ::iterator i;

    std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

    int FirstElement = *(s.begin());
    double label = labelTraining[FirstElement];
    int labelIndex = 0;
    if (label == -1) {labelIndex = 1;}
    else {labelIndex = label - 1;}

    // compute the support of the sequence
    int Support = s.size() + LastRNN.size();
    std::set<int, std::less<int>> PreviousRNN;

    if (Support >= classSupport[labelIndex]) {
        for (int le = startFrom - 1; le >= 1; le--) {
            if (NNConsistent(le, s) == 0) {
                MPL = le + 1;
                break;
            } else {

                PreviousRNN = SetRNN(le, s);

                if ( LastRNN == PreviousRNN) {

                    PreviousRNN.clear();

                } else {
                    /*printSet(s);
                    printSet(PreviousUsefulRNN);
                    std::cout << "Support=" << support << "\n";*/
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

int getMPLTest(std::set<int, std::less<int>>& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = DIMENSION;

    std::set<int, std::less<int> > ::iterator i;
    if (s.size() == 1) { // simple RNN Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // compute the support of the sequence
        int Support = s.size() + LastRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastRNN.size() > 0) {
                std::set<int, std::less<int>> PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {

                    PreviousRNN = SetRNN(le, s);

                    if (  LastRNN == PreviousRNN ) {

                        PreviousRNN.clear();

                    } else {

                        MPL = le + 1;
                        break;

                    }

                }// end of for

            } else {MPL = DIMENSION;}
        } else {
            MPL = DIMENSION;
        }

    } else { // super-sequence Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // compute the support of the sequence
        int Support = s.size() + LastRNN.size();
        std::set<int, std::less<int>> PreviousRNN;

        if (Support >= classSupport[labelIndex]) {
            for (int le = startFrom - 1; le >= 1; le--) {
                if (NNConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                }
                /*
                else {
                    PreviousRNN = SetRNN(le, s);
                    if ( LastRNN == PreviousRNN) {
                        PreviousRNN.clear();
                    } else {
                        printSet(s);
                        printSet(PreviousUsefulRNN);
                        std::cout << "Support=" << support << "\n";
                        MPL = le + 1;
                        break;
                    }
                }// end of else
                */
            }// end of for

        } else {
            MPL = DIMENSION;
        }
    }
    return MPL;
}

int getMPLTest(std::set<int, std::less<int>>& s) {

    int MPL = DIMENSION;

    std::set<int, std::less<int> > ::iterator i;
    if (s.size() == 1) { // simple RNN Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // compute the support of the sequence
        int Support = s.size() + LastRNN.size();

        if (Support >= classSupport[labelIndex]) {
            if (LastRNN.size() > 0) {
                std::set<int, std::less<int>> PreviousRNN;
                for (int le = DIMENSION - 1; le >= 1; le--) {

                    PreviousRNN = SetRNN(le, s);

                    if (  LastRNN == PreviousRNN ) {

                        PreviousRNN.clear();

                    } else {

                        MPL = le + 1;
                        break;

                    }

                }// end of for

            } else {MPL = DIMENSION;}
        } else {
            MPL = DIMENSION;
        }

    } else { // super-sequence Method
        std::set<int, std::less<int>> LastRNN = SetRNN(DIMENSION, s); // get full length RNN

        int FirstElement = *(s.begin());
        double label = labelTraining[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}

        // compute the support of the sequence
        int Support = s.size() + LastRNN.size();

        if (Support >= classSupport[labelIndex]) {
            std::set<int, std::less<int>> PreviousRNN;
            for (int le = DIMENSION - 1; le >= 1; le--) {
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

int updateMPLTest(std::set<int, std::less<int>> s) {
    int length = getMPLTest(s);
    std::set<int, std::less<int> > ::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (PredictionPrefix[*jj] > length) {
            PredictionPrefix[*jj] = length;
        }
    }
    return length;
}
