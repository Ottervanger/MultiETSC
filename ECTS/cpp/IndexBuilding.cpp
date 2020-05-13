// Build index for the training file and the testing file
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include "DataSetInformation.h"
#include "Euclidean.h"
#include "minValue.h"
#include "math.h"

using namespace std;
using namespace ECG;

double training[ROWTRAINING][DIMENSION]; // training data set
double labelTraining[ROWTRAINING] = {0}; // training data class labels
double testing [ROWTESTING][DIMENSION] = {0}; //  testing data set
double labelTesting[ROWTESTING] = {0}; // testing data class labels
int  TrainingIndex[ROWTRAINING][DIMENSION] = {0}; //  store the 1NN for each space, no ranking tie
double DisArray[ROWTRAINING][ROWTRAINING] = {0}; //  the pairwise distance array of full length
double  CurrentMinDis[ROWTRAINING][DIMENSION] = {0}; //

void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[]  );
void BuildIndex();
void SaveTrIndex(const char * fileName, int index[ROWTRAINING][DIMENSION]);
void SaveDisArray(const char * fileName, double disarr[ROWTRAINING][ROWTRAINING]);


int main () {
	LoadData(trainingFileName, training, labelTraining);
	BuildIndex();
}// end main

void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[]  ) {
	ifstream inputFile( fileName, ios::in);
	if ( !inputFile ) {
		cerr << "file could not be opened" << endl;
		exit(1);
	} // end if
	int row = 0;
	int col = 0;
	while ( !inputFile.eof() ) {
		for ( row = 0; row < ROWTRAINING; row++)
			for ( col = 0; col < DIMENSION + 1; col++) {
				if (col == 0) {
					inputFile >> Labels[row];
				} else {
					inputFile >> Data[row][col - 1];
				}
			}
	}
	inputFile.close();
}

void BuildIndex() {
	cout << "BuildingIndex\n";
	clock_t t3, t4;
	int  LargeNumber = 10000;
	for ( int row = 0; row < ROWTRAINING; row++) {
		for (int col = 0; col < DIMENSION; col++) {
			CurrentMinDis[row][col] = LargeNumber;
		}
	}
	// initial the distance between itself as -1;
	for ( int row = 0; row < ROWTRAINING; row++) {
		DisArray[row][row] = -1;
	}
	t3 = clock();
	// compute the pairwise distance, d(i,i)=0, d(i,j)=d(j,i)
	for (int i = 0; i < ROWTRAINING; i++) {
		for (int j = i + 1; j < ROWTRAINING; j++) {
			double prefixEuclidean = 0;
			for (int l = 0; l < DIMENSION; l++) {
				prefixEuclidean = prefixEuclidean + (training[i][l] - training[j][l]) * (training[i][l] - training[j][l]);
				if (prefixEuclidean < CurrentMinDis[i][l]) {
					CurrentMinDis[i][l] = prefixEuclidean ;
					TrainingIndex[i][l] = j;
				}
				if (prefixEuclidean < CurrentMinDis[j][l]) {
					CurrentMinDis[j][l] = prefixEuclidean ;
					TrainingIndex[j][l] = i;
				}
				if (l == DIMENSION - 1) {
					DisArray[i][j] = sqrt(prefixEuclidean);
					DisArray[j][i] = DisArray[i][j];
				}
			} // end of for l
		} // end of for j
	} // end of for i
	t4 = clock();
	double indexTime = (double)(t4 - t3) / CLOCKS_PER_SEC  ;
	cout << "\nindexTime is" << indexTime;

	SaveTrIndex(trainingIndexFileName, TrainingIndex);
	SaveDisArray(DisArrayFileName, DisArray);
}// end of building index

void SaveTrIndex(const char * fileName, int index[ROWTRAINING][DIMENSION]) {
	ofstream outputFile(fileName, ios::out);
	for (int row = 0; row < ROWTRAINING; row++) {
		for (int col = 0; col < DIMENSION; col++) {
			outputFile << index[row][col];
			outputFile << " ";
		}
		outputFile << endl;
	}
	outputFile.close();
}

void SaveDisArray(const char * fileName, double disarr[ROWTRAINING][ROWTRAINING]) {
	ofstream outputFile(fileName, ios::out);
	for (int row = 0; row < ROWTRAINING; row++) {
		for (int col = 0; col < ROWTRAINING; col++) {
			outputFile << disarr[row][col];
			outputFile << " ";
		}
		outputFile << endl;
	}
	outputFile.close();
}