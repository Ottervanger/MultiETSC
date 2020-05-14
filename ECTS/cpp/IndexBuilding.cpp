// Build index for the training file and the testing file
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <limits>

#include "DataSetInformation.h"

#include "util.h"

void computeDist(const std::vector<std::vector<double> > &data,
                 std::vector<std::vector<int> > &distIdx,
                 std::vector<std::vector<double> > &dist) {
    int n = data.size();
    // currentMinDis[i][l] stores the distance to i's nn at prefix l
    std::vector<std::vector<double> > currentMinDis(n);
    // distIdx[i][l] stores the index of that nn
    distIdx.resize(n);
    // dist[i][j] stores the full length distance between i and j
    dist.resize(n, std::vector<double>(n, -1));

    for (int i = 0; i < n; i++) {
        currentMinDis[i].resize(data[i].size(), std::numeric_limits<double>::infinity());
        distIdx[i].resize(data[i].size());
    }

    // compute the pairwise distances
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double prefixSqDist = 0;
            int L = std::min(data[i].size(), data[j].size());
            for (int l = 0; l < L; l++) {
                double prefixDiff = data[i][l] - data[j][l];
                prefixSqDist += prefixDiff * prefixDiff;
                if (prefixSqDist < currentMinDis[i][l]) {
                    currentMinDis[i][l] = prefixSqDist;
                    distIdx[i][l] = j;
                }
                if (prefixSqDist < currentMinDis[j][l]) {
                    currentMinDis[j][l] = prefixSqDist ;
                    distIdx[j][l] = i;
                }
            }
            // full length distance
            dist[i][j] = dist[j][i] = sqrt(prefixSqDist);
        }
    }
}

int main () {
    std::vector<std::vector<double> > data;
    std::vector<int> labels;
    util::readUCRData(ECG::trainingFileName, data, labels);

    std::vector<std::vector<int> > distIdx;
    std::vector<std::vector<double> > dist;

    clock_t t;
    t = clock();
    computeDist(data, distIdx, dist);

    double indexTime = (double)(clock() - t) / CLOCKS_PER_SEC;
    std::cout << "\nindex time: " << indexTime << "s" << std::endl;

    util::saveMatrix(ECG::trainingIndexFileName, distIdx);
    util::saveMatrix(ECG::DisArrayFileName, dist);
}