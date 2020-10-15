#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "util.h"

namespace util {

double euclidean(double * a, double * b, int length) {
    double ret = 0;
    for (int i=0; i<length;i++) {
        double dist = a[i]-b[i];
        ret += dist * dist;
    }
    return ret > 0 ? sqrt(ret) : 0;
}

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
bool readUCRData(const std::string &f,
                 std::vector<std::vector<double> > &data,
                 std::vector<int> &labels) {
    std::ifstream ifs(f);
    if (ifs.fail()) {
        std::cerr << "File \'" << f << "\' could not be opened for reading" << std::endl;
        return false;
    }

    std::string line;
    // double type of labels for compatibility with original files
    double label;
    std::vector<double> row;
    while (ifs >> label) {
        // the first value of each row is the label
        labels.push_back((int)label);
        // the remaining values are the time series
        std::getline(ifs, line);
        std::stringstream ss(line);
        double v;
        while (ss >> v) row.push_back(v);
        data.push_back(row);
        row.clear();
    }
    return true;
}

} // namespace util
