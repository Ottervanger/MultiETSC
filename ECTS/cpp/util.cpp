#include "util.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace util {

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
void readUCRData(const char * fileName,
                 std::vector<std::vector<double> > data,
                 std::vector<int> labels) {
    std::ifstream inputFile(fileName);
    if (inputFile.fail()) {
        std::cerr << "File \'" << fileName << "\' could not be opened" << std::endl;
        exit(1);
    }

    std::string line;
    // double type of labels for compatibility with original files
    double label;
    std::vector<double> row;
    while (inputFile >> label) {
        labels.push_back((int)label);
        std::getline(inputFile, line);
        std::stringstream ss(line);
        row.clear();
        double v;
        while (ss >> v) row.push_back(v);
        data.push_back(row);
    }
}

} // namespace util
