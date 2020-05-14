#include "util.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace util {

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
void readUCRData(const char * f,
                 std::vector<std::vector<double> > &data,
                 std::vector<int> &labels) {
    std::ifstream ifs(f);
    if (ifs.fail()) {
        std::cerr << "File \'" << f << "\' could not be opened for reading" << std::endl;
        exit(1);
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
}


template void readDMatrix(const char * f, std::vector<std::vector<int> > &data);
template void readDMatrix(const char * f, std::vector<std::vector<double> > &data);

template<typename T>
void readDMatrix(const char * f, std::vector<std::vector<T> > &data) {
    std::ifstream ifs(f);
    if (ifs.fail()) {
        std::cerr << "File \'" << f << "\' could not be opened for reading" << std::endl;
        exit(1);
    }
    std::string line;
    std::vector<T> row;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        T v;
        while (ss >> v) row.push_back(v);
        data.push_back(row);
        row.clear();
    }
}

template void saveMatrix(const char * f, const std::vector<std::vector<int> > &data);
template void saveMatrix(const char * f, const std::vector<std::vector<double> > &data);

template<typename T>
void saveMatrix(const char * f, const std::vector<std::vector<T> > &data) {
    std::ofstream ofs(f);
    if (ofs.fail()) {
        std::cerr << "File \'" << f << "\' could not be opened for writing" << std::endl;
        exit(1);
    }
    for (auto r : data) {
        for (T v : r)
            ofs << v << "\t";
        ofs << std::endl;
    }
}

} // namespace util
