#ifndef ECTS_UTIL_H_
#define ECTS_UTIL_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

namespace util {

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
void readUCRData(const char * file,
                 std::vector<std::vector<double> > &data,
                 std::vector<int> &labels);

template<typename T>
void readDMatrix(const char * file,
                 std::vector<std::vector<T> > &data) {
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "File \'" << file << "\' could not be opened" << std::endl;
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

} // namespace util

#endif  // ECTS_UTIL_H_
