#ifndef ECTS_UTIL_H_
#define ECTS_UTIL_H_

#include <vector>

namespace util {

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
bool readUCRData(const char * file,
                 std::vector<std::vector<double> > &data,
                 std::vector<int> &labels);

// load whitespace separated data from file to 2d vector
template<typename T>
bool readDMatrix(const char * file,
                 std::vector<std::vector<T> > &data);

// save tab separated data from file to 2d vector
template<typename T>
bool saveMatrix(const char * file,
                const std::vector<std::vector<T> > &data);

} // namespace util

#endif  // ECTS_UTIL_H_
