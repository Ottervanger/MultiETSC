#ifndef ECTS_UTIL_H_
#define ECTS_UTIL_H_

#include <vector>
#include <string>

namespace util {

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
bool readUCRData(const std::string &file,
                 std::vector<std::vector<double> > &data,
                 std::vector<int> &labels);

} // namespace util

#endif  // ECTS_UTIL_H_
