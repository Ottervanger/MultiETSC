#ifndef EDSC_UTIL_H_
#define EDSC_UTIL_H_

#include <string>

namespace util {

void quicksort(double a[], int start, int end);
double euclidean(double * a, double * b, int length);

// load data in UCR format to the 2d vector 'data' with corresponding 'labels'
bool readUCRData(const std::string &file,
                 std::vector<std::vector<double> > &data,
                 std::vector<int> &labels);

} // namespace util

#endif  // EDSC_UTIL_H_
