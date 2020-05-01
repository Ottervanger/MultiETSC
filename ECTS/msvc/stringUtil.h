#ifndef stringUtil_h
#define stringUtil_h

#include <sstream>
#include <string>
#include <vector>

using namespace std;

// Split string on a delim into an existing vector
vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while(getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

// Split string on a delim and return new vector
vector<string> split(const string &s, char delim=',') {
    vector<string> elems;
    return split(s, delim, elems);
}

// Split string on a delim into an existing vector
vector<int> &splitInt(const string &s, char delim, vector<int> &elems) {
    stringstream ss(s);
    string item;
    string str;
    int num;
    stringstream tmp;
    while(getline(ss, item, delim)) {
        tmp.clear(); // clear error flags
        tmp << item;
        tmp >> num;
        elems.push_back(num);
    }
    return elems;
}

// Split string on a delim and return new vector
vector<int> splitInt(const string &s, char delim=',') {
    vector<int> elems;
    return splitInt(s, delim, elems);
}

// Split string on a delim and return new vector
vector<int> splitInt(const char* c, char delim=',') {
    vector<int> elems;
    return splitInt(string(c), delim, elems);
}

#endif