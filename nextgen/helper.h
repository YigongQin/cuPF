#pragma once

#include <string>

void print2d(float* array, int fnx, int fny);
void printCudaInfo(int rank, int i);
void getParam(std::string lineText, std::string key, float& param);
void read_input(std::string input, float* target);
void read_input(std::string input, int* target);
template <typename T>
std::string to_stringp(const T a_value, int n );

