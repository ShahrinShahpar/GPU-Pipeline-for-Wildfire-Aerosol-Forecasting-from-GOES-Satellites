// csv_io.cuh — shared CSV read/write utilities
#pragma once
#include <vector>
#include <string>

// Read a CSV with optional header. Returns flat row-major data.
// nrows and ncols are set to the dimensions of the data (excluding header).
std::vector<double> read_csv_flat(const std::string& path, int& nrows, int& ncols,
                                   bool has_header = true);

// Read a single-column CSV (first column only) into a vector
std::vector<double> read_csv_col(const std::string& path, bool has_header = true);

// Write a flat row-major matrix to CSV (no header)
void write_csv_flat(const std::string& path, const double* data, int nrows, int ncols);

// Write a vector to CSV (one value per row, no header)
void write_csv_vec(const std::string& path, const double* data, int n);
