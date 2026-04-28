#include "csv_io.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

std::vector<double> read_csv_flat(const std::string& path, int& nrows, int& ncols,
                                   bool has_header) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    if (has_header) std::getline(f, line);
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> row;
        while (std::getline(ss, tok, ',')) {
            // Strip surrounding quotes if present
            if (tok.size() >= 2 && tok.front() == '"' && tok.back() == '"')
                tok = tok.substr(1, tok.size() - 2);
            try {
                row.push_back(std::stod(tok));
            } catch (...) {
                row.push_back(std::nan(""));
            }
        }
        if (!row.empty()) rows.push_back(row);
    }
    nrows = (int)rows.size();
    ncols = nrows > 0 ? (int)rows[0].size() : 0;
    std::vector<double> flat(nrows * ncols, std::nan(""));
    for (int i = 0; i < nrows; i++)
        for (int j = 0; j < (int)rows[i].size() && j < ncols; j++)
            flat[i * ncols + j] = rows[i][j];
    return flat;
}

std::vector<double> read_csv_col(const std::string& path, bool has_header) {
    int nrows, ncols;
    auto flat = read_csv_flat(path, nrows, ncols, has_header);
    if (ncols == 0) throw std::runtime_error("read_csv_col: no columns in " + path);
    std::vector<double> col(nrows);
    for (int i = 0; i < nrows; i++) col[i] = flat[i * ncols];
    return col;
}

void write_csv_flat(const std::string& path, const double* data, int nrows, int ncols) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot write: " + path);
    // Header row — matches R's write.csv convention so read.csv(header=TRUE) works
    for (int j = 0; j < ncols; j++) {
        if (j > 0) f << ',';
        f << 'V' << (j + 1);
    }
    f << '\n';
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (j > 0) f << ',';
            if (std::isnan(data[i * ncols + j])) f << "NaN";
            else f << data[i * ncols + j];
        }
        f << '\n';
    }
    if (!f.good()) throw std::runtime_error("Write error: " + path);
}

void write_csv_vec(const std::string& path, const double* data, int n) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot write: " + path);
    for (int i = 0; i < n; i++) {
        if (std::isnan(data[i])) f << "NaN\n";
        else f << data[i] << '\n';
    }
    if (!f.good()) throw std::runtime_error("Write error: " + path);
}
