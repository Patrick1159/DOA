/// @file cost_guard.h
/// @brief Pure (ROS-free) guard for association cost matrices. Munkres never
///        terminates on NaN input, so non-finite entries are replaced by the
///        "gated" cost before solve().
#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace umap {
namespace match_utils {

/// Gate value used for invalid assignments in the coarse/detailed cost matrices.
constexpr double kCostGate = 999999.0;

/// @brief Replace every non-finite entry of `cost` by `gate`.
/// @return number of replaced entries.
inline int sanitizeCostMatrix(Eigen::MatrixXd& cost, double gate = kCostGate) {
    int replaced = 0;
    for (Eigen::Index i = 0; i < cost.rows(); ++i) {
        for (Eigen::Index j = 0; j < cost.cols(); ++j) {
            if (!std::isfinite(cost(i, j))) {
                cost(i, j) = gate;
                ++replaced;
            }
        }
    }
    return replaced;
}

/// @brief True if all components of `v` are finite.
template <typename Derived>
inline bool allFinite(const Eigen::MatrixBase<Derived>& v) {
    for (Eigen::Index k = 0; k < v.size(); ++k) {
        if (!std::isfinite(static_cast<double>(v(k)))) return false;
    }
    return true;
}

}  // namespace match_utils
}  // namespace umap
