/// @file match_utils.hpp
/// @brief Small, header-only helpers shared between the coarse-match and
///        detailed-match (rematch) tracker nodes. Only factor out functions
///        that are *identical* across both nodes — do not over-abstract.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <utility>
#include <vector>

#include <umap/ObstacleTracker.h>
#include <umap/cost_guard.h>

namespace umap {
namespace match_utils {

/// @brief Return indices of detections that were not matched to any tracker.
/// @param matches         (tracker_idx, detection_idx) pairs from association.
/// @param num_detections  Total number of detections this frame.
inline std::vector<int> findUnmatchedDetections(
    const std::vector<std::pair<int, int>>& matches,
    const int num_detections) {
    std::vector<bool> matched(num_detections, false);
    for (const auto& m : matches) {
        if (m.second >= 0 && m.second < num_detections) {
            matched[m.second] = true;
        }
    }
    std::vector<int> unmatched;
    for (int i = 0; i < num_detections; ++i) {
        if (!matched[i]) unmatched.push_back(i);
    }
    return unmatched;
}

/// @brief Volumetric IoU between two axis-aligned 3D boxes given by center +
///        size (full width/height/thickness, not half-extents).
/// @return iou in [0, 1]; 0 if either box has non-positive volume or no overlap.
inline double computeIoU3D(const Eigen::Vector3d& center_a,
                           const Eigen::Vector3d& size_a,
                           const Eigen::Vector3d& center_b,
                           const Eigen::Vector3d& size_b) {
    double intersection = 1.0;
    for (int k = 0; k < 3; ++k) {
        double a_min = center_a[k] - size_a[k] / 2.0;
        double a_max = center_a[k] + size_a[k] / 2.0;
        double b_min = center_b[k] - size_b[k] / 2.0;
        double b_max = center_b[k] + size_b[k] / 2.0;
        double overlap = std::max(0.0, std::min(a_max, b_max) - std::max(a_min, b_min));
        if (overlap <= 0.0) return 0.0;
        intersection *= overlap;
    }
    double vol_a = size_a[0] * size_a[1] * size_a[2];
    double vol_b = size_b[0] * size_b[1] * size_b[2];
    double union_vol = vol_a + vol_b - intersection;
    if (union_vol <= 0.0) return 0.0;
    return intersection / union_vol;
}

/// @brief Remove trackers whose time-since-last-update exceeds `max_age`.
///        Calls `on_erase(dead_track_id)` immediately before erasing each
///        tracker, giving the caller a chance to prune per-tracker caches.
///        If `max_age <= 0.0` the function is a no-op (lifecycle disabled).
/// @tparam OnErase  Callable with signature void(int dead_track_id).
template <class OnErase>
inline void removeStaleTrackers(
    std::vector<onboardDetector::ObstacleTracker>& trackers,
    double current_time,
    double max_age,
    OnErase&& on_erase) {
    if (trackers.empty() || max_age <= 0.0) return;
    auto it = trackers.begin();
    while (it != trackers.end()) {
        const double age = current_time - it->getLastUpdateTime();
        if (age > max_age) {
            on_erase(it->track_id);
            it = trackers.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace match_utils
}  // namespace umap
