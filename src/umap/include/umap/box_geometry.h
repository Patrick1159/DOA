/// @file box_geometry.h
/// @brief Pure (ROS/OpenCV-free) 3D box geometry from a U-map box + its
///        vertical pixel range. Header-only so it can be unit-tested without ROS.
#pragma once

#include <algorithm>
#include <cmath>

namespace umap {

/// Result of boxGeometryFromUVBox(); same fields/meaning as `box3D` in umapType.h.
struct BoxGeometry {
    float x = 0, y = 0, z = 0;                    ///< geometric centroid (camera optical frame, m)
    float x_width = 0, y_width = 0, z_width = 0;  ///< width, height, thickness (m)
};

/// Below this front-face depth (m) the box is treated as non-physical
/// (object at the camera origin): the secant rescaling divides by it.
constexpr float kMinBoxDepthM = 1e-4f;
/// Below this front-point norm (m) the view-ray direction is undefined.
constexpr float kMinFrontNormM = 1e-4f;

/// @brief Compute the 3D box of an obstacle from its U-map column range and
///        the vertical pixel range found in the depth image.
///
/// The formulas are exactly those previously inlined in
/// umapProcess::extract_3Dbox (front-face reprojection at depth_in_near, then
/// translation half a thickness along the view ray).
///
/// @param u_x            left column of the U-map box (bb_U.x), px
/// @param u_width        width of the U-map box (bb_U.width), px
/// @param y_up,y_down    vertical pixel range of the obstacle, px
/// @param depth_in_near_m front-face depth, metres (already divided by depth_factor)
/// @param depth_in_far_m  back-face depth, metres
/// @param fx,fy,px,py    camera intrinsics
/// @param out            filled only when the function returns true
/// @return false if the geometry is non-physical or non-finite (the caller
///         must skip the box): depth_in_near <= kMinBoxDepthM, front_norm <=
///         kMinFrontNormM, or any output component is NaN/inf.
inline bool boxGeometryFromUVBox(int u_x, int u_width, int y_up, int y_down,
                                 float depth_in_near_m, float depth_in_far_m,
                                 float fx, float fy, float px, float py,
                                 BoxGeometry& out) {
    const float depth_in_near = depth_in_near_m;
    const float depth_in_far = depth_in_far_m;
    if (!std::isfinite(depth_in_near) || !std::isfinite(depth_in_far) ||
        !(depth_in_near > kMinBoxDepthM)) {
        return false;
    }

    const float u_center = u_x + u_width / 2.0f;
    const float v_center = (y_up + y_down) / 2.0f;

    const float x_front = (u_center - px) * depth_in_near / fx;
    const float y_front = (v_center - py) * depth_in_near / fy;
    const float front_norm = std::sqrt(x_front * x_front +
                                       y_front * y_front +
                                       depth_in_near * depth_in_near);
    if (!std::isfinite(front_norm) || !(front_norm > kMinFrontNormM)) {
        return false;
    }

    const float thickness_z = depth_in_far - depth_in_near;
    const float thickness_view = thickness_z * front_norm / depth_in_near;
    const float scale = (front_norm + thickness_view / 2.0f) / front_norm;

    BoxGeometry g;
    g.x = x_front * scale;
    g.y = y_front * scale;
    g.z = depth_in_near * scale;
    g.x_width = std::max(0.1f, u_width * depth_in_near / fx);
    g.y_width = std::max(0.1f, (y_down - y_up) * depth_in_near / fy);
    g.z_width = std::max(0.1f, thickness_view);

    if (!std::isfinite(g.x) || !std::isfinite(g.y) || !std::isfinite(g.z) ||
        !std::isfinite(g.x_width) || !std::isfinite(g.y_width) ||
        !std::isfinite(g.z_width)) {
        return false;
    }
    out = g;
    return true;
}

}  // namespace umap
