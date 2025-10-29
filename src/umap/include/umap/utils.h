#ifndef UMAP_UTILS_H
#define UMAP_UTILS_H

#include <string>
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace umap {

// Generate a globally unique UUID (Version 4)
inline std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    // Generate two 64-bit random numbers
    uint64_t random1 = dis(gen);
    uint64_t random2 = dis(gen);

    // Format as UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    
    // First part (8 hex digits)
    ss << std::setw(8) << ((random1 >> 32) & 0xFFFFFFFF) << "-";
    
    // Second part (4 hex digits)
    ss << std::setw(4) << ((random1 >> 16) & 0xFFFF) << "-";
    
    // Third part (4 hex digits, version 4)
    ss << std::setw(4) << (0x4000 | ((random1 >> 4) & 0x0FFF)) << "-";
    
    // Fourth part (4 hex digits, variant bits)
    ss << std::setw(4) << (0x8000 | ((random2 >> 52) & 0x3FFF)) << "-";
    
    // Fifth part (12 hex digits)
    ss << std::setw(12) << (random2 & 0xFFFFFFFFFFFF);
    
    return ss.str();
}

// Alternative: Generate UUID with timestamp for better uniqueness
inline std::string generate_uuid_with_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    
    uint64_t random = dis(gen);
    
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    
    // Use timestamp in first parts
    ss << std::setw(8) << ((timestamp >> 32) & 0xFFFFFFFF) << "-";
    ss << std::setw(4) << ((timestamp >> 16) & 0xFFFF) << "-";
    ss << std::setw(4) << (0x4000 | ((timestamp >> 4) & 0x0FFF)) << "-";
    ss << std::setw(4) << (0x8000 | ((random >> 48) & 0x3FFF)) << "-";
    ss << std::setw(12) << (random & 0xFFFFFFFFFFFF);
    
    return ss.str();
}

} // namespace umap

#endif // UMAP_UTILS_H