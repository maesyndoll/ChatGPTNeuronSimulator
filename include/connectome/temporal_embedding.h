#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace connectome {

struct TemporalEdge {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    double delay = 1.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct EmbeddingOptions {
    std::size_t landmark_count = 32;
    double component_gap = 2.5;
};

struct EmbeddingResult {
    std::vector<Vec3> positions;
    // Weakly connected component index for each input node.
    std::vector<std::uint32_t> components;
    std::size_t component_count = 0;
};

// Embeds a directed, delay-weighted graph in 3D. Directed edge delays remain
// untouched; only the geometric distance is symmetrized. If both directed
// shortest paths exist their mean is used; otherwise the available direction
// is used. Weakly disconnected components are embedded independently.
EmbeddingResult embed_temporal_graph(
    std::size_t node_count,
    const std::vector<TemporalEdge>& edges,
    const EmbeddingOptions& options = {});

} // namespace connectome
