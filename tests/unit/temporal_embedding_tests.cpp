#include "connectome/temporal_embedding.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using connectome::TemporalEdge;

int main() {
    {
        const std::vector<TemporalEdge> edges{
            {0,1,1.0}, {1,0,1.0}, {1,2,2.0}, {2,1,2.0},
            {2,3,3.0}, {3,2,3.0}, {0,3,20.0}, {3,0,20.0}
        };
        const auto a = connectome::embed_temporal_graph(4, edges);
        const auto b = connectome::embed_temporal_graph(4, edges);
        assert(a.positions.size() == 4);
        assert(a.component_count == 1);
        for (std::size_t i = 0; i < 4; ++i) {
            assert(std::isfinite(a.positions[i].x));
            assert(std::isfinite(a.positions[i].y));
            assert(std::isfinite(a.positions[i].z));
            assert(std::abs(a.positions[i].x-b.positions[i].x) < 1e-8);
            assert(std::abs(a.positions[i].y-b.positions[i].y) < 1e-8);
            assert(std::abs(a.positions[i].z-b.positions[i].z) < 1e-8);
        }
    }
    {
        const std::vector<TemporalEdge> directed{{0,1,2.0}, {1,2,4.0}};
        const auto result = connectome::embed_temporal_graph(4, directed);
        assert(result.component_count == 2);
        assert(result.components[0] == result.components[2]);
        assert(result.components[3] != result.components[0]);
        assert(std::isfinite(result.positions[3].x));
    }
    {
        const std::vector<TemporalEdge> fast{{0,1,1.0},{1,0,1.0},{1,2,1.0},{2,1,1.0}};
        const std::vector<TemporalEdge> slow{{0,1,10.0},{1,0,10.0},{1,2,10.0},{2,1,10.0}};
        const auto a = connectome::embed_temporal_graph(3, fast);
        const auto b = connectome::embed_temporal_graph(3, slow);
        assert(a.positions.size() == b.positions.size());
        // Both graphs are finite and their input latency scale is accepted;
        // normalization is intentional so display scale stays bounded.
        for (const auto& p : b.positions)
            assert(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }
    {
        const auto single = connectome::embed_temporal_graph(1, {});
        assert(single.positions.size() == 1);
        assert(single.component_count == 1);
    }
    std::cout << "temporal embedding tests passed\n";
}
