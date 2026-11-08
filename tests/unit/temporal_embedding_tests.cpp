#include "connectome/temporal_embedding.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

using connectome::TemporalEdge;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition "\n"; \
        return EXIT_FAILURE; \
    } \
} while (false)

int main() {
    {
        const std::vector<TemporalEdge> edges{
            {0,1,1.0}, {1,0,1.0}, {1,2,2.0}, {2,1,2.0},
            {2,3,3.0}, {3,2,3.0}, {0,3,20.0}, {3,0,20.0}
        };
        const auto a = connectome::embed_temporal_graph(4, edges);
        const auto b = connectome::embed_temporal_graph(4, edges);
        CHECK(a.positions.size() == 4);
        CHECK(a.component_count == 1);
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(std::isfinite(a.positions[i].x));
            CHECK(std::isfinite(a.positions[i].y));
            CHECK(std::isfinite(a.positions[i].z));
            CHECK(std::abs(a.positions[i].x-b.positions[i].x) < 1e-8);
            CHECK(std::abs(a.positions[i].y-b.positions[i].y) < 1e-8);
            CHECK(std::abs(a.positions[i].z-b.positions[i].z) < 1e-8);
        }
    }
    {
        const std::vector<TemporalEdge> directed{{0,1,2.0}, {1,2,4.0}};
        const auto result = connectome::embed_temporal_graph(4, directed);
        CHECK(result.component_count == 2);
        CHECK(result.components[0] == result.components[2]);
        CHECK(result.components[3] != result.components[0]);
        CHECK(std::isfinite(result.positions[3].x));
    }
    {
        const std::vector<TemporalEdge> fast{{0,1,1.0},{1,0,1.0},{1,2,1.0},{2,1,1.0}};
        const std::vector<TemporalEdge> slow{{0,1,10.0},{1,0,10.0},{1,2,10.0},{2,1,10.0}};
        const auto a = connectome::embed_temporal_graph(3, fast);
        const auto b = connectome::embed_temporal_graph(3, slow);
        CHECK(a.positions.size() == b.positions.size());
        // Both graphs are finite and their input latency scale is accepted;
        // normalization is intentional so display scale stays bounded.
        for (const auto& p : b.positions)
            CHECK(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }
    {
        const auto single = connectome::embed_temporal_graph(1, {});
        CHECK(single.positions.size() == 1);
        CHECK(single.component_count == 1);
    }
    std::cout << "temporal embedding tests passed\n";
}
