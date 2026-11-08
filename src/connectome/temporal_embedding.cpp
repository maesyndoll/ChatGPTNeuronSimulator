#include "connectome/temporal_embedding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace connectome {
namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Arc { std::uint32_t node; double delay; };

std::vector<double> dijkstra(
    std::uint32_t start, const std::vector<std::vector<Arc>>& graph,
    const std::vector<std::uint32_t>& component,
    std::uint32_t component_id) {
    std::vector<double> distance(graph.size(), kInf);
    using Entry = std::pair<double, std::uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    distance[start] = 0.0;
    queue.push({0.0, start});
    while (!queue.empty()) {
        const auto [cost, node] = queue.top();
        queue.pop();
        if (cost != distance[node]) continue;
        for (const auto& arc : graph[node]) {
            if (component[arc.node] != component_id) continue;
            const double candidate = cost + arc.delay;
            if (candidate < distance[arc.node]) {
                distance[arc.node] = candidate;
                queue.push({candidate, arc.node});
            }
        }
    }
    return distance;
}

double symmetric_distance(double forward, double reverse) {
    if (std::isfinite(forward) && std::isfinite(reverse))
        return 0.5 * (forward + reverse);
    if (std::isfinite(forward)) return forward;
    if (std::isfinite(reverse)) return reverse;
    return kInf;
}

// Jacobi eigensolver for a small real symmetric matrix. Eigenvectors are
// returned as columns, sorted by descending eigenvalue.
void symmetric_eigen(std::vector<double> matrix, std::size_t n,
                     std::vector<double>& values,
                     std::vector<double>& vectors) {
    vectors.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) vectors[i*n+i] = 1.0;
    const std::size_t max_iterations = 80 * n * n;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::size_t p = 0, q = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                if (std::abs(matrix[i*n+j]) > largest) {
                    largest = std::abs(matrix[i*n+j]); p = i; q = j;
                }
        if (largest < 1e-10) break;
        const double app = matrix[p*n+p], aqq = matrix[q*n+q];
        const double apq = matrix[p*n+q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
            (std::abs(tau) + std::sqrt(1.0 + tau*tau));
        const double c = 1.0 / std::sqrt(1.0 + t*t);
        const double s = t*c;
        for (std::size_t k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            const double mkp = matrix[k*n+p], mkq = matrix[k*n+q];
            matrix[k*n+p] = matrix[p*n+k] = c*mkp - s*mkq;
            matrix[k*n+q] = matrix[q*n+k] = s*mkp + c*mkq;
        }
        matrix[p*n+p] = app - t*apq;
        matrix[q*n+q] = aqq + t*apq;
        matrix[p*n+q] = matrix[q*n+p] = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = vectors[k*n+p], vkq = vectors[k*n+q];
            vectors[k*n+p] = c*vkp - s*vkq;
            vectors[k*n+q] = s*vkp + c*vkq;
        }
    }
    values.resize(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = matrix[i*n+i];
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return values[a] > values[b];
    });
    std::vector<double> sorted_values(n), sorted_vectors(n*n);
    for (std::size_t k = 0; k < n; ++k) {
        sorted_values[k] = values[order[k]];
        for (std::size_t row = 0; row < n; ++row)
            sorted_vectors[row*n+k] = vectors[row*n+order[k]];
    }
    values.swap(sorted_values);
    vectors.swap(sorted_vectors);
}

void embed_component(const std::vector<std::uint32_t>& members,
                     const std::vector<std::vector<Arc>>& forward_graph,
                     const std::vector<std::vector<Arc>>& reverse_graph,
                     std::uint32_t component_id,
                     const std::vector<std::uint32_t>& component,
                     std::size_t landmark_limit,
                     double offset,
                     std::vector<Vec3>& output) {
    const std::size_t n = members.size();
    if (n == 1) { output[members.front()] = {offset, 0, 0}; return; }
    const std::size_t m = std::min(n, std::max<std::size_t>(4, landmark_limit));
    std::vector<std::uint32_t> landmarks;
    landmarks.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t index = (i * n) / m;
        landmarks.push_back(members[index]);
    }

    std::vector<std::vector<double>> from(m), to(m);
    for (std::size_t i = 0; i < m; ++i) {
        from[i] = dijkstra(landmarks[i], forward_graph, component, component_id);
        to[i] = dijkstra(landmarks[i], reverse_graph, component, component_id);
    }

    std::vector<double> distances(m*m, 0.0);
    double max_distance = 1.0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const double d = symmetric_distance(
                from[i][landmarks[j]], to[i][landmarks[j]]);
            distances[i*m+j] = std::isfinite(d) ? d : 0.0;
            max_distance = std::max(max_distance, distances[i*m+j]);
        }
    }
    // Classical MDS: B = -1/2 J D^2 J.
    std::vector<double> row_mean(m, 0.0);
    double grand_mean = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j)
            row_mean[i] += distances[i*m+j] * distances[i*m+j];
        row_mean[i] /= static_cast<double>(m);
        grand_mean += row_mean[i];
    }
    grand_mean /= static_cast<double>(m);
    std::vector<double> gram(m*m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            const double d2 = distances[i*m+j] * distances[i*m+j];
            gram[i*m+j] = -0.5 * (d2 - row_mean[i] - row_mean[j] + grand_mean);
        }
    std::vector<double> eigenvalues, eigenvectors;
    symmetric_eigen(std::move(gram), m, eigenvalues, eigenvectors);
    std::vector<Vec3> landmark_positions(m);
    for (std::size_t i = 0; i < m; ++i) {
        double* axes[] = {&landmark_positions[i].x,
                          &landmark_positions[i].y,
                          &landmark_positions[i].z};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double lambda = axis < m ? std::max(0.0, eigenvalues[axis]) : 0.0;
            *axes[axis] = eigenvectors[i*m+axis] * std::sqrt(lambda);
        }
    }

    // Landmark nodes have exact MDS coordinates. Other nodes are trilaterated
    // from their temporal distances to the landmark set.
    for (std::size_t i = 0; i < m; ++i)
        output[landmarks[i]] = landmark_positions[i];

    for (const auto node : members) {
        bool is_landmark = false;
        for (const auto id : landmarks) if (id == node) { is_landmark = true; break; }
        if (is_landmark) continue;
        std::vector<double> normal(9, 0.0), rhs(3, 0.0);
        std::size_t usable = 0;
        const double d0 = symmetric_distance(from[0][node], to[0][node]);
        if (!std::isfinite(d0)) { output[node] = {0,0,0}; continue; }
        for (std::size_t k = 1; k < m; ++k) {
            const double dk = symmetric_distance(from[k][node], to[k][node]);
            if (!std::isfinite(dk)) continue;
            const Vec3 a = landmark_positions[k], b = landmark_positions[0];
            const double row[3] = {2.0*(a.x-b.x), 2.0*(a.y-b.y), 2.0*(a.z-b.z)};
            const double value = a.x*a.x+a.y*a.y+a.z*a.z -
                b.x*b.x-b.y*b.y-b.z*b.z + d0*d0-dk*dk;
            for (int r = 0; r < 3; ++r) {
                rhs[r] += row[r]*value;
                for (int c = 0; c < 3; ++c) normal[r*3+c] += row[r]*row[c];
            }
            ++usable;
        }
        if (usable < 3) { output[node] = {0,0,0}; continue; }
        // Gaussian elimination with partial pivoting.
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            for (int r = col+1; r < 3; ++r)
                if (std::abs(normal[r*3+col]) > std::abs(normal[pivot*3+col])) pivot = r;
            if (std::abs(normal[pivot*3+col]) < 1e-12) break;
            for (int c = col; c < 3; ++c) std::swap(normal[col*3+c], normal[pivot*3+c]);
            std::swap(rhs[col], rhs[pivot]);
            const double divisor = normal[col*3+col];
            for (int c = col; c < 3; ++c) normal[col*3+c] /= divisor;
            rhs[col] /= divisor;
            for (int r = 0; r < 3; ++r) if (r != col) {
                const double factor = normal[r*3+col];
                for (int c = col; c < 3; ++c) normal[r*3+c] -= factor*normal[col*3+c];
                rhs[r] -= factor*rhs[col];
            }
        }
        output[node] = {rhs[0]+offset, rhs[1], rhs[2]};
    }
    // Center and normalize this component to a consistent visual scale.
    Vec3 center{};
    for (const auto id : members) {
        center.x += output[id].x-offset; center.y += output[id].y; center.z += output[id].z;
    }
    center.x /= n; center.y /= n; center.z /= n;
    double radius = 0.0;
    for (const auto id : members) {
        auto& p = output[id];
        p.x -= center.x + offset; p.y -= center.y; p.z -= center.z;
        radius = std::max(radius, std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z));
    }
    if (radius > 1e-12)
        for (const auto id : members) {
            output[id].x = offset + output[id].x / radius;
            output[id].y /= radius; output[id].z /= radius;
        }
    (void)max_distance;
}

} // namespace

EmbeddingResult embed_temporal_graph(std::size_t node_count,
    const std::vector<TemporalEdge>& edges, const EmbeddingOptions& options) {
    if (node_count == 0) throw std::invalid_argument("temporal graph must contain nodes");
    if (node_count > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("temporal graph exceeds uint32 node IDs");
    std::vector<std::vector<Arc>> forward(node_count), reverse(node_count), weak(node_count);
    for (const auto& edge : edges) {
        if (edge.source >= node_count || edge.target >= node_count)
            throw std::invalid_argument("temporal edge endpoint is out of range");
        if (edge.source == edge.target) continue;
        if (!std::isfinite(edge.delay) || edge.delay <= 0.0)
            throw std::invalid_argument("temporal edge delay must be finite and positive");
        forward[edge.source].push_back({edge.target, edge.delay});
        reverse[edge.target].push_back({edge.source, edge.delay});
        weak[edge.source].push_back({edge.target, edge.delay});
        weak[edge.target].push_back({edge.source, edge.delay});
    }
    EmbeddingResult result;
    result.positions.resize(node_count);
    result.components.assign(node_count, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::vector<std::uint32_t>> members;
    for (std::uint32_t start = 0; start < node_count; ++start) {
        if (result.components[start] != std::numeric_limits<std::uint32_t>::max()) continue;
        const auto id = static_cast<std::uint32_t>(members.size());
        members.emplace_back();
        std::queue<std::uint32_t> queue;
        queue.push(start); result.components[start] = id;
        while (!queue.empty()) {
            const auto node = queue.front(); queue.pop();
            members.back().push_back(node);
            for (const auto& arc : weak[node])
                if (result.components[arc.node] == std::numeric_limits<std::uint32_t>::max()) {
                    result.components[arc.node] = id; queue.push(arc.node);
                }
        }
    }
    result.component_count = members.size();
    double offset = 0.0;
    const auto landmark_count = std::max<std::size_t>(4, options.landmark_count);
    for (std::uint32_t id = 0; id < members.size(); ++id) {
        embed_component(members[id], forward, reverse, id, result.components,
                        landmark_count, offset, result.positions);
        offset += 2.0 + std::max(0.0, options.component_gap);
    }
    for (const auto& p : result.positions)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::runtime_error("temporal embedding produced non-finite coordinates");
    return result;
}

} // namespace connectome
