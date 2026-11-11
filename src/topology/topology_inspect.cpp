#include "topology/topology_io.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_distribution(
    const char* name,
    const spiking::TopologyDistributionStats& stats) {

    std::cout << std::left << std::setw(22) << name
              << " count=" << stats.count
              << " min=" << stats.minimum
              << " mean=" << stats.mean
              << " max=" << stats.maximum << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: topology_inspect <topology.csrbin>\n";
        return 2;
    }

    try {
        const auto topology = spiking::load_csr_topology(argv[1]);
        const auto stats = spiking::analyze_csr_topology(topology);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Topology inspection\n"
                  << "===================\n"
                  << "File: " << argv[1] << '\n'
                  << "Excitatory neurons: " << stats.excitatory_count << '\n'
                  << "Inhibitory neurons: " << stats.inhibitory_count << '\n'
                  << "Total neurons: "
                  << (static_cast<std::uint64_t>(stats.excitatory_count) +
                      stats.inhibitory_count) << '\n'
                  << "Total synapses: " << stats.synapse_count << '\n'
                  << "Connection density: " << stats.density << '\n'
                  << '\n'
                  << "Connectivity by population\n"
                  << "---------------------------\n"
                  << "E -> E: " << stats.excitatory_to_excitatory << '\n'
                  << "E -> I: " << stats.excitatory_to_inhibitory << '\n'
                  << "I -> E: " << stats.inhibitory_to_excitatory << '\n'
                  << "I -> I: " << stats.inhibitory_to_inhibitory << '\n'
                  << '\n'
                  << "Distributions\n"
                  << "-------------\n";

        print_distribution("Outgoing degree", stats.outgoing_degree);
        print_distribution("Incoming degree", stats.incoming_degree);
        print_distribution("Synaptic weight", stats.weights);
        print_distribution("Synaptic delay", stats.delays);

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Topology inspection failed: "
                  << error.what() << '\n';
        return 1;
    }
}
