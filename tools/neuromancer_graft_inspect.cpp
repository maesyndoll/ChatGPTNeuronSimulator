#include "runtime/graft_package.h"
#include <exception>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: neuromancer_graft_inspect <package-directory-or-manifest>\n";
        return 2;
    }
    try {
        const auto info = spiking::GraftPackageInspector::inspect(argv[1]);
        std::cout << "NEUROMANCER GRAFT PACKAGE\n"
                  << "id=" << info.id << '\n'
                  << "name=" << info.name << '\n'
                  << "description=" << info.description << '\n'
                  << "neurons=" << (info.excitatory_count + info.inhibitory_count)
                  << " excitatory=" << info.excitatory_count
                  << " inhibitory=" << info.inhibitory_count << '\n'
                  << "csr_edges=" << info.synapse_count << '\n'
                  << "module_binary=" << (info.has_module_binary ? "present" : "none") << '\n'
                  << "input_adapters=" << info.input_adapters.size() << '\n'
                  << "output_adapters=" << info.output_adapters.size() << '\n'
                  << "inspection=PASS\nattachment=NOT_PERFORMED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Graft inspection failed: " << error.what() << '\n';
        return 1;
    }
}
