cmake_minimum_required(VERSION 3.24)

project(SpikingModel VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)

add_custom_target(check_parameter_schema
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_parameters.py --check
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking generated C++ parameter definitions"
)

include(CTest)

add_library(neuron_analysis STATIC
    src/analysis/izhikevich_analysis.cpp
)
target_include_directories(neuron_analysis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuron_analysis PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

if(BUILD_TESTING)
    add_executable(izhikevich_analysis_tests
        tests/unit/izhikevich_analysis_tests.cpp
    )
    target_link_libraries(izhikevich_analysis_tests PRIVATE neuron_analysis)
    add_test(NAME izhikevich_analysis_tests
             COMMAND izhikevich_analysis_tests)
endif()
if(BUILD_TESTING)
    add_executable(parameter_tests tests/unit/parameter_tests.cpp)
    target_include_directories(parameter_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_dependencies(parameter_tests check_parameter_schema)
    add_test(NAME parameter_tests COMMAND parameter_tests)

   add_executable(ipc_parameter_json_tests
       tests/unit/ipc_parameter_json_tests.cpp
       src/ipc/parameter_json.cpp
   )
   target_include_directories(ipc_parameter_json_tests PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/include
   )
   target_link_libraries(ipc_parameter_json_tests PRIVATE
       nlohmann_json::nlohmann_json
   )
   add_dependencies(ipc_parameter_json_tests check_parameter_schema)
   add_test(NAME ipc_parameter_json_tests COMMAND ipc_parameter_json_tests)

    add_executable(topology_tests
        tests/unit/topology_tests.cpp
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(topology_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_dependencies(topology_tests check_parameter_schema)
    add_test(NAME topology_tests COMMAND topology_tests)

    add_executable(active_neuron_telemetry_tests
        tests/unit/active_neuron_telemetry_tests.cpp
    )
    target_include_directories(active_neuron_telemetry_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME active_neuron_telemetry_tests COMMAND active_neuron_telemetry_tests)

    add_executable(stdp_tests tests/unit/stdp_tests.cpp)
    target_include_directories(stdp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME stdp_tests COMMAND stdp_tests)

    add_executable(stdp_weight_analysis_tests
        tests/unit/stdp_weight_analysis_tests.cpp
    )
    target_include_directories(stdp_weight_analysis_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME stdp_weight_analysis_tests COMMAND stdp_weight_analysis_tests)

    add_executable(stdp_timing_optimizer_tests
        tests/unit/stdp_timing_optimizer_tests.cpp
    )
    target_include_directories(stdp_timing_optimizer_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME stdp_timing_optimizer_tests COMMAND stdp_timing_optimizer_tests)


    add_executable(stdp_experiment tools/stdp_experiment.cpp)
    target_include_directories(stdp_experiment PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(stdp_experiment PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
endif()

# The mathematical STDP timing fitter is useful independently of the
# unit-test suite, so keep it available when BUILD_TESTING=OFF.
add_executable(stdp_timing_optimizer experiments/stdp_timing_optimizer.cpp)
target_include_directories(stdp_timing_optimizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(stdp_timing_optimizer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

option(SPIKINGMODEL_ENABLE_QT_CLIENT "Build the optional Qt IPC client" OFF)
if(SPIKINGMODEL_ENABLE_QT_CLIENT)
    find_package(Qt6 6.4 REQUIRED COMPONENTS Core Network Qml Quick)

    add_executable(spiking_model_qt_client
        src/qt/qt_ipc_probe.cpp
    )
    target_link_libraries(spiking_model_qt_client PRIVATE
        Qt6::Core
        Qt6::Network
    )
    target_compile_options(spiking_model_qt_client PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )

    add_executable(spiking_model_qt_async_client
        include/qt/simulation_client.h
        src/qt/simulation_client.cpp
        src/qt/qt_async_client_probe.cpp
    )
    target_include_directories(spiking_model_qt_async_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(spiking_model_qt_async_client PRIVATE
        Qt6::Core
        Qt6::Network
    )
    target_compile_options(spiking_model_qt_async_client PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(spiking_model_qt_async_client PROPERTIES
        AUTOMOC ON
    )

    add_executable(spiking_model_qt_telemetry_probe
        include/qt/simulation_client.h
        src/qt/simulation_client.cpp
        src/qt/qt_telemetry_probe.cpp
    )
    target_include_directories(spiking_model_qt_telemetry_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(spiking_model_qt_telemetry_probe PRIVATE
        Qt6::Core
        Qt6::Network
    )
    target_compile_options(spiking_model_qt_telemetry_probe PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(spiking_model_qt_telemetry_probe PROPERTIES
        AUTOMOC ON
    )

    add_executable(spiking_model_dashboard
        include/qt/simulation_client.h
        include/qt/simulation_bridge.h
        src/qt/simulation_client.cpp
        src/qt/simulation_bridge.cpp
        src/runtime/graft_package.cpp
        src/topology/topology_io.cpp
        src/qt/qt_dashboard_main.cpp
    )
    target_include_directories(spiking_model_dashboard PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(spiking_model_dashboard PRIVATE
        Qt6::Core
        Qt6::Network
        Qt6::Qml
        Qt6::Quick
        nlohmann_json::nlohmann_json
    )
    target_compile_options(spiking_model_dashboard PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(spiking_model_dashboard PROPERTIES
        AUTOMOC ON
    )

    qt_add_resources(spiking_model_dashboard dashboard_qml
        PREFIX "/"
        FILES
            qml/Main.qml
    )

    # V11: local desktop capture frontend. This milestone is observe-only;
    # it does not inject mouse or keyboard events.
    add_executable(neuromancer_desktop_agent
        include/qt/simulation_client.h
        include/qt/simulation_bridge.h
        include/qt/desktop_capture.h
        src/qt/simulation_client.cpp
        src/qt/simulation_bridge.cpp
        src/qt/desktop_capture.cpp
        src/qt/desktop_agent_main.cpp
    )
    target_include_directories(neuromancer_desktop_agent PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_desktop_agent PRIVATE
        Qt6::Core
        Qt6::Network
        Qt6::Qml
        Qt6::Quick
    )
    target_compile_options(neuromancer_desktop_agent PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_desktop_agent PROPERTIES
        AUTOMOC ON
    )
    qt_add_resources(neuromancer_desktop_agent desktop_agent_qml
        PREFIX "/"
        FILES
            qml/DesktopAgent.qml
    )
endif()

add_executable(topology_inspect
    src/topology/topology_inspect.cpp
    src/topology/topology_io.cpp
)
target_include_directories(topology_inspect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(topology_inspect PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

option(SPIKINGMODEL_ENABLE_CUDA "Build the CUDA simulation executable" ON)
if(SPIKINGMODEL_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to compile")
    find_package(CUDAToolkit REQUIRED)

    add_executable(spiking_model
        src/main.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/runtime/graft_topology.cpp
        src/runtime/simulation_controller.cpp
        src/runtime/neural_self_observer.cpp
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
        src/ipc/parameter_json.cpp
        src/ipc/protocol.cpp
        src/ipc/unix_socket_server.cpp
    )
    target_include_directories(spiking_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(spiking_model PRIVATE
        CUDA::cudart
        Threads::Threads
        nlohmann_json::nlohmann_json
    )
    target_compile_options(spiking_model PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    target_compile_definitions(spiking_model PRIVATE SPIKINGMODEL_VERSION="${PROJECT_VERSION}")
    target_compile_options(spiking_model PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(spiking_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_dependencies(spiking_model check_parameter_schema)

    # Neuromancer phase-timing experiment: executes the real CUDA runtime,
    # external-current scheduler, CSR propagation, and spike-event recorder.
    add_executable(neuromancer_phase_runtime
        experiments/neuromancer_phase_runtime.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_phase_runtime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_phase_runtime PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_phase_runtime PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_phase_runtime PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_phase_runtime COMMAND neuromancer_phase_runtime)

    # V9: real recurrent recruitment with an optional GPU-side postsynaptic
    # phase gate. V8 remains an independent, unchanged baseline target.
    add_executable(neuromancer_recurrent_phase_v9
        experiments/neuromancer_recurrent_phase_v9.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_recurrent_phase_v9 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_recurrent_phase_v9 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_recurrent_phase_v9 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_recurrent_phase_v9 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_recurrent_phase_v9
             COMMAND neuromancer_recurrent_phase_v9)
    set_tests_properties(neuromancer_recurrent_phase_v9 PROPERTIES TIMEOUT 120)

    # V15: two STDP associations learned in one shared CSR runtime and
    # retrieved from partial cues with frozen weights; includes an untrained
    # paired-path control and a plasticity-disabled control.
    add_executable(neuromancer_associative_memory_v15
        experiments/neuromancer_associative_memory_v15.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_associative_memory_v15 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_associative_memory_v15 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_associative_memory_v15 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_associative_memory_v15 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_associative_memory_v15
             COMMAND neuromancer_associative_memory_v15)
    set_tests_properties(neuromancer_associative_memory_v15 PROPERTIES TIMEOUT 180)

    # V16: STDP training exported to a new plasticity-disabled runtime;
    # measure recruitment across 25%, 50%, 75%, and 100% cue sizes.
    add_executable(neuromancer_associative_memory_v16
        experiments/neuromancer_associative_memory_v16.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_associative_memory_v16 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_associative_memory_v16 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_associative_memory_v16 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_associative_memory_v16 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_associative_memory_v16
             COMMAND neuromancer_associative_memory_v16)
    set_tests_properties(neuromancer_associative_memory_v16 PROPERTIES TIMEOUT 180)

    # V17: sparse randomized CSR association blocks, STDP training,
    # exported frozen weights, and cue-size recruitment measurements.
    add_executable(neuromancer_associative_memory_v17
        experiments/neuromancer_associative_memory_v17.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_associative_memory_v17 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_associative_memory_v17 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_associative_memory_v17 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_associative_memory_v17 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_associative_memory_v17
             COMMAND neuromancer_associative_memory_v17)
    set_tests_properties(neuromancer_associative_memory_v17 PROPERTIES TIMEOUT 180)

    # V18: global random sparse E->E graph; association labels are
    # measurement-only and are not used to generate connectivity.
    add_executable(neuromancer_random_topology_association_v18
        experiments/neuromancer_random_topology_association_v18.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_random_topology_association_v18 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_random_topology_association_v18 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_random_topology_association_v18 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_random_topology_association_v18 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_random_topology_association_v18
             COMMAND neuromancer_random_topology_association_v18)
    set_tests_properties(neuromancer_random_topology_association_v18 PROPERTIES TIMEOUT 180)

    # V27: non-plastic E/I diagnostics on an existing saved CSR topology.
    # Compares spontaneous activity, E-driven inhibitory recruitment, and
    # direct inhibitory stimulation without changing the supplied model.
    add_executable(neuromancer_ei_diagnostics_v27
        experiments/neuromancer_ei_diagnostics_v27.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_ei_diagnostics_v27 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_ei_diagnostics_v27 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_ei_diagnostics_v27 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_ei_diagnostics_v27 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V28: parameterized candidate evaluator for multi-run CSR/STDP optimization.
    add_executable(neuromancer_csr_stdp_candidate_v28
        experiments/neuromancer_csr_stdp_candidate_v28.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_csr_stdp_candidate_v28 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_csr_stdp_candidate_v28 PRIVATE
        CUDA::cudart nlohmann_json::nlohmann_json
    )
    target_compile_options(neuromancer_csr_stdp_candidate_v28 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_csr_stdp_candidate_v28 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V31: phase-relative E/I interaction with matched topology controls.
    add_executable(neuromancer_phase_ei_interaction_v31
        experiments/neuromancer_phase_ei_interaction_v31.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_phase_ei_interaction_v31 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_phase_ei_interaction_v31 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_phase_ei_interaction_v31 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_phase_ei_interaction_v31 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V33: matched E/I recruitment competition near the excitatory threshold.
    add_executable(neuromancer_ei_competition_v33
        experiments/neuromancer_ei_competition_v33.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_ei_competition_v33 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_ei_competition_v33 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_ei_competition_v33 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_ei_competition_v33 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V34: source-selective phase gating and spike-free E/I interaction assay.
    add_executable(neuromancer_phase_selective_inhibition_v34
        experiments/neuromancer_phase_selective_inhibition_v34.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_phase_selective_inhibition_v34 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_phase_selective_inhibition_v34 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_phase_selective_inhibition_v34 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_phase_selective_inhibition_v34 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V35: map inhibitory-only phase gating across a full oscillator cycle.
    add_executable(neuromancer_inhibitory_phase_response_v35
        experiments/neuromancer_inhibitory_phase_response_v35.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_inhibitory_phase_response_v35 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_inhibitory_phase_response_v35 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_inhibitory_phase_response_v35 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_inhibitory_phase_response_v35 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V37: closed-loop Tetris benchmark with random, heuristic, and CUDA neural controllers.
    add_executable(neuromancer_tetris_v37
        experiments/neuromancer_tetris_v37.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_tetris_v37 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_tetris_v37 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_tetris_v37 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_tetris_v37 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )


    # V38: closed-loop Tetris benchmark with random, heuristic, and CUDA neural controllers.
    add_executable(neuromancer_tetris_v38
        experiments/neuromancer_tetris_v38.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_tetris_v38 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_tetris_v38 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_tetris_v38 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_tetris_v38 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )


    # V41: closed-loop Tetris benchmark with random, heuristic, and CUDA neural controllers.
    add_executable(neuromancer_tetris_v41
        experiments/neuromancer_tetris_v41.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_tetris_v41 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_tetris_v41 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_tetris_v41 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_tetris_v41 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V36: test phase-gate robustness across widths and frequencies.
    add_executable(neuromancer_inhibitory_phase_robustness_v36
        experiments/neuromancer_phase_gate_robustness_v36.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_inhibitory_phase_robustness_v36 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_inhibitory_phase_robustness_v36 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_inhibitory_phase_robustness_v36 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_inhibitory_phase_robustness_v36 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V32: phase-gated recruitment, E/I strength and timing sweep, shifted-phase control.
    add_executable(neuromancer_phase_recruitment_v32
        experiments/neuromancer_phase_recruitment_v32.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_phase_recruitment_v32 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_phase_recruitment_v32 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_phase_recruitment_v32 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_phase_recruitment_v32 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V30: phase-gated excitatory and inhibitory CSR transmission assay.
    add_executable(neuromancer_phase_inhibitory_v30
        experiments/neuromancer_phase_inhibitory_v30.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_phase_inhibitory_v30 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_phase_inhibitory_v30 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_phase_inhibitory_v30 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_phase_inhibitory_v30 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V29: comparative topology growth, inhibitory regulation and Markov telemetry.
    add_executable(neuromancer_topology_candidate_v29
        experiments/neuromancer_topology_candidate_v29.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_topology_candidate_v29 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_topology_candidate_v29 PRIVATE
        CUDA::cudart nlohmann_json::nlohmann_json
    )
    target_compile_options(neuromancer_topology_candidate_v29 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_topology_candidate_v29 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # V26: overnight continual-learning experiment with online STDP,
    # periodic learned-CSR checkpoints, held-out frozen-weight evaluation.
    add_executable(neuromancer_overnight_v26
        experiments/neuromancer_overnight_v26.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_overnight_v26 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_overnight_v26 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_overnight_v26 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_overnight_v26 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_overnight_v26_smoke
             COMMAND neuromancer_overnight_v26 --smoke-test
                     ${CMAKE_CURRENT_BINARY_DIR}/neuromancer_v26_smoke)
    set_tests_properties(neuromancer_overnight_v26_smoke PROPERTIES TIMEOUT 600)

    # V24: reusable inference API and command-line runner.
    add_library(neuromancer_inference
        src/neuromancer/inference_engine.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_inference PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_inference PUBLIC CUDA::cudart)
    target_compile_options(neuromancer_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_inference PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    add_executable(neuromancer_infer apps/neuromancer_infer.cpp)
    target_link_libraries(neuromancer_infer PRIVATE neuromancer_inference)
    add_test(NAME neuromancer_model_package_v25
             COMMAND neuromancer_infer --self-test ${CMAKE_CURRENT_BINARY_DIR}/neuromancer_v23_model/model.nmc)
    set_tests_properties(neuromancer_model_package_v25 PROPERTIES
        TIMEOUT 180
        FIXTURES_REQUIRED neuromancer_v23_model
    )
    # V23: two-class spiking inference with a trained, frozen CSR readout.
    add_executable(neuromancer_inference_prototype_v23
        experiments/neuromancer_inference_prototype_v23.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_inference_prototype_v23 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_inference_prototype_v23 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_inference_prototype_v23 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_inference_prototype_v23 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_inference_prototype_v23
             COMMAND neuromancer_inference_prototype_v23 ${CMAKE_CURRENT_BINARY_DIR}/neuromancer_v23_model)
    set_tests_properties(neuromancer_inference_prototype_v23 PROPERTIES TIMEOUT 180)
    set_tests_properties(neuromancer_inference_prototype_v23 PROPERTIES
        FIXTURES_SETUP neuromancer_v23_model
    )

    # V22: matched disconnected-target control and event-aligned synaptic transfer curve.
    add_executable(neuromancer_transmission_audit_v22
        experiments/neuromancer_transmission_audit_v22.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_transmission_audit_v22 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_transmission_audit_v22 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_transmission_audit_v22 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_transmission_audit_v22 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_transmission_audit_v22
             COMMAND neuromancer_transmission_audit_v22)
    set_tests_properties(neuromancer_transmission_audit_v22 PROPERTIES TIMEOUT 180)

    # V21: external-current and isolated synaptic impulse-response identification.
    add_executable(neuromancer_neuron_dynamics_identification_v21
        experiments/neuromancer_neuron_dynamics_identification_v21.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_neuron_dynamics_identification_v21 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_neuron_dynamics_identification_v21 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_neuron_dynamics_identification_v21 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_neuron_dynamics_identification_v21 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_neuron_dynamics_identification_v21
             COMMAND neuromancer_neuron_dynamics_identification_v21)
    set_tests_properties(neuromancer_neuron_dynamics_identification_v21 PROPERTIES TIMEOUT 180)

    # V20: isolated positive/negative pre-post lag sweep to identify the STDP kernel.
    add_executable(neuromancer_stdp_kernel_calibration_v20
        experiments/neuromancer_stdp_kernel_calibration_v20.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_stdp_kernel_calibration_v20 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_stdp_kernel_calibration_v20 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_stdp_kernel_calibration_v20 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_stdp_kernel_calibration_v20 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_stdp_kernel_calibration_v20
             COMMAND neuromancer_stdp_kernel_calibration_v20)
    set_tests_properties(neuromancer_stdp_kernel_calibration_v20 PROPERTIES TIMEOUT 180)

    # V19: edge-wise STDP deltas and stepwise frozen-recall voltage traces.
    add_executable(neuromancer_system_identification_v19
        experiments/neuromancer_system_identification_v19.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_system_identification_v19 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_system_identification_v19 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_system_identification_v19 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_system_identification_v19 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_system_identification_v19
             COMMAND neuromancer_system_identification_v19)
    set_tests_properties(neuromancer_system_identification_v19 PROPERTIES TIMEOUT 180)

    # V10: causal pair-STDP association learning followed by frozen,
    # phase-gated cue-to-target recall. V8 and V9 remain unchanged.
    add_executable(neuromancer_stdp_learning_v10
        experiments/neuromancer_stdp_learning_v10.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
        src/topology/topology_io.cpp
    )
    target_include_directories(neuromancer_stdp_learning_v10 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neuromancer_stdp_learning_v10 PRIVATE CUDA::cudart)
    target_compile_options(neuromancer_stdp_learning_v10 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neuromancer_stdp_learning_v10 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_test(NAME neuromancer_stdp_learning_v10
             COMMAND neuromancer_stdp_learning_v10)
    set_tests_properties(neuromancer_stdp_learning_v10 PROPERTIES TIMEOUT 120)

    if(BUILD_TESTING)
        add_executable(stdp_cuda_tests
            tests/cuda/stdp_cuda_tests.cpp
            src/cuda/neuron_izhikevich.cu
        )
        target_include_directories(stdp_cuda_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(stdp_cuda_tests PRIVATE CUDA::cudart)
        set_target_properties(stdp_cuda_tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME stdp_cuda_tests COMMAND stdp_cuda_tests)

        add_executable(external_stimulation_cuda_tests
            tests/cuda/external_stimulation_cuda_tests.cpp
            src/cuda/neuron_izhikevich.cu
        )
        target_include_directories(external_stimulation_cuda_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(external_stimulation_cuda_tests PRIVATE CUDA::cudart)
        set_target_properties(external_stimulation_cuda_tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME external_stimulation_cuda_tests
                 COMMAND external_stimulation_cuda_tests)

        add_executable(cuda_runtime_smoke
            tests/cuda/runtime_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
        )
        target_include_directories(cuda_runtime_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(cuda_runtime_smoke PRIVATE CUDA::cudart)
        set_target_properties(cuda_runtime_smoke PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME cuda_runtime_smoke COMMAND cuda_runtime_smoke)
        add_dependencies(cuda_runtime_smoke check_parameter_schema)

        add_executable(network_integration_tests
            tests/cuda/network_integration_tests.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
        )
        target_include_directories(network_integration_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(network_integration_tests PRIVATE CUDA::cudart)
        set_target_properties(network_integration_tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME network_integration_tests COMMAND network_integration_tests)
        set_tests_properties(network_integration_tests PROPERTIES TIMEOUT 120)
        add_dependencies(network_integration_tests check_parameter_schema)

        add_executable(determinism_smoke
            tests/cuda/determinism_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
        )
        target_include_directories(determinism_smoke PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(determinism_smoke PRIVATE CUDA::cudart)
        set_target_properties(determinism_smoke PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME determinism_smoke COMMAND determinism_smoke)
        set_tests_properties(determinism_smoke PROPERTIES TIMEOUT 120)
        add_dependencies(determinism_smoke check_parameter_schema)

        add_executable(controller_smoke
            tests/cuda/controller_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
        )
        target_include_directories(controller_smoke PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(controller_smoke PRIVATE CUDA::cudart)
        set_target_properties(controller_smoke PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME controller_smoke COMMAND controller_smoke)
        add_dependencies(controller_smoke check_parameter_schema)

        add_executable(stdp_associative_recall
            experiments/stdp_associative_recall.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
        )
        target_include_directories(stdp_associative_recall PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(stdp_associative_recall PRIVATE
            CUDA::cudart nlohmann_json::nlohmann_json
        )
        set_target_properties(stdp_associative_recall PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME stdp_associative_recall COMMAND stdp_associative_recall)
        set_tests_properties(stdp_associative_recall PROPERTIES TIMEOUT 120)
        add_dependencies(stdp_associative_recall check_parameter_schema)

        add_executable(stdp_recurrent_association
            experiments/stdp_recurrent_association.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
        )
        target_include_directories(stdp_recurrent_association PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(stdp_recurrent_association PRIVATE
            CUDA::cudart
        )
        set_target_properties(stdp_recurrent_association PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_dependencies(stdp_recurrent_association check_parameter_schema)

        add_executable(stdp_parameter_sweep
            experiments/stdp_parameter_sweep.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/topology/topology_generator.cpp
            src/topology/topology_io.cpp
            src/ipc/parameter_json.cpp
        )
        target_include_directories(stdp_parameter_sweep PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(stdp_parameter_sweep PRIVATE
            CUDA::cudart Threads::Threads nlohmann_json::nlohmann_json
        )
        set_target_properties(stdp_parameter_sweep PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_dependencies(stdp_parameter_sweep check_parameter_schema)

        add_executable(cpu_gpu_neuron_validation
            tests/cuda/cpu_gpu_neuron_validation.cpp
            src/cuda/neuron_izhikevich.cu
            src/analysis/izhikevich_analysis.cpp
        )
        target_include_directories(cpu_gpu_neuron_validation PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(cpu_gpu_neuron_validation PRIVATE CUDA::cudart)
        set_target_properties(cpu_gpu_neuron_validation PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(
            NAME cpu_gpu_neuron_validation
            COMMAND cpu_gpu_neuron_validation
        )

        add_test(
            NAME ipc_server_integration
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/ipc_server_test.py
                $<TARGET_FILE:spiking_model>
        )
        set_tests_properties(ipc_server_integration PROPERTIES TIMEOUT 30)
    endif()
endif()


if(BUILD_TESTING)
    add_executable(phase_plane_tests
        tests/unit/phase_plane_tests.cpp
    )
    target_link_libraries(phase_plane_tests PRIVATE neuron_analysis)
    add_test(NAME phase_plane_tests COMMAND phase_plane_tests)
endif()

if(BUILD_TESTING)
    add_executable(trajectory_tests
        tests/unit/trajectory_tests.cpp
    )
    target_link_libraries(trajectory_tests PRIVATE neuron_analysis)
    add_test(NAME trajectory_tests COMMAND trajectory_tests)
endif()

# V42 deterministic host-side learning primitive smoke test. The integrated
# CUDA experiment will be registered after the runtime phase interface lands.
add_executable(neuromancer_learning_primitives_v42
    experiments/neuromancer_learning_primitives_v42.cpp
)
target_include_directories(neuromancer_learning_primitives_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuromancer_learning_primitives_v42 PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_learning_primitives_v42
             COMMAND neuromancer_learning_primitives_v42)
endif()

# V42 CUDA integration test: host-updated endogenous phase state controls
# relative-phase synaptic transmission in the existing CUDA propagation kernel.
add_executable(neuromancer_dynamic_phase_runtime_v42
    experiments/neuromancer_dynamic_phase_runtime_v42.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_dynamic_phase_runtime_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_dynamic_phase_runtime_v42 PRIVATE CUDA::cudart)
set_target_properties(neuromancer_dynamic_phase_runtime_v42 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# V42 controlled CUDA phase-transfer controls: ungated, aligned, and
# misaligned conditions share identical topology and source stimulation.
add_executable(neuromancer_phase_transfer_controls_v42
    experiments/neuromancer_phase_transfer_controls_v42.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_phase_transfer_controls_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_phase_transfer_controls_v42 PRIVATE CUDA::cudart)
set_target_properties(neuromancer_phase_transfer_controls_v42 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_phase_transfer_controls_v42
             COMMAND neuromancer_phase_transfer_controls_v42)
    set_tests_properties(neuromancer_phase_transfer_controls_v42 PROPERTIES TIMEOUT 180)
endif()

# V42 end-to-end test: learn a temporal lag, freeze it, then apply that
# measured lag to the CUDA synaptic phase gate.
add_executable(neuromancer_learned_phase_cuda_transfer_v42
    experiments/neuromancer_learned_phase_cuda_transfer_v42.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_learned_phase_cuda_transfer_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_learned_phase_cuda_transfer_v42 PRIVATE CUDA::cudart)
set_target_properties(neuromancer_learned_phase_cuda_transfer_v42 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_learned_phase_cuda_transfer_v42
             COMMAND neuromancer_learned_phase_cuda_transfer_v42)
    set_tests_properties(neuromancer_learned_phase_cuda_transfer_v42 PROPERTIES TIMEOUT 180)
endif()

# V42 endogenous phase learning, negative-control, and frozen-retention diagnostic.
add_executable(neuromancer_autonomous_phase_learning_v42
    experiments/neuromancer_autonomous_phase_learning_v42.cpp
)
target_include_directories(neuromancer_autonomous_phase_learning_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_autonomous_phase_learning_v42
             COMMAND neuromancer_autonomous_phase_learning_v42)
    set_tests_properties(neuromancer_autonomous_phase_learning_v42 PROPERTIES TIMEOUT 60)
endif()


# V42 integrated factorial implementation gate. This deterministic host-side
# reference combines temporal phase learning, activity-driven CSR rewiring,
# pair-based STDP, and frozen partial-cue graph recall before CUDA recall is added.
add_executable(neuromancer_integrated_learning_gate_v42
    experiments/neuromancer_integrated_learning_gate_v42.cpp
)
target_include_directories(neuromancer_integrated_learning_gate_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuromancer_integrated_learning_gate_v42 PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_integrated_learning_gate_v42
             COMMAND neuromancer_integrated_learning_gate_v42)
    set_tests_properties(neuromancer_integrated_learning_gate_v42 PROPERTIES TIMEOUT 120)
endif()


# V42 isolated feasibility assay for autonomous population phase learning and
# activity-dependent CSR structural synaptic plasticity (host-side primitives).
add_executable(neuromancer_phase_structural_feasibility_v42
    experiments/neuromancer_phase_structural_feasibility_v42.cpp
)
target_include_directories(neuromancer_phase_structural_feasibility_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuromancer_phase_structural_feasibility_v42 PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_phase_structural_feasibility_v42
             COMMAND neuromancer_phase_structural_feasibility_v42)
    set_tests_properties(neuromancer_phase_structural_feasibility_v42 PROPERTIES TIMEOUT 60)
endif()


# V42 random, initially unprewired association discovery with phase and
# activity-dependent structural synaptic plasticity (host-side reference).
add_executable(neuromancer_unprewired_discovery_v42
    experiments/neuromancer_unprewired_discovery_v42.cpp
)
target_include_directories(neuromancer_unprewired_discovery_v42 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuromancer_unprewired_discovery_v42 PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_unprewired_discovery_v42
             COMMAND neuromancer_unprewired_discovery_v42)
    set_tests_properties(neuromancer_unprewired_discovery_v42 PROPERTIES TIMEOUT 120)
endif()

# V43: integrated autonomous learning reference plus live CUDA spike-to-phase
# feedback and dynamic synaptic phase gating.
add_executable(neuromancer_integrated_autonomous_learning_v43
    experiments/neuromancer_integrated_autonomous_learning_v43.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_integrated_autonomous_learning_v43 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_integrated_autonomous_learning_v43 PRIVATE CUDA::cudart)
target_compile_options(neuromancer_integrated_autonomous_learning_v43 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_integrated_autonomous_learning_v43 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_integrated_autonomous_learning_v43
             COMMAND neuromancer_integrated_autonomous_learning_v43)
    set_tests_properties(neuromancer_integrated_autonomous_learning_v43 PROPERTIES TIMEOUT 240)
endif()


# V44: learn temporal phase from live CUDA spike events, freeze learning,
# then causally test whether the learned lag controls CUDA transmission.
add_executable(neuromancer_learned_cuda_phase_causality_v44
    experiments/neuromancer_learned_cuda_phase_causality_v44.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_learned_cuda_phase_causality_v44 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_learned_cuda_phase_causality_v44 PRIVATE CUDA::cudart)
target_compile_options(neuromancer_learned_cuda_phase_causality_v44 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_learned_cuda_phase_causality_v44 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_learned_cuda_phase_causality_v44
             COMMAND neuromancer_learned_cuda_phase_causality_v44)
    set_tests_properties(neuromancer_learned_cuda_phase_causality_v44 PROPERTIES TIMEOUT 240)
endif()


# V45: CUDA STDP training on a globally random CSR graph, event-driven phase
# learning, activity-dependent host CSR remodeling, and frozen fresh-runtime
# partial-cue recall. Structural/phase controllers remain host-side explicitly.
add_executable(neuromancer_cuda_associative_recall_v45
    experiments/neuromancer_cuda_associative_recall_v45.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_cuda_associative_recall_v45 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_cuda_associative_recall_v45 PRIVATE CUDA::cudart)
target_compile_options(neuromancer_cuda_associative_recall_v45 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_cuda_associative_recall_v45 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_cuda_associative_recall_v45
             COMMAND neuromancer_cuda_associative_recall_v45)
    set_tests_properties(neuromancer_cuda_associative_recall_v45 PROPERTIES TIMEOUT 300)
endif()


# V46: staged positive/negative controls for single-synapse CUDA transmission,
# assembly recruitment, and an engineered A->B->C propagation chain.
add_executable(neuromancer_cuda_transmission_diagnostics_v46
    experiments/neuromancer_cuda_transmission_diagnostics_v46.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_cuda_transmission_diagnostics_v46 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_cuda_transmission_diagnostics_v46 PRIVATE CUDA::cudart)
target_compile_options(neuromancer_cuda_transmission_diagnostics_v46 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_cuda_transmission_diagnostics_v46 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_cuda_transmission_diagnostics_v46
             COMMAND neuromancer_cuda_transmission_diagnostics_v46)
    set_tests_properties(neuromancer_cuda_transmission_diagnostics_v46 PROPERTIES TIMEOUT 300)
endif()


# V47: calibrate direct B->C recruitment and engineered A->B->C chains
# after V46 identified a subthreshold second-hop response.
add_executable(neuromancer_two_hop_threshold_calibration_v47
    experiments/neuromancer_two_hop_threshold_calibration_v47.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_two_hop_threshold_calibration_v47 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_two_hop_threshold_calibration_v47 PRIVATE CUDA::cudart)
target_compile_options(neuromancer_two_hop_threshold_calibration_v47 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_two_hop_threshold_calibration_v47 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_two_hop_threshold_calibration_v47
             COMMAND neuromancer_two_hop_threshold_calibration_v47)
    set_tests_properties(neuromancer_two_hop_threshold_calibration_v47 PROPERTIES TIMEOUT 300)
endif()


# V48: dynamically loaded runtime module and mixed frozen/open per-edge STDP.
add_library(neuromancer_edge_policy_module_v48 SHARED
    plugins/edge_policy_module.cpp
)
target_include_directories(neuromancer_edge_policy_module_v48 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(neuromancer_edge_policy_module_v48 PROPERTIES
    PREFIX ""
    OUTPUT_NAME "neuromancer_edge_policy_v48"
)

add_executable(neuromancer_hot_module_mixed_plasticity_v48
    experiments/neuromancer_hot_module_mixed_plasticity_v48.cpp
    src/runtime/module_manager.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_hot_module_mixed_plasticity_v48 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_hot_module_mixed_plasticity_v48 PRIVATE
    CUDA::cudart ${CMAKE_DL_LIBS}
)
target_compile_options(neuromancer_hot_module_mixed_plasticity_v48 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_hot_module_mixed_plasticity_v48 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_dependencies(neuromancer_hot_module_mixed_plasticity_v48
    neuromancer_edge_policy_module_v48
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_hot_module_mixed_plasticity_v48
        COMMAND neuromancer_hot_module_mixed_plasticity_v48
            $<TARGET_FILE:neuromancer_edge_policy_module_v48>
            ${CMAKE_CURRENT_BINARY_DIR}/v48_hot_module_test
    )
    set_tests_properties(neuromancer_hot_module_mixed_plasticity_v48 PROPERTIES TIMEOUT 300)
endif()


# V49: matched CUDA open/frozen/no-edge transmission controls. The frozen
# condition attaches and unloads the V48 policy module before source-only recall.
add_executable(neuromancer_frozen_edge_transmission_v49
    experiments/neuromancer_frozen_edge_transmission_v49.cpp
    src/runtime/module_manager.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_frozen_edge_transmission_v49 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_frozen_edge_transmission_v49 PRIVATE
    CUDA::cudart ${CMAKE_DL_LIBS}
)
target_compile_options(neuromancer_frozen_edge_transmission_v49 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_frozen_edge_transmission_v49 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_dependencies(neuromancer_frozen_edge_transmission_v49
    neuromancer_edge_policy_module_v48
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_frozen_edge_transmission_v49
        COMMAND neuromancer_frozen_edge_transmission_v49
            $<TARGET_FILE:neuromancer_edge_policy_module_v48>
            ${CMAKE_CURRENT_BINARY_DIR}/v49_frozen_edge_test
    )
    set_tests_properties(neuromancer_frozen_edge_transmission_v49 PROPERTIES TIMEOUT 300)
endif()


# V50: phase-gated transmission controls the postsynaptic spike events available
# to CUDA STDP; aligned/misaligned and open/frozen conditions are compared.
add_executable(neuromancer_phase_controlled_plasticity_v50
    experiments/neuromancer_phase_controlled_plasticity_v50.cpp
    src/runtime/module_manager.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_phase_controlled_plasticity_v50 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_phase_controlled_plasticity_v50 PRIVATE
    CUDA::cudart ${CMAKE_DL_LIBS}
)
target_compile_options(neuromancer_phase_controlled_plasticity_v50 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_phase_controlled_plasticity_v50 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_dependencies(neuromancer_phase_controlled_plasticity_v50
    neuromancer_edge_policy_module_v48
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_phase_controlled_plasticity_v50
        COMMAND neuromancer_phase_controlled_plasticity_v50
            $<TARGET_FILE:neuromancer_edge_policy_module_v48>
            ${CMAKE_CURRENT_BINARY_DIR}/v50_phase_plasticity_test
    )
    set_tests_properties(neuromancer_phase_controlled_plasticity_v50 PROPERTIES TIMEOUT 360)
endif()

# V51: consolidated phase/STDP reference, autonomous phase, structural
# plasticity, and CUDA inhibitory competition validation batch.
add_executable(neuromancer_validation_batch_v51
    experiments/neuromancer_validation_batch_v51.cpp
    src/cuda/neuron_izhikevich.cu
    src/runtime/cuda_simulation_runtime.cu
    src/topology/topology_generator.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_validation_batch_v51 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_validation_batch_v51 PRIVATE CUDA::cudart)
target_compile_options(neuromancer_validation_batch_v51 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(neuromancer_validation_batch_v51 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
if(BUILD_TESTING)
    add_test(NAME neuromancer_validation_batch_v51
        COMMAND neuromancer_validation_batch_v51
            ${CMAKE_CURRENT_BINARY_DIR}/v51_validation_batch_test
    )
    set_tests_properties(neuromancer_validation_batch_v51 PROPERTIES TIMEOUT 360)
endif()


# Neuromancer graft package inspector: validates manifest, package-local paths,
# and SPKCSR topology without loading executable module code.
add_library(neuromancer_graft_package STATIC
    src/runtime/graft_package.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_graft_package PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_graft_package PUBLIC
    nlohmann_json::nlohmann_json
)
target_compile_options(neuromancer_graft_package PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

# The live backend uses the package inspector when attaching grafts.
if(TARGET spiking_model)
    target_link_libraries(spiking_model PRIVATE neuromancer_graft_package)
endif()

add_executable(neuromancer_graft_inspect
    tools/neuromancer_graft_inspect.cpp
)
target_link_libraries(neuromancer_graft_inspect PRIVATE neuromancer_graft_package)

if(BUILD_TESTING)
    add_executable(graft_package_tests tests/unit/graft_package_tests.cpp)
    target_link_libraries(graft_package_tests PRIVATE neuromancer_graft_package)
    add_test(NAME graft_package_tests COMMAND graft_package_tests)
endif()


# Persistent modular neural memory: versioned CSR topology plus resumable state.
add_library(neuromancer_neural_memory STATIC
    src/runtime/neural_memory_store.cpp
    src/topology/topology_io.cpp
)
target_include_directories(neuromancer_neural_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuromancer_neural_memory PUBLIC
    nlohmann_json::nlohmann_json
)
target_compile_options(neuromancer_neural_memory PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

if(BUILD_TESTING)
    add_executable(neural_memory_store_tests
        tests/unit/neural_memory_store_tests.cpp
    )
    target_link_libraries(neural_memory_store_tests PRIVATE neuromancer_neural_memory)
    add_test(NAME neural_memory_store_tests COMMAND neural_memory_store_tests)
endif()

# Graft topology merge: deterministic CSR composition and per-edge plasticity policy.
add_library(neuromancer_graft_topology STATIC
    src/runtime/graft_topology.cpp
)
target_include_directories(neuromancer_graft_topology PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuromancer_graft_topology PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
if(BUILD_TESTING)
    add_executable(graft_topology_tests tests/unit/graft_topology_tests.cpp)
    target_link_libraries(graft_topology_tests PRIVATE neuromancer_graft_topology)
    add_test(NAME graft_topology_tests COMMAND graft_topology_tests)
endif()


# Read-only self-observation of the live connectome snapshot.
add_library(neuromancer_neural_self_observer STATIC
    src/runtime/neural_self_observer.cpp
)
target_include_directories(neuromancer_neural_self_observer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(neuromancer_neural_self_observer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
if(BUILD_TESTING)
    add_executable(neural_self_observer_tests
        tests/unit/neural_self_observer_tests.cpp
    )
    target_link_libraries(neural_self_observer_tests PRIVATE
        neuromancer_neural_self_observer
    )
    add_test(NAME neural_self_observer_tests COMMAND neural_self_observer_tests)
endif()
