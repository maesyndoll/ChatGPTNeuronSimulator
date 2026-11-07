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
        src/runtime/simulation_controller.cpp
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
