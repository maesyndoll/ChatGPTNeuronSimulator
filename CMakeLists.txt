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
    )
    target_include_directories(topology_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_dependencies(topology_tests check_parameter_schema)
    add_test(NAME topology_tests COMMAND topology_tests)

    add_executable(temporal_embedding_tests
        tests/unit/temporal_embedding_tests.cpp
        src/connectome/temporal_embedding.cpp
    )
    target_include_directories(temporal_embedding_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(temporal_embedding_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    add_test(NAME temporal_embedding_tests COMMAND temporal_embedding_tests)

    add_executable(tissue_model_tests
        tests/unit/tissue_model_tests.cpp
        src/tissue/tissue_model.cpp
    )
    target_include_directories(tissue_model_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME tissue_model_tests COMMAND tissue_model_tests)
endif()

option(SPIKINGMODEL_ENABLE_QT_CLIENT "Build the optional Qt IPC client" OFF)
if(SPIKINGMODEL_ENABLE_QT_CLIENT)
    find_package(Qt6 6.4 REQUIRED COMPONENTS Core Network Qml Quick Concurrent)

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
        include/qt/connectome_field.h
        src/connectome/temporal_embedding.cpp
        src/qt/connectome_field.cpp
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
        Qt6::Concurrent
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
endif()

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
        src/tissue/tissue_model.cpp
        src/topology/topology_generator.cpp
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

    # Standalone, reproducible GPU spike-raster showcase with HTML/CSV output.
    add_executable(neural_observatory
        src/showcase/neural_observatory.cpp
        src/cuda/neuron_izhikevich.cu
        src/runtime/cuda_simulation_runtime.cu
        src/topology/topology_generator.cpp
    )
    target_include_directories(neural_observatory PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(neural_observatory PRIVATE CUDA::cudart)
    target_compile_options(neural_observatory PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    set_target_properties(neural_observatory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    add_dependencies(neural_observatory check_parameter_schema)

    if(BUILD_TESTING)
        add_executable(cuda_runtime_smoke
            tests/cuda/runtime_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/tissue/tissue_model.cpp
            src/topology/topology_generator.cpp
        )
        target_include_directories(cuda_runtime_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(cuda_runtime_smoke PRIVATE CUDA::cudart)
        set_target_properties(cuda_runtime_smoke PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME cuda_runtime_smoke COMMAND cuda_runtime_smoke)

        # Larger associative-memory and adaptive phase-routing experiment.
        add_executable(associative_phase_experiment
            tests/cuda/associative_phase_experiment.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/topology/topology_generator.cpp
        )
        target_include_directories(associative_phase_experiment PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(associative_phase_experiment PRIVATE CUDA::cudart)
        set_target_properties(associative_phase_experiment PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME associative_phase_experiment COMMAND associative_phase_experiment)
        set_tests_properties(associative_phase_experiment PROPERTIES TIMEOUT 180)
        add_dependencies(associative_phase_experiment check_parameter_schema)
        add_dependencies(cuda_runtime_smoke check_parameter_schema)

        add_executable(controller_smoke
            tests/cuda/controller_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/tissue/tissue_model.cpp
            src/topology/topology_generator.cpp
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

        add_test(
            NAME ipc_server_integration
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/ipc_server_test.py
                $<TARGET_FILE:spiking_model>
        )
        set_tests_properties(ipc_server_integration PROPERTIES TIMEOUT 30)
    endif()
endif()
