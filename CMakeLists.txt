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
endif()

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

    if(BUILD_TESTING)
        add_executable(cuda_runtime_smoke
            tests/cuda/runtime_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/runtime/simulation_controller.cpp
            src/topology/topology_generator.cpp
        )
        target_include_directories(cuda_runtime_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(cuda_runtime_smoke PRIVATE CUDA::cudart)
        set_target_properties(cuda_runtime_smoke PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
        add_test(NAME cuda_runtime_smoke COMMAND cuda_runtime_smoke)
        add_dependencies(cuda_runtime_smoke check_parameter_schema)

        add_executable(determinism_smoke
            tests/cuda/determinism_smoke.cpp
            src/cuda/neuron_izhikevich.cu
            src/runtime/cuda_simulation_runtime.cu
            src/topology/topology_generator.cpp
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
