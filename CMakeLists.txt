cmake_minimum_required(VERSION 3.8)

project(Equanimity CXX)
find_package(OpenCL REQUIRED)

set(SOURCE
    Source/Main.cpp
	)
source_group("Source" FILES ${SOURCE})

set(SOURCE_NEURAL
    Source/Neural/NeuralNetwork.cpp
    Source/Neural/NeuralNetwork.h
    Source/Neural/NeuralNetworkBuilder.cpp
    Source/Neural/NeuralNetworkBuilder.h
    Source/Neural/NeuralNetworkLayer.cpp
    Source/Neural/NeuralNetworkLayer.h
    Source/Neural/NeuralNetworkLayerBuilder.cpp
    Source/Neural/NeuralNetworkLayerBuilder.h
	)
source_group("Source\\Neural" FILES ${SOURCE_NEURAL})

set(SOURCE_NEURAL_LAYERS
    Source/Neural/Layers/ActivationLayer.cpp
    Source/Neural/Layers/ActivationLayer.h
    Source/Neural/Layers/FullyConnectedLayer.cpp
    Source/Neural/Layers/FullyConnectedLayer.h
    Source/Neural/Layers/InputLayer.cpp
    Source/Neural/Layers/InputLayer.h
    Source/Neural/Layers/SigmoidActivationLayer.cpp
    Source/Neural/Layers/SigmoidActivationLayer.h
	)
source_group("Source\\Neural\\Layers" FILES ${SOURCE_NEURAL_LAYERS})

set(SOURCE_FILES
	${SOURCE}
	${SOURCE_NEURAL}
	${SOURCE_NEURAL_LAYERS}
	)

include_directories(Source)

add_executable(Equanimity ${SOURCE_FILES})
target_link_libraries(Equanimity OpenCL::OpenCL)

set_target_properties(Equanimity PROPERTIES
    PROJECT_LABEL Equanimity
    LINKER_LANGUAGE CXX
    )
