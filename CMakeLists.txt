cmake_minimum_required(VERSION 3.8)

project(Equanimity CXX)
find_package(OpenCL REQUIRED)

set(SOURCE
    Source/Main.cpp
	)
source_group("Source" FILES ${SOURCE})

set(SOURCE_NEURAL
    Source/Neural/Layer.h
    Source/Neural/MultilayerPerceptron.h
	)
source_group("Source\\Neural" FILES ${SOURCE_NEURAL})

set(SOURCE_FILES
	${SOURCE}
	${SOURCE_NEURAL}
	)
	
include_directories(Source)

add_executable(Equanimity ${SOURCE_FILES})
target_link_libraries(Equanimity OpenCL::OpenCL)

set_target_properties(Equanimity PROPERTIES
    PROJECT_LABEL Equanimity
    LINKER_LANGUAGE CXX
    )
