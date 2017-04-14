cmake_minimum_required(VERSION 3.8)

project(Equanimity)

set(SOURCE
    Source/ComputationalGraph.h
    Source/MultilayerPerceptron.h
    Source/Main.cpp
	)

source_group("Source" FILES ${SOURCE})

set(SOURCE_FILES
	${SOURCE}
	)
	
include_directories(Source)

add_executable(Equanimity ${SOURCE_FILES})

set_target_properties(Equanimity PROPERTIES
    PROJECT_LABEL Equanimity
    LINKER_LANGUAGE CXX
    )
