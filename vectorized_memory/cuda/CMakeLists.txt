set(mylib "gpu") # set in this scope
set(mylib ${mylib} PARENT_SCOPE) # set in the parent scope too

# packages
find_package(CUDA)

# nvcc flags
# set(CUDA_NVCC_FLAGS -O3;-G;-g)
# set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
# set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)
set(CUDA_NVCC_FLAGS -std=c++14 -O2 -arch=sm_35 -G -g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp)
file(GLOB CURRENT_SOURCES  *.cpp *.cc *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(${mylib} SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
# cuda_add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

# target_include_directories(${lib_name} PUBLIC ${PROJECT_SOURCE_DIR})
