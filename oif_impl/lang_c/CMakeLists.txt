cmake_minimum_required(VERSION 3.18)

project(oif_dispatch_c LANGUAGES C)

add_library(oif_dispatch_c SHARED dispatch_c.c)
target_include_directories(oif_dispatch_c
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)

target_link_libraries(oif_dispatch_c PRIVATE oif_c_data_structures)
target_link_libraries(oif_dispatch_c PRIVATE oif_c_util)
target_link_libraries(oif_dispatch_c PRIVATE ffi)
