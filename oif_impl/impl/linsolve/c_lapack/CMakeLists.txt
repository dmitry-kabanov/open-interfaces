add_library(oif_linsolve_c_lapack SHARED linsolve_impl.c)
set(BLA_VENDOR Intel10_64_dyn)
find_package(LAPACK)
target_link_libraries(oif_linsolve_c_lapack PRIVATE LAPACK::LAPACK)
target_include_directories(oif_linsolve_c_lapack
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_linsolve_c_lapack
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif_impl/lang_c/include)
