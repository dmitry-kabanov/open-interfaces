find_package(SUNDIALS REQUIRED)

add_library(oif_ivp_sundials_cvode SHARED sundials_cvode.c)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE m)
target_include_directories(oif_ivp_sundials_cvode
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_ivp_sundials_cvode
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif_impl/lang_c/include)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE oif_common_data_structures)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE oif_common_util)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE SUNDIALS::cvode)
