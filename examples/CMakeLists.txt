cmake_minimum_required(VERSION 3.18)

add_executable(call_qeq_from_c call_qeq_from_c.c)
target_include_directories(call_qeq_from_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(call_qeq_from_c
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/interfaces/include)
target_link_libraries(call_qeq_from_c PRIVATE oif_c)

add_executable(call_linsolve_from_c call_linsolve_from_c.c)
target_include_directories(call_linsolve_from_c
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(call_linsolve_from_c
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_link_libraries(call_linsolve_from_c PRIVATE oif_c)

add_executable(call_ivp_from_c call_ivp_from_c.c)
target_include_directories(call_ivp_from_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(call_ivp_from_c
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_link_libraries(call_ivp_from_c PRIVATE oif_c)

add_executable(call_ivp_from_c_burgers_eq call_ivp_from_c_burgers_eq.c)
target_include_directories(call_ivp_from_c_burgers_eq
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(call_ivp_from_c_burgers_eq
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_link_libraries(call_ivp_from_c_burgers_eq PRIVATE oif_c)
target_link_libraries(call_ivp_from_c_burgers_eq PRIVATE m)
