#include <gtest/gtest.h>

#include "oif/c_bindings.h"
#include "oif/interfaces/qeq.h"

TEST(QeqPyQeqSolverTestSuite, LinearCase)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "py_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 0.0, 2.0, -1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], 0.5);
    EXPECT_EQ(roots->data[1], 0.5);

    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqPyQeqSolverTestSuite, TwoRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "py_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1.0, 2.0, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -1.0);
    EXPECT_EQ(roots->data[1], -1.0);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqPyQeqSolverTestSuite, TwoDistinctRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "py_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1, 5, -14, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -7);
    EXPECT_EQ(roots->data[1], +2);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqCQeqSolverTestSuite, LinearCase)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "c_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 0.0, 2.0, -1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], 0.5);
    EXPECT_EQ(roots->data[1], 0.5);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqCQeqSolverTestSuite, TwoRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "c_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1.0, 2.0, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -1.0);
    EXPECT_EQ(roots->data[1], -1.0);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqCQeqSolverTestSuite, TwoDistinctRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "c_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1, 5, -14, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -7);
    EXPECT_EQ(roots->data[1], +2);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqCQeqSolverTestSuite, ExtremeRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "c_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1, -20'000, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], 19999.999949999998);
    EXPECT_EQ(roots->data[1], 5.000000012500001e-05);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqJlQeqSolverTestSuite, TwoRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "jl_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1.0, 2.0, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -1.0);
    EXPECT_EQ(roots->data[1], -1.0);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqJlQeqSolverTestSuite, TwoDistinctRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "jl_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1, 5, -14, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -7);
    EXPECT_EQ(roots->data[1], +2);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

TEST(QeqJlQeqSolverTestSuite, ExtremeRoots)
{
    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    ImplHandle implh = oif_load_impl("qeq", "jl_qeq_solver", 1, 0);

    int status = oif_solve_qeq(implh, 1, -20'000, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], 19999.999949999998);
    EXPECT_EQ(roots->data[1], 5.000000012500001e-05);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}
