// Dispatch library that is called from other languages, and dispatches it
// to the appropriate backend.
#include "dispatch.h"

#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <string.h>

enum {
    BACKEND_C,
    BACKEND_CXX,
    BACKEND_PYTHON,
    BACKEND_JULIA,
    BACKEND_R,
};


BackendHandle load_backend(
        const char *backend_name,
        const char *operation,
        size_t version_major,
        size_t version_minor
) {
    BackendHandle bh;
    if (strcmp(backend_name, "c") == 0) {
        bh = load_backend_c(operation, version_major, version_minor);
    } else if (strcmp(backend_name, "python") == 0) {
        bh = load_backend_python(operation, version_major, version_minor);
    } else {
        perror("Unknown backend");
        bh = 0;
    }

    return bh;
}


BackendHandle load_backend_c(
        const char *operation,
        size_t version_major,
        size_t version_minor
) {
    return BACKEND_C;
}


BackendHandle load_backend_python(
        const char *operation,
        size_t version_major,
        size_t version_minor
) {
    perror("This is not yet implemented correctly");
    exit(EXIT_FAILURE);
    return BACKEND_PYTHON;
}


int call_interface_method(
    BackendHandle bh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals
) {
    int status;
    switch (bh) {
    case BACKEND_C:
        status = run_interface_method_c(method, args, retvals);
        break;
    case BACKEND_PYTHON:
        status = run_interface_method_python(method, args, retvals);
        break;
    default:
        perror("Unknown backend");
        exit(EXIT_FAILURE);
    }
    return status;
}


int
run_interface_method_c(const char *method, OIFArgs *args, OIFArgs *out_args) {
    void *lib_handle = dlopen("./liboif_backend_c.so", RTLD_LOCAL | RTLD_LAZY); 
    void *func = dlsym(lib_handle, method);

    fprintf(stderr, "I am definitely here\n");

    int num_args = args->num_args;
    fprintf(stderr, "num_args = %d\n", num_args);
    fflush(stderr);

    ffi_cif cif;
    ffi_type **arg_types = malloc(sizeof(ffi_type) * num_args);
    fprintf(stderr, "Ready to malloc\n");
    void **arg_values = malloc(num_args * sizeof(void *));
    fprintf(stderr, "Malloced\n");

    for (size_t i = 0; i < num_args; ++i) {
        fprintf(stderr, "In a loop\n");
        if (args->arg_types[i] == OIF_FLOAT64) {
            arg_types[i] = &ffi_type_double;
        } else {
            fflush(stdout);
            fprintf(stderr, "Unknown arg type: %d", args->arg_types[i]);
            fflush(stderr);
            exit(EXIT_FAILURE);
        }
    }

    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args, &ffi_type_uint, arg_types) != FFI_OK) {
        perror("ffi_prep_cif was not OK");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < num_args; ++i) {
        arg_values[i] = args->args[i];
    }

    ffi_arg result;

    ffi_call(&cif, FFI_FN(func), &result, arg_values);

    printf("Result is %ld\n", result);

    return 0;
}

int run_interface_method_python(const char *method, OIFArgs *args, OIFArgs *retvals) {
    return 1;
}
