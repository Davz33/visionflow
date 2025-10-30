/*
 * C Extension Source Code for Critical Performance Paths
 * Compile with: python setup.py build_ext --inplace
 */

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NumPy array object header
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Accelerate framework headers (macOS)
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Fast matrix multiplication using BLAS
static PyObject* fast_matmul(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B, *C;
    int M, N, K;
    
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B)) {
        return NULL;
    }
    
    // Get dimensions
    M = PyArray_DIM(A, 0);
    K = PyArray_DIM(A, 1);
    N = PyArray_DIM(B, 1);
    
    // Validate dimensions
    if (PyArray_DIM(B, 0) != K) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions mismatch");
        return NULL;
    }
    
    // Create output array
    npy_intp dims[2] = {M, N};
    C = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    
    // Get data pointers
    float *a_data = (float*)PyArray_DATA(A);
    float *b_data = (float*)PyArray_DATA(B);
    float *c_data = (float*)PyArray_DATA(C);
    
#ifdef __APPLE__
    // Use Accelerate framework cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, a_data, K,
                b_data, N,
                0.0f, c_data, N);
#else
    // Fallback to manual implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }
#endif
    
    return PyArray_Return(C);
}

// Fast convolution using vImage (macOS Accelerate)
static PyObject* fast_conv2d(PyObject* self, PyObject* args) {
    PyArrayObject *input, *kernel;
    int stride, padding;
    
    if (!PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &input,
                         &PyArray_Type, &kernel, &stride, &padding)) {
        return NULL;
    }
    
    // Implementation would use vImageConvolve_ARGB8888 or similar
    // For now, return input as placeholder
    (void)stride;  // Suppress unused warning
    (void)padding;  // Suppress unused warning
    Py_INCREF(input);
    return PyArray_Return(input);
}

// Fast FFT using vDSP (macOS Accelerate)
static PyObject* fast_fft(PyObject* self, PyObject* args) {
    PyArrayObject *input;
    int axis;
    
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &input, &axis)) {
        return NULL;
    }
    
    // Implementation would use vDSP_fft_zrip or similar
    // For now, return input as placeholder
    Py_INCREF(input);
    return PyArray_Return(input);
}

// Method definitions
static PyMethodDef CExtMethods[] = {
    {"fast_matmul", fast_matmul, METH_VARARGS, "Fast matrix multiplication"},
    {"fast_conv2d", fast_conv2d, METH_VARARGS, "Fast 2D convolution"},
    {"fast_fft", fast_fft, METH_VARARGS, "Fast FFT"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cextmodule = {
    PyModuleDef_HEAD_INIT,
    "visionflow_c_extensions",
    "C extensions for VisionFlow performance optimization",
    -1,
    CExtMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_visionflow_c_extensions(void) {
    import_array();
    return PyModule_Create(&cextmodule);
}
