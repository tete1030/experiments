from libcpp cimport bool
from cpython.ref cimport PyObject

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef int getCudaEnabledDeviceCount();
    cdef void setDevice(int device);
    cdef int getDevice();
    cdef void resetDevice();
