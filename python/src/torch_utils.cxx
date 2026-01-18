/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void bind_torch_utils(py::module& m) {
    // Helper functions for PyTorch tensor conversion
    // These will be used from Python side with dlpack
    
    m.def("supports_torch", []() {
        // Check if torch is available at runtime
        return py::module_::import("importlib.util").attr("find_spec")("torch").is_none() == false;
    }, "Check if PyTorch is available");
    
    // Note: Actual DLPack tensor conversion will be handled in Python layer
    // using __dlpack__ protocol which is supported by both numpy and torch
}
