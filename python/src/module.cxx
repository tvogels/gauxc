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
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for submodule binding functions
void bind_molecule(py::module& m);
void bind_basisset(py::module& m);
void bind_grid(py::module& m);
void bind_integrator(py::module& m);
void bind_torch_utils(py::module& m);

PYBIND11_MODULE(_gauxc_core, m) {
    m.doc() = "GauXC Python bindings - Core module";

    // Bind submodules
    bind_molecule(m);
    bind_basisset(m);
    bind_grid(m);
    bind_integrator(m);
    bind_torch_utils(m);
}
