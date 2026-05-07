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

#include <gauxc/basisset.hpp>
#include <gauxc/shell.hpp>

namespace py = pybind11;
using namespace GauXC;

void bind_basisset(py::module& m) {
    // Bind PrimSize
    py::class_<PrimSize>(m, "PrimSize")
        .def(py::init<int32_t>())
        .def("get", &PrimSize::get);

    // Bind AngularMomentum
    py::class_<AngularMomentum>(m, "AngularMomentum")
        .def(py::init<int32_t>())
        .def("get", &AngularMomentum::get);

    // Bind SphericalType enum
    py::enum_<SphericalType>(m, "SphericalType")
        .value("Cartesian", SphericalType::Cartesian)
        .value("Spherical", SphericalType::Spherical)
        .export_values();

    // Bind Shell<double>
    py::class_<Shell<double>>(m, "Shell")
        .def(py::init<>())
        .def(py::init<SphericalType, std::array<double,3>, AngularMomentum, 
                      std::vector<double>, std::vector<double>, bool>(),
             py::arg("pure"), py::arg("O"), py::arg("l"),
             py::arg("alpha"), py::arg("coeff"), py::arg("normalized") = true)
        .def("size", &Shell<double>::size)
        .def("nprim", &Shell<double>::nprim)
        .def("l", &Shell<double>::l)
        .def("pure", &Shell<double>::pure)
        .def("cart_size", &Shell<double>::cart_size)
        .def("O", &Shell<double>::O)
        .def("alpha", [](const Shell<double>& sh) { 
            return std::vector<double>(sh.alpha(), sh.alpha() + sh.nprim());
        })
        .def("coeff", [](const Shell<double>& sh) { 
            return std::vector<double>(sh.coeff(), sh.coeff() + sh.nprim());
        })
        .def("set_shell_tolerance", &Shell<double>::set_shell_tolerance)
        .def("__repr__", [](const Shell<double>& sh) {
            return "<Shell l=" + std::to_string(sh.l()) + 
                   " nprim=" + std::to_string(sh.nprim()) + 
                   " size=" + std::to_string(sh.size()) + ">";
        });

    // Bind BasisSet<double>
    py::class_<BasisSet<double>>(m, "BasisSet")
        .def(py::init<>())
        .def(py::init<std::vector<Shell<double>>>())
        .def("nshells", &BasisSet<double>::nshells)
        .def("nbf", &BasisSet<double>::nbf)
        .def("nbf_cart", &BasisSet<double>::nbf_cart)
        .def("max_l", &BasisSet<double>::max_l)
        .def("__len__", &BasisSet<double>::nshells)
        .def("__getitem__", [](const BasisSet<double>& basis, size_t i) {
            if (i >= basis.size()) throw py::index_error();
            return basis[i];
        })
        .def("append", [](BasisSet<double>& basis, const Shell<double>& shell) {
            basis.push_back(shell);
        })
        .def("set_basis_tolerance", [](BasisSet<double>& basis, double tol) {
            for (auto& sh : basis) {
                sh.set_shell_tolerance(tol);
            }
        })
        .def("__repr__", [](const BasisSet<double>& basis) {
            return "<BasisSet nshells=" + std::to_string(basis.nshells()) + 
                   " nbf=" + std::to_string(basis.nbf()) + ">";
        });

    // Helper to create a simple basis set
    m.def("create_simple_shell",
        [](int Z, int l, py::array_t<double> alphas, py::array_t<double> coeffs,
           double x, double y, double z, bool spherical) {
            auto a_buf = alphas.request();
            auto c_buf = coeffs.request();
            
            if (a_buf.ndim != 1 || c_buf.ndim != 1)
                throw std::runtime_error("alphas and coeffs must be 1D arrays");
            if (a_buf.shape[0] != c_buf.shape[0])
                throw std::runtime_error("alphas and coeffs must have same length");
            
            std::vector<double> alpha_vec(a_buf.shape[0]);
            std::vector<double> coeff_vec(c_buf.shape[0]);
            
            double* a_ptr = static_cast<double*>(a_buf.ptr);
            double* c_ptr = static_cast<double*>(c_buf.ptr);
            
            for (size_t i = 0; i < a_buf.shape[0]; i++) {
                alpha_vec[i] = a_ptr[i];
                coeff_vec[i] = c_ptr[i];
            }
            
            SphericalType sph = spherical ? SphericalType::Spherical : SphericalType::Cartesian;
            std::array<double, 3> origin = {x, y, z};
            
            return Shell<double>(sph, origin, AngularMomentum(l), 
                                 alpha_vec, coeff_vec, true);
        },
        py::arg("Z"),
        py::arg("l"),
        py::arg("alphas"),
        py::arg("coeffs"),
        py::arg("x") = 0.0,
        py::arg("y") = 0.0,
        py::arg("z") = 0.0,
        py::arg("spherical") = true,
        "Create a simple GTO shell"
    );
}
