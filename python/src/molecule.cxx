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

#include <gauxc/molecule.hpp>
#include <gauxc/atom.hpp>

namespace py = pybind11;
using namespace GauXC;

void bind_molecule(py::module& m) {
    // Bind AtomicNumber
    py::class_<AtomicNumber>(m, "AtomicNumber")
        .def(py::init<int>())
        .def("get", &AtomicNumber::get)
        .def("__int__", &AtomicNumber::get)
        .def("__repr__", [](const AtomicNumber& z) {
            return "<AtomicNumber " + std::to_string(z.get()) + ">";
        });

    // Bind Atom
    py::class_<Atom>(m, "Atom")
        .def(py::init<>())
        .def(py::init<AtomicNumber, double, double, double>(),
             py::arg("Z"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("Z", &Atom::Z)
        .def_readwrite("x", &Atom::x)
        .def_readwrite("y", &Atom::y)
        .def_readwrite("z", &Atom::z)
        .def("__repr__", [](const Atom& a) {
            return "<Atom Z=" + std::to_string(a.Z.get()) + 
                   " pos=(" + std::to_string(a.x) + ", " + 
                   std::to_string(a.y) + ", " + 
                   std::to_string(a.z) + ")>";
        });

    // Bind Molecule
    py::class_<Molecule>(m, "Molecule")
        .def(py::init<>())
        .def(py::init<std::vector<Atom>>())
        .def("natoms", &Molecule::natoms)
        .def("maxZ", &Molecule::maxZ)
        .def("__len__", &Molecule::natoms)
        .def("__getitem__", [](const Molecule& mol, size_t i) {
            if (i >= mol.size()) throw py::index_error();
            return mol[i];
        })
        .def("__setitem__", [](Molecule& mol, size_t i, const Atom& atom) {
            if (i >= mol.size()) throw py::index_error();
            mol[i] = atom;
        })
        .def("append", [](Molecule& mol, const Atom& atom) {
            mol.push_back(atom);
        })
        .def("__repr__", [](const Molecule& mol) {
            return "<Molecule with " + std::to_string(mol.natoms()) + " atoms>";
        });

    // Helper function to create molecule from arrays
    m.def("create_molecule", 
        [](py::array_t<int> atomic_numbers, py::array_t<double> coords) {
            auto z_buf = atomic_numbers.request();
            auto c_buf = coords.request();
            
            if (z_buf.ndim != 1 || c_buf.ndim != 2)
                throw std::runtime_error("atomic_numbers must be 1D, coords must be 2D");
            if (c_buf.shape[1] != 3)
                throw std::runtime_error("coords must have shape (natoms, 3)");
            if (z_buf.shape[0] != c_buf.shape[0])
                throw std::runtime_error("atomic_numbers and coords must have same length");
            
            Molecule mol;
            int* z_ptr = static_cast<int*>(z_buf.ptr);
            double* c_ptr = static_cast<double*>(c_buf.ptr);
            
            for (size_t i = 0; i < z_buf.shape[0]; i++) {
                mol.push_back(Atom(
                    AtomicNumber(z_ptr[i]),
                    c_ptr[i*3 + 0],
                    c_ptr[i*3 + 1],
                    c_ptr[i*3 + 2]
                ));
            }
            
            return mol;
        },
        py::arg("atomic_numbers"),
        py::arg("coords"),
        "Create a Molecule from atomic numbers (N,) and coordinates (N, 3) arrays"
    );

    // Helper function to extract coordinates
    m.def("get_molecule_coords",
        [](const Molecule& mol) {
            size_t natoms = mol.size();
            py::array_t<double> coords({natoms, 3});
            auto buf = coords.request();
            double* ptr = static_cast<double*>(buf.ptr);
            
            for (size_t i = 0; i < natoms; i++) {
                ptr[i*3 + 0] = mol[i].x;
                ptr[i*3 + 1] = mol[i].y;
                ptr[i*3 + 2] = mol[i].z;
            }
            
            return coords;
        },
        py::arg("mol"),
        "Extract coordinates as (N, 3) array"
    );
}
