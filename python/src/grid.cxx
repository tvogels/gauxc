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

#include <gauxc/molgrid.hpp>
#include <gauxc/grid_factory.hpp>
#include <gauxc/grid.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/runtime_environment.hpp>

namespace py = pybind11;
using namespace GauXC;

void bind_grid(py::module& m) {
    // Bind BatchSize
    py::class_<BatchSize>(m, "BatchSize")
        .def(py::init<int64_t>())
        .def("get", &BatchSize::get);

    // Bind RadialSize
    py::class_<RadialSize>(m, "RadialSize")
        .def(py::init<int64_t>())
        .def("get", &RadialSize::get);

    // Bind AngularSize
    py::class_<AngularSize>(m, "AngularSize")
        .def(py::init<int64_t>())
        .def("get", &AngularSize::get);

    // Bind RadialScale
    py::class_<RadialScale>(m, "RadialScale")
        .def(py::init<double>())
        .def("get", &RadialScale::get);

    // Bind enums
    py::enum_<RadialQuad>(m, "RadialQuad")
        .value("Becke", RadialQuad::Becke)
        .value("MuraKnowles", RadialQuad::MuraKnowles)
        .value("TreutlerAhlrichs", RadialQuad::TreutlerAhlrichs)
        .value("MurrayHandyLaming", RadialQuad::MurrayHandyLaming)
        .export_values();

    py::enum_<PruningScheme>(m, "PruningScheme")
        .value("Unpruned", PruningScheme::Unpruned)
        .value("Robust", PruningScheme::Robust)
        .value("Treutler", PruningScheme::Treutler)
        .export_values();

    py::enum_<AtomicGridSizeDefault>(m, "AtomicGridSizeDefault")
        .value("FineGrid", AtomicGridSizeDefault::FineGrid)
        .value("UltraFineGrid", AtomicGridSizeDefault::UltraFineGrid)
        .value("SuperFineGrid", AtomicGridSizeDefault::SuperFineGrid)
        .value("GM3", AtomicGridSizeDefault::GM3)
        .value("GM5", AtomicGridSizeDefault::GM5)
        .export_values();

    py::enum_<ExecutionSpace>(m, "ExecutionSpace")
        .value("Host", ExecutionSpace::Host)
#ifdef GAUXC_HAS_DEVICE
        .value("Device", ExecutionSpace::Device)
#endif
        .export_values();

    // Bind Grid class
    py::class_<Grid>(m, "Grid")
        .def("__repr__", [](const Grid&) {
            return "<Grid>";
        });

    // Bind MolGrid class
    py::class_<MolGrid>(m, "MolGrid")
        .def("natoms_uniq", &MolGrid::natoms_uniq)
        .def("max_nbatches", &MolGrid::max_nbatches)
        .def("__repr__", [](const MolGrid& mg) {
            return "<MolGrid natoms=" + std::to_string(mg.natoms_uniq()) + ">";
        });

    // Helper function to create molecular grid
    m.def("create_molgrid",
        [](const Molecule& mol, PruningScheme prune, BatchSize batch_size,
           RadialQuad rad_quad, AtomicGridSizeDefault grid_size) {
            return MolGridFactory::create_default_molgrid(
                mol, prune, batch_size, rad_quad, grid_size
            );
        },
        py::arg("mol"),
        py::arg("pruning_scheme") = PruningScheme::Robust,
        py::arg("batch_size") = BatchSize(512),
        py::arg("radial_quad") = RadialQuad::MuraKnowles,
        py::arg("grid_size") = AtomicGridSizeDefault::UltraFineGrid,
        "Create a molecular grid from a molecule"
    );

    // Bind LoadBalancer (minimal for grid extraction)
    py::class_<LoadBalancer>(m, "LoadBalancer")
        .def("__repr__", [](const LoadBalancer&) {
            return "<LoadBalancer>";
        });

    // Create load balancer and extract grid data
    m.def("extract_grid_data",
        [](const Molecule& mol, const MolGrid& molgrid, 
           const BasisSet<double>& basis, ExecutionSpace exec_space) {
            
            // Create runtime environment (no MPI for simplicity)
            #ifdef GAUXC_HAS_DEVICE
            RuntimeEnvironment rt = DeviceRuntimeEnvironment(0.9);
            #else
            RuntimeEnvironment rt = RuntimeEnvironment();
            #endif
            
            // Create load balancer
            LoadBalancerFactory lb_factory(exec_space, "Replicated");
            auto lb = lb_factory.get_shared_instance(rt, mol, molgrid, basis);
            
            // Apply molecular weights
            MolecularWeightsFactory mw_factory(exec_space, "Default", 
                MolecularWeightsSettings{});
            auto mw = mw_factory.get_instance();
            mw.modify_weights(*lb);
            
            // Extract points and weights from tasks
            const auto& tasks = lb->get_tasks();
            
            // Count total points
            size_t total_npts = 0;
            for (const auto& task : tasks) {
                total_npts += task.npts;
            }
            
            // Allocate arrays
            py::array_t<double> points({total_npts, 3});
            py::array_t<double> weights(total_npts);
            
            auto p_buf = points.request();
            auto w_buf = weights.request();
            double* p_ptr = static_cast<double*>(p_buf.ptr);
            double* w_ptr = static_cast<double*>(w_buf.ptr);
            
            // Copy data
            size_t offset = 0;
            for (const auto& task : tasks) {
                for (size_t i = 0; i < task.npts; i++) {
                    p_ptr[(offset + i) * 3 + 0] = task.points[i][0];
                    p_ptr[(offset + i) * 3 + 1] = task.points[i][1];
                    p_ptr[(offset + i) * 3 + 2] = task.points[i][2];
                    w_ptr[offset + i] = task.weights[i];
                }
                offset += task.npts;
            }
            
            // Return as dict
            py::dict result;
            result["points"] = points;
            result["weights"] = weights;
            result["npts"] = total_npts;
            
            return result;
        },
        py::arg("mol"),
        py::arg("molgrid"),
        py::arg("basis"),
        py::arg("exec_space") = ExecutionSpace::Host,
        "Extract grid points and weights from molecular grid"
    );
}
