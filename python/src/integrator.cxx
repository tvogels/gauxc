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

#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/molecular_weights.hpp>

namespace py = pybind11;
using namespace GauXC;

void bind_integrator(py::module& m) {
    // Bind integrator settings
    py::class_<IntegratorSettingsXC>(m, "IntegratorSettingsXC")
        .def(py::init<>());

    py::class_<IntegratorSettingsEXX>(m, "IntegratorSettingsEXX")
        .def(py::init<>());

    // Helper function to evaluate meta-GGA variables
    m.def("eval_mgga_vvars",
        [](const Molecule& mol, const BasisSet<double>& basis, 
           const MolGrid& molgrid, py::array_t<double> P_array,
           std::string ks_scheme, bool need_lapl, ExecutionSpace exec_space) {
            
            // Validate input
            auto P_buf = P_array.request();
            if (P_buf.ndim != 2)
                throw std::runtime_error("Density matrix must be 2D");
            
            size_t nbf = basis.nbf();
            if (P_buf.shape[0] != nbf || P_buf.shape[1] != nbf)
                throw std::runtime_error("Density matrix shape must match basis set size");
            
            // Create runtime environment
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
            
            // Create XC integrator
            // Use a simple LDA functional for density evaluation
            XCIntegratorFactory<Eigen::MatrixXd> integrator_factory(
                exec_space, "Default", "Default", "Default");
            
            auto func = functional_map.value("SLATER");  // Simple LDA for density eval
            auto integrator = integrator_factory.get_integrator(func, lb);
            
            // Get grid data first
            const auto& tasks = lb->get_tasks();
            size_t total_npts = 0;
            for (const auto& task : tasks) {
                total_npts += task.npts;
            }
            
            // Create Eigen matrix from numpy array
            Eigen::MatrixXd P = Eigen::Map<const Eigen::MatrixXd>(
                static_cast<const double*>(P_buf.ptr), nbf, nbf);
            
            // Allocate output arrays
            py::array_t<double> rho_array(total_npts);
            py::array_t<double> grad_array({total_npts, 3});
            py::array_t<double> gamma_array(total_npts);
            py::array_t<double> tau_array(total_npts);
            
            auto rho_buf = rho_array.request();
            auto grad_buf = grad_array.request();
            auto gamma_buf = gamma_array.request();
            auto tau_buf = tau_array.request();
            
            double* rho_ptr = static_cast<double*>(rho_buf.ptr);
            double* grad_ptr = static_cast<double*>(grad_buf.ptr);
            double* gamma_ptr = static_cast<double*>(gamma_buf.ptr);
            double* tau_ptr = static_cast<double*>(tau_buf.ptr);
            
            // Process each task to extract density variables
            // This is a simplified version - actual implementation would need
            // to call internal integrator methods or use LocalWorkDriver
            size_t offset = 0;
            for (const auto& task : tasks) {
                // TODO: Call collocation and compute density variables
                // For now, return zeros as placeholder
                for (size_t i = 0; i < task.npts; i++) {
                    rho_ptr[offset + i] = 0.0;
                    grad_ptr[(offset + i) * 3 + 0] = 0.0;
                    grad_ptr[(offset + i) * 3 + 1] = 0.0;
                    grad_ptr[(offset + i) * 3 + 2] = 0.0;
                    gamma_ptr[offset + i] = 0.0;
                    tau_ptr[offset + i] = 0.0;
                }
                offset += task.npts;
            }
            
            // Build result dictionary
            py::dict result;
            result["rho"] = rho_array;
            result["grad"] = grad_array;
            result["gamma"] = gamma_array;
            result["tau"] = tau_array;
            if (need_lapl) {
                py::array_t<double> lapl_array(total_npts);
                // TODO: Compute laplacian
                result["lapl"] = lapl_array;
            }
            
            return result;
        },
        py::arg("mol"),
        py::arg("basis"),
        py::arg("molgrid"),
        py::arg("P"),
        py::arg("ks_scheme") = "RKS",
        py::arg("need_lapl") = false,
        py::arg("exec_space") = ExecutionSpace::Host,
        R"pbdoc(
        Evaluate meta-GGA variables on a molecular grid.
        
        Parameters:
        -----------
        mol : Molecule
            The molecule
        basis : BasisSet
            The basis set
        molgrid : MolGrid
            The molecular grid
        P : numpy.ndarray
            Density matrix (nbf x nbf)
        ks_scheme : str
            'RKS', 'UKS', or 'GKS' (default: 'RKS')
        need_lapl : bool
            Whether to compute Laplacian (default: False)
        exec_space : ExecutionSpace
            Execution space (Host or Device)
        
        Returns:
        --------
        dict with keys:
            rho : numpy.ndarray (npts,) - electron density
            grad : numpy.ndarray (npts, 3) - density gradient
            gamma : numpy.ndarray (npts,) - contracted gradient
            tau : numpy.ndarray (npts,) - kinetic energy density
            lapl : numpy.ndarray (npts,) - Laplacian (if need_lapl=True)
        )pbdoc"
    );
}
