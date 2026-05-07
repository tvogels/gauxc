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
#include <gauxc/basisset.hpp>
#include <gauxc/molgrid.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <gauxc/basisset_map.hpp>

#include <vector>
#include <memory>
#include <cstring>

namespace py = pybind11;
using namespace GauXC;

void bind_integrator(py::module& m) {
    // Implementation of meta-GGA variable evaluation
    m.def("eval_mgga_vvars_impl",
        [](const Molecule& mol, const BasisSet<double>& basis,
           const MolGrid& molgrid, py::array_t<double> P_array,
           bool need_lapl, ExecutionSpace exec_space) {
            
            // Validate density matrix
            auto P_buf = P_array.request();
            if (P_buf.ndim != 2)
                throw std::runtime_error("Density matrix must be 2D");
            
            size_t nbf = basis.nbf();
            if (P_buf.shape[0] != nbf || P_buf.shape[1] != nbf)
                throw std::runtime_error("Density matrix size must match basis set");
            
            const double* P = static_cast<const double*>(P_buf.ptr);
            
            // Create runtime environment
            #ifdef GAUXC_HAS_MPI
            #ifdef GAUXC_HAS_DEVICE
            RuntimeEnvironment rt = DeviceRuntimeEnvironment(MPI_COMM_SELF, 0.9);
            #else
            RuntimeEnvironment rt = RuntimeEnvironment(MPI_COMM_SELF);
            #endif
            #else
            #ifdef GAUXC_HAS_DEVICE
            RuntimeEnvironment rt = DeviceRuntimeEnvironment(0.9);
            #else
            RuntimeEnvironment rt = RuntimeEnvironment();
            #endif
            #endif
            
            // Create load balancer
            LoadBalancerFactory lb_factory(exec_space, "Replicated");
            auto lb = lb_factory.get_shared_instance(rt, mol, molgrid, basis);
            
            // Apply molecular weights
            MolecularWeightsFactory mw_factory(exec_space, "Default",
                MolecularWeightsSettings{});
            auto mw = mw_factory.get_instance();
            mw.modify_weights(*lb);
            
            // Create local work driver
            auto lwd = LocalWorkDriverFactory::make_local_work_driver(
                exec_space, "Default", LocalWorkSettings());
            
            // Get tasks
            const auto& tasks = lb->get_tasks();
            
            // Count total points
            size_t total_npts = 0;
            for (const auto& task : tasks) {
                total_npts += task.npts;
            }
            
            // Allocate output arrays
            py::array_t<double> rho_array(total_npts);
            py::array_t<double> grad_array({total_npts, 3});
            py::array_t<double> gamma_array(total_npts);
            py::array_t<double> tau_array(total_npts);
            py::array_t<double> lapl_array;
            
            if (need_lapl) {
                lapl_array = py::array_t<double>(total_npts);
            }
            
            auto rho_buf = rho_array.request();
            auto grad_buf = grad_array.request();
            auto gamma_buf = gamma_array.request();
            auto tau_buf = tau_array.request();
            
            double* rho_ptr = static_cast<double*>(rho_buf.ptr);
            double* grad_ptr = static_cast<double*>(grad_buf.ptr);
            double* gamma_ptr = static_cast<double*>(gamma_buf.ptr);
            double* tau_ptr = static_cast<double*>(tau_buf.ptr);
            double* lapl_ptr = need_lapl ? static_cast<double*>(lapl_array.request().ptr) : nullptr;
            
            // Create basis set map for submatrix extraction
            BasisSetMap basis_map(basis, mol);
            
            // Process each task
            size_t global_offset = 0;
            for (const auto& task : tasks) {
                const size_t npts = task.npts;
                const size_t nshells = task.bfn_screening.shell_list.size();
                const size_t nbe = task.bfn_screening.nbe;
                
                if (npts == 0 || nbe == 0) continue;
                
                const double* pts = task.points.data()->data();
                const int32_t* shell_list = task.bfn_screening.shell_list.data();
                
                // Allocate scratch space for collocation
                std::vector<double> basis_eval(nbe * npts);
                std::vector<double> dbasis_x(nbe * npts);
                std::vector<double> dbasis_y(nbe * npts);
                std::vector<double> dbasis_z(nbe * npts);
                std::vector<double> lbasis(nbe * npts, 0.0);
                
                // Evaluate collocation with gradients
                if (need_lapl) {
                    // Need hessian for laplacian (tr(Hessian))
                    std::vector<double> d2basis_xx(nbe * npts);
                    std::vector<double> d2basis_yy(nbe * npts);
                    std::vector<double> d2basis_zz(nbe * npts);
                    std::vector<double> d2basis_xy(nbe * npts);
                    std::vector<double> d2basis_xz(nbe * npts);
                    std::vector<double> d2basis_yz(nbe * npts);
                    
                    lwd->eval_collocation_hessian(npts, nshells, nbe, pts, basis,
                        shell_list, basis_eval.data(), dbasis_x.data(),
                        dbasis_y.data(), dbasis_z.data(), d2basis_xx.data(),
                        d2basis_xy.data(), d2basis_xz.data(), d2basis_yy.data(),
                        d2basis_yz.data(), d2basis_zz.data());
                    
                    // Compute laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
                    for (size_t i = 0; i < nbe * npts; ++i) {
                        lbasis[i] = d2basis_xx[i] + d2basis_yy[i] + d2basis_zz[i];
                    }
                } else {
                    lwd->eval_collocation_gradient(npts, nshells, nbe, pts, basis,
                        shell_list, basis_eval.data(), dbasis_x.data(),
                        dbasis_y.data(), dbasis_z.data());
                }
                
                // Compute X matrix: X = P * basis_eval (submatrix)
                std::vector<double> X(nbe * npts);
                std::vector<double> scr(nbf * nbe);
                
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, P, nbf, basis_eval.data(), nbe, X.data(), nbe, scr.data());
                
                // Compute M matrices for tau (kinetic energy density)
                // M_x = ∇_x P * ∇_x basis, etc.
                std::vector<double> mmat_x(nbe * npts);
                std::vector<double> mmat_y(nbe * npts);
                std::vector<double> mmat_z(nbe * npts);
                
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, P, nbf, dbasis_x.data(), nbe, mmat_x.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, P, nbf, dbasis_y.data(), nbe, mmat_y.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, P, nbf, dbasis_z.data(), nbe, mmat_z.data(), nbe, scr.data());
                
                // Evaluate density variables
                double* task_rho = rho_ptr + global_offset;
                double* task_grad_x = grad_ptr + global_offset * 3;
                double* task_grad_y = task_grad_x + 1;
                double* task_grad_z = task_grad_x + 2;
                double* task_gamma = gamma_ptr + global_offset;
                double* task_tau = tau_ptr + global_offset;
                double* task_lapl = need_lapl ? (lapl_ptr + global_offset) : nullptr;
                
                // Temporary arrays for gradient components
                std::vector<double> dden_x(npts);
                std::vector<double> dden_y(npts);
                std::vector<double> dden_z(npts);
                
                lwd->eval_uvvar_mgga_rks(npts, nbe, basis_eval.data(),
                    dbasis_x.data(), dbasis_y.data(), dbasis_z.data(),
                    lbasis.data(), X.data(), nbe, mmat_x.data(),
                    mmat_y.data(), mmat_z.data(), nbe, task_rho,
                    dden_x.data(), dden_y.data(), dden_z.data(),
                    task_gamma, task_tau, task_lapl);
                
                // Copy gradient components to output (interleaved format)
                for (size_t i = 0; i < npts; ++i) {
                    task_grad_x[i * 3] = dden_x[i];
                    task_grad_y[i * 3] = dden_y[i];
                    task_grad_z[i * 3] = dden_z[i];
                }
                
                global_offset += npts;
            }
            
            // Build result dictionary
            py::dict result;
            result["rho"] = rho_array;
            result["grad"] = grad_array;
            result["gamma"] = gamma_array;
            result["tau"] = tau_array;
            if (need_lapl) {
                result["lapl"] = lapl_array;
            }
            
            return result;
        },
        py::arg("mol"),
        py::arg("basis"),
        py::arg("molgrid"),
        py::arg("P"),
        py::arg("need_lapl") = false,
        py::arg("exec_space") = ExecutionSpace::Host,
        R"pbdoc(
        Evaluate meta-GGA variables on a molecular grid.
        
        This function computes density and related quantities needed for
        meta-GGA functionals using GauXC's collocation and density evaluation
        kernels.
        
        Parameters:
        -----------
        mol : Molecule
            The molecule
        basis : BasisSet
            The basis set
        molgrid : MolGrid
            The molecular grid
        P : numpy.ndarray (nbf, nbf)
            Density matrix (symmetric)
        need_lapl : bool
            Whether to compute Laplacian (default: False)
        exec_space : ExecutionSpace
            Execution space (Host or Device)
        
        Returns:
        --------
        dict with keys:
            rho : numpy.ndarray (npts,) - electron density ρ(r)
            grad : numpy.ndarray (npts, 3) - density gradient ∇ρ(r)
            gamma : numpy.ndarray (npts,) - contracted gradient γ(r) = |∇ρ(r)|²
            tau : numpy.ndarray (npts,) - kinetic energy density τ(r)
            lapl : numpy.ndarray (npts,) - Laplacian ∇²ρ(r) (if need_lapl=True)
        )pbdoc"
    );
    
    // UKS implementation
    m.def("eval_mgga_vvars_impl_uks",
        [](const Molecule& mol, const BasisSet<double>& basis,
           const MolGrid& molgrid, py::array_t<double> Ps_array,
           py::array_t<double> Pz_array, bool need_lapl, ExecutionSpace exec_space) {
            
            // Validate density matrices
            auto Ps_buf = Ps_array.request();
            auto Pz_buf = Pz_array.request();
            
            if (Ps_buf.ndim != 2 || Pz_buf.ndim != 2)
                throw std::runtime_error("Density matrices must be 2D");
            
            size_t nbf = basis.nbf();
            if (Ps_buf.shape[0] != nbf || Ps_buf.shape[1] != nbf)
                throw std::runtime_error("Ps size must match basis set");
            if (Pz_buf.shape[0] != nbf || Pz_buf.shape[1] != nbf)
                throw std::runtime_error("Pz size must match basis set");
            
            const double* Ps = static_cast<const double*>(Ps_buf.ptr);
            const double* Pz = static_cast<const double*>(Pz_buf.ptr);
            
            // Create runtime environment
            #ifdef GAUXC_HAS_MPI
            #ifdef GAUXC_HAS_DEVICE
            RuntimeEnvironment rt = DeviceRuntimeEnvironment(MPI_COMM_SELF, 0.9);
            #else
            RuntimeEnvironment rt = RuntimeEnvironment(MPI_COMM_SELF);
            #endif
            #else
            #ifdef GAUXC_HAS_DEVICE
            RuntimeEnvironment rt = DeviceRuntimeEnvironment(0.9);
            #else
            RuntimeEnvironment rt = RuntimeEnvironment();
            #endif
            #endif
            
            // Create load balancer
            LoadBalancerFactory lb_factory(exec_space, "Replicated");
            auto lb = lb_factory.get_shared_instance(rt, mol, molgrid, basis);
            
            // Apply molecular weights
            MolecularWeightsFactory mw_factory(exec_space, "Default",
                MolecularWeightsSettings{});
            auto mw = mw_factory.get_instance();
            mw.modify_weights(*lb);
            
            // Create local work driver
            auto lwd = LocalWorkDriverFactory::make_local_work_driver(
                exec_space, "Default", LocalWorkSettings());
            
            // Get tasks
            const auto& tasks = lb->get_tasks();
            
            // Count total points
            size_t total_npts = 0;
            for (const auto& task : tasks) {
                total_npts += task.npts;
            }
            
            // Allocate output arrays
            py::array_t<double> rho_array(total_npts);
            py::array_t<double> grad_array({total_npts, 3});
            py::array_t<double> gamma_array(total_npts);
            py::array_t<double> tau_array(total_npts);
            py::array_t<double> lapl_array;
            
            if (need_lapl) {
                lapl_array = py::array_t<double>(total_npts);
            }
            
            auto rho_buf = rho_array.request();
            auto grad_buf = grad_array.request();
            auto gamma_buf = gamma_array.request();
            auto tau_buf = tau_array.request();
            
            double* rho_ptr = static_cast<double*>(rho_buf.ptr);
            double* grad_ptr = static_cast<double*>(grad_buf.ptr);
            double* gamma_ptr = static_cast<double*>(gamma_buf.ptr);
            double* tau_ptr = static_cast<double*>(tau_buf.ptr);
            double* lapl_ptr = need_lapl ? static_cast<double*>(lapl_array.request().ptr) : nullptr;
            
            // Create basis set map for submatrix extraction
            BasisSetMap basis_map(basis, mol);
            
            // Process each task
            size_t global_offset = 0;
            for (const auto& task : tasks) {
                const size_t npts = task.npts;
                const size_t nshells = task.bfn_screening.shell_list.size();
                const size_t nbe = task.bfn_screening.nbe;
                
                if (npts == 0 || nbe == 0) continue;
                
                const double* pts = task.points.data()->data();
                const int32_t* shell_list = task.bfn_screening.shell_list.data();
                
                // Allocate scratch space for collocation
                std::vector<double> basis_eval(nbe * npts);
                std::vector<double> dbasis_x(nbe * npts);
                std::vector<double> dbasis_y(nbe * npts);
                std::vector<double> dbasis_z(nbe * npts);
                std::vector<double> lbasis(nbe * npts, 0.0);
                
                // Evaluate collocation with gradients
                if (need_lapl) {
                    // Need hessian for laplacian (tr(Hessian))
                    std::vector<double> d2basis_xx(nbe * npts);
                    std::vector<double> d2basis_yy(nbe * npts);
                    std::vector<double> d2basis_zz(nbe * npts);
                    std::vector<double> d2basis_xy(nbe * npts);
                    std::vector<double> d2basis_xz(nbe * npts);
                    std::vector<double> d2basis_yz(nbe * npts);
                    
                    lwd->eval_collocation_hessian(npts, nshells, nbe, pts, basis,
                        shell_list, basis_eval.data(), dbasis_x.data(),
                        dbasis_y.data(), dbasis_z.data(), d2basis_xx.data(),
                        d2basis_xy.data(), d2basis_xz.data(), d2basis_yy.data(),
                        d2basis_yz.data(), d2basis_zz.data());
                    
                    // Compute laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
                    for (size_t i = 0; i < nbe * npts; ++i) {
                        lbasis[i] = d2basis_xx[i] + d2basis_yy[i] + d2basis_zz[i];
                    }
                } else {
                    lwd->eval_collocation_gradient(npts, nshells, nbe, pts, basis,
                        shell_list, basis_eval.data(), dbasis_x.data(),
                        dbasis_y.data(), dbasis_z.data());
                }
                
                // Compute X matrices for spin-up and spin-down
                std::vector<double> Xs(nbe * npts);
                std::vector<double> Xz(nbe * npts);
                std::vector<double> scr(nbf * nbe);
                
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Ps, nbf, basis_eval.data(), nbe, Xs.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Pz, nbf, basis_eval.data(), nbe, Xz.data(), nbe, scr.data());
                
                // Compute M matrices for tau (kinetic energy density)
                std::vector<double> mmat_xs(nbe * npts);
                std::vector<double> mmat_ys(nbe * npts);
                std::vector<double> mmat_zs(nbe * npts);
                std::vector<double> mmat_xz(nbe * npts);
                std::vector<double> mmat_yz(nbe * npts);
                std::vector<double> mmat_zz(nbe * npts);
                
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Ps, nbf, dbasis_x.data(), nbe, mmat_xs.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Ps, nbf, dbasis_y.data(), nbe, mmat_ys.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Ps, nbf, dbasis_z.data(), nbe, mmat_zs.data(), nbe, scr.data());
                
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Pz, nbf, dbasis_x.data(), nbe, mmat_xz.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Pz, nbf, dbasis_y.data(), nbe, mmat_yz.data(), nbe, scr.data());
                lwd->eval_xmat(npts, nbf, nbe, task.bfn_screening.submat_map,
                    1.0, Pz, nbf, dbasis_z.data(), nbe, mmat_zz.data(), nbe, scr.data());
                
                // Evaluate density variables
                double* task_rho = rho_ptr + global_offset;
                double* task_grad_x = grad_ptr + global_offset * 3;
                double* task_grad_y = task_grad_x + 1;
                double* task_grad_z = task_grad_x + 2;
                double* task_gamma = gamma_ptr + global_offset;
                double* task_tau = tau_ptr + global_offset;
                double* task_lapl = need_lapl ? (lapl_ptr + global_offset) : nullptr;
                
                // Temporary arrays for gradient components
                std::vector<double> dden_x(npts);
                std::vector<double> dden_y(npts);
                std::vector<double> dden_z(npts);
                
                lwd->eval_uvvar_mgga_uks(npts, nbe, basis_eval.data(),
                    dbasis_x.data(), dbasis_y.data(), dbasis_z.data(),
                    lbasis.data(), Xs.data(), nbe, Xz.data(), nbe,
                    mmat_xs.data(), mmat_ys.data(), mmat_zs.data(), nbe,
                    mmat_xz.data(), mmat_yz.data(), mmat_zz.data(), nbe,
                    task_rho, dden_x.data(), dden_y.data(), dden_z.data(),
                    task_gamma, task_tau, task_lapl);
                
                // Copy gradient components to output (interleaved format)
                for (size_t i = 0; i < npts; ++i) {
                    task_grad_x[i * 3] = dden_x[i];
                    task_grad_y[i * 3] = dden_y[i];
                    task_grad_z[i * 3] = dden_z[i];
                }
                
                global_offset += npts;
            }
            
            // Build result dictionary
            py::dict result;
            result["rho"] = rho_array;
            result["grad"] = grad_array;
            result["gamma"] = gamma_array;
            result["tau"] = tau_array;
            if (need_lapl) {
                result["lapl"] = lapl_array;
            }
            
            return result;
        },
        py::arg("mol"),
        py::arg("basis"),
        py::arg("molgrid"),
        py::arg("Ps"),
        py::arg("Pz"),
        py::arg("need_lapl") = false,
        py::arg("exec_space") = ExecutionSpace::Host,
        R"pbdoc(
        Evaluate meta-GGA variables on a molecular grid (UKS).
        
        This function computes density and related quantities for unrestricted
        Kohn-Sham calculations with separate spin-up and spin-down density matrices.
        
        Parameters:
        -----------
        mol : Molecule
            The molecule
        basis : BasisSet
            The basis set
        molgrid : MolGrid
            The molecular grid
        Ps : numpy.ndarray (nbf, nbf)
            Spin-up density matrix
        Pz : numpy.ndarray (nbf, nbf)
            Spin-down density matrix
        need_lapl : bool
            Whether to compute Laplacian (default: False)
        exec_space : ExecutionSpace
            Execution space (Host or Device)
        
        Returns:
        --------
        dict with keys:
            rho : numpy.ndarray (npts,) - total electron density (alpha + beta)
            grad : numpy.ndarray (npts, 3) - total density gradient
            gamma : numpy.ndarray (npts,) - contracted gradient
            tau : numpy.ndarray (npts,) - total kinetic energy density
            lapl : numpy.ndarray (npts,) - Laplacian (if need_lapl=True)
        )pbdoc"
    );
}
