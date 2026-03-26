/**
 * Minimal OneDFT driver for testing eval_exc_vxc_onedft with arbitrary models.
 *
 * Usage:
 *   onedft_driver <input.hdf5> <model_path> [--device]
 *   mpirun -np 2 onedft_driver <input.hdf5> <model_path> [--device]
 *
 * The HDF5 file must contain: MOLECULE, BASIS, DENSITY_SCALAR, DENSITY_Z
 */
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>

#define EIGEN_DONT_VECTORIZE
#define EIGEN_NO_CUDA
#include <Eigen/Core>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

using matrix = Eigen::MatrixXd;

int main(int argc, char** argv) {
#ifdef GAUXC_HAS_MPI
  MPI_Init(NULL, NULL);
#endif
  {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " <input.hdf5> <model_path> [--device]" << std::endl;
#ifdef GAUXC_HAS_MPI
      MPI_Finalize();
#endif
      return 1;
    }

    std::string input_file = argv[1];
    std::string model_path = argv[2];
    bool use_device = false;
    for (int i = 3; i < argc; ++i) {
      if (std::string(argv[i]) == "--device") use_device = true;
    }

    // Create runtime — assign each MPI rank to a separate GPU
#ifdef GAUXC_HAS_DEVICE
    {
      int rank = 0;
      GAUXC_MPI_CODE(MPI_Comm_rank(MPI_COMM_WORLD, &rank);)
      int num_gpus = 0;
      cudaGetDeviceCount(&num_gpus);
      if (num_gpus > 0) cudaSetDevice(rank % num_gpus);
    }
    auto rt = GauXC::DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);
#else
    auto rt = GauXC::RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif

    auto world_rank = rt.comm_rank();
    auto world_size = rt.comm_size();

    // Load molecule and basis from HDF5
    GauXC::Molecule mol;
    GauXC::BasisSet<double> basis;
    GauXC::read_hdf5_record(mol, input_file, "/MOLECULE");
    GauXC::read_hdf5_record(basis, input_file, "/BASIS");

    // Load density matrices
    HighFive::File h5file(input_file, HighFive::File::ReadOnly);
    auto dset = h5file.getDataSet("/DENSITY_SCALAR");
    auto dims = dset.getDimensions();
    matrix P_s(dims[0], dims[1]);
    matrix P_z(dims[0], dims[1]);
    dset.read(P_s.data());
    h5file.getDataSet("/DENSITY_Z").read(P_z.data());

    if (!world_rank) {
      std::cout << "OneDFT Driver" << std::endl
                << "  Input      : " << input_file << std::endl
                << "  Model      : " << model_path << std::endl
                << "  Atoms      : " << mol.size() << std::endl
                << "  Basis fns  : " << basis.nbf() << std::endl
                << "  MPI ranks  : " << world_size << std::endl
#ifdef GAUXC_HAS_DEVICE
                << "  Exec space : " << (use_device ? "Device" : "Host") << std::endl
#else
                << "  Exec space : Host" << std::endl
#endif
                << std::endl;
    }

    // Choose execution space
#ifdef GAUXC_HAS_DEVICE
    auto exec_space = use_device ? GauXC::ExecutionSpace::Device
                                 : GauXC::ExecutionSpace::Host;
#else
    auto exec_space = GauXC::ExecutionSpace::Host;
#endif

    // Create molecular grid
    auto grid = GauXC::MolGridFactory::create_default_molgrid(
      mol, GauXC::PruningScheme::Robust,
      GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles,
      GauXC::AtomicGridSizeDefault::FineGrid);

    // Setup load balancer
    GauXC::LoadBalancerFactory lb_factory(exec_space, "Replicated");
    auto lb = lb_factory.get_shared_instance(rt, mol, grid, basis);

    // Apply molecular weights
    GauXC::MolecularWeightsFactory mw_factory(exec_space, "Default",
      GauXC::MolecularWeightsSettings{});
    auto mw = mw_factory.get_instance();
    mw.modify_weights(*lb);

    // Print grid statistics for debugging MPI differences
    {
      auto& tasks = lb->get_tasks();
      size_t local_npts = 0;
      double local_weight_sum = 0.0;
      std::vector<size_t> atom_npts(mol.size(), 0);
      for (const auto& task : tasks) {
        local_npts += task.npts;
        for (size_t i = 0; i < task.npts; ++i)
          local_weight_sum += task.weights[i];
        if (task.iParent >= 0 && task.iParent < (int)mol.size())
          atom_npts[task.iParent] += task.npts;
      }
      size_t global_npts = local_npts;
      double global_weight_sum = local_weight_sum;
#ifdef GAUXC_HAS_MPI
      MPI_Allreduce(MPI_IN_PLACE, &global_npts, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &global_weight_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      if (!world_rank) {
        std::cout << "Grid statistics:" << std::endl
                  << "  Total grid pts : " << global_npts << std::endl
                  << "  Weight sum     : " << std::fixed << std::setprecision(6) << global_weight_sum << std::endl
                  << "  Local pts (r0) : " << local_npts << std::endl
                  << "  Atoms per-pts  :";
        for (size_t a = 0; a < mol.size(); ++a) std::cout << " " << atom_npts[a];
        std::cout << std::endl << std::endl;
      }
    }

    // Setup integrator (empty functional — OneDFT provides its own)
    GauXC::functional_type func;
    GauXC::XCIntegratorFactory<matrix> integrator_factory(
      exec_space, "Replicated", "Default", "Default", "Default");
    auto integrator = integrator_factory.get_instance(func, lb);

    // Configure OneDFT model
    GauXC::OneDFTSettings onedft_settings;
    onedft_settings.model = model_path;

    // Warmup run
    if (!world_rank) std::cout << "Warmup run..." << std::endl;
    auto [EXC_w, VXC_w, VXCz_w] = integrator.eval_exc_vxc_onedft(P_s, P_z, onedft_settings);
    if (!world_rank) {
      std::cout << std::scientific << std::setprecision(12);
      std::cout << "  EXC          = " << EXC_w << " Eh" << std::endl
                << "  |VXC(a+b)|_F = " << VXC_w.norm() << std::endl
                << "  |VXC(a-b)|_F = " << VXCz_w.norm() << std::endl
                << std::endl;
    }

    // Timed run
#ifdef GAUXC_HAS_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    auto t0 = std::chrono::high_resolution_clock::now();

    auto [EXC, VXC_s, VXC_z] = integrator.eval_exc_vxc_onedft(P_s, P_z, onedft_settings);

#ifdef GAUXC_HAS_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    auto t1 = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(t1 - t0).count();

    if (!world_rank) {
      std::cout << "Timed run:" << std::endl;
      std::cout << std::scientific << std::setprecision(12);
      std::cout << "  EXC          = " << EXC << " Eh" << std::endl
                << "  |VXC(a+b)|_F = " << VXC_s.norm() << std::endl
                << "  |VXC(a-b)|_F = " << VXC_z.norm() << std::endl
                << "  Runtime      = " << std::fixed << std::setprecision(3) << dur << " s" << std::endl
                << std::endl;
    }

    // Verify warmup == timed (consistency check)
    if (!world_rank) {
      double exc_diff = std::abs(EXC - EXC_w);
      if (exc_diff > 1e-10) {
        std::cerr << "WARNING: EXC mismatch between runs: " << exc_diff << std::endl;
      } else {
        std::cout << "Consistency check passed (EXC matches across runs)" << std::endl;
      }
    }
  }
#ifdef GAUXC_HAS_MPI
  MPI_Finalize();
#endif
  return 0;
}
