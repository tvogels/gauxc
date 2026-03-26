/**
 * GauXC Benchmark — times multiple XC functionals on a molecule.
 *
 * Usage:
 *   benchmark <input.hdf5> [--model <path.fun>] [--device] [--iters N] [--warmup N] [--json]
 *   mpirun -np 2 benchmark <input.hdf5> [--model <path.fun>] [--device] [--json]
 *
 * The HDF5 file must contain: MOLECULE, BASIS, DENSITY_SCALAR, DENSITY_Z (UKS).
 */
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/xc_integrator_settings.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>
#ifdef GAUXC_HAS_DEVICE
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#define EIGEN_DONT_VECTORIZE
#define EIGEN_NO_CUDA
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <sstream>

using matrix = Eigen::MatrixXd;
using namespace GauXC;
using namespace ExchCXX;

struct BenchmarkResult {
  std::string name;
  std::string type;
  std::string description;
  double warmup_s;
  std::vector<double> times_s;

  double mean() const {
    return std::accumulate(times_s.begin(), times_s.end(), 0.0) / times_s.size();
  }
  double min() const {
    return *std::min_element(times_s.begin(), times_s.end());
  }
  double stddev() const {
    double m = mean();
    double sq = 0;
    for (auto t : times_s) sq += (t - m) * (t - m);
    return std::sqrt(sq / times_s.size());
  }
};

double barrier_time() {
#ifdef GAUXC_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  return std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

BenchmarkResult run_benchmark(const std::string& name,
                              std::function<void()> fn,
                              int warmup_iters, int timed_iters) {
  BenchmarkResult result;
  result.name = name;

  // Warmup
  double t0 = barrier_time();
  for (int i = 0; i < warmup_iters; ++i) fn();
  double t1 = barrier_time();
  result.warmup_s = (t1 - t0) / warmup_iters;

  // Timed
  for (int i = 0; i < timed_iters; ++i) {
    double s = barrier_time();
    fn();
    double e = barrier_time();
    result.times_s.push_back(e - s);
  }
  return result;
}

int main(int argc, char** argv) {
#ifdef GAUXC_HAS_MPI
  MPI_Init(NULL, NULL);
#endif
  {
    // Parse args
    std::string input_file;
    std::vector<std::pair<std::string, std::string>> models; // (name, path) pairs
    bool use_device = false;
    bool json_output = false;
    int iters = 5, warmup = 2;
    double gpu_mem_frac = 0.1;

    if (argc < 2) {
      std::cerr << "Usage: " << argv[0]
                << " <input.hdf5> [--model <name>:<path.fun>]... [--device] [--iters N] [--warmup N] [--json] [--gpu-mem-frac F]"
                << std::endl;
#ifdef GAUXC_HAS_MPI
      MPI_Finalize();
#endif
      return 1;
    }
    input_file = argv[1];
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--model" && i + 1 < argc) {
        std::string spec = argv[++i];
        auto colon = spec.find(':');
        if (colon != std::string::npos)
          models.push_back({spec.substr(0, colon), spec.substr(colon + 1)});
        else
          models.push_back({spec.substr(spec.rfind('/') + 1), spec});
      }
      else if (arg == "--device") use_device = true;
      else if (arg == "--json") json_output = true;
      else if (arg == "--gpu-mem-frac" && i + 1 < argc) gpu_mem_frac = std::stod(argv[++i]);
      else if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
      else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
    }

    // Runtime setup
#ifdef GAUXC_HAS_DEVICE
    {
      int rank = 0;
      GAUXC_MPI_CODE(MPI_Comm_rank(MPI_COMM_WORLD, &rank);)
      int num_gpus = 0;
      cudaGetDeviceCount(&num_gpus);
      if (num_gpus > 0) cudaSetDevice(rank % num_gpus);
    }
    auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) gpu_mem_frac);
#else
    auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif
    int world_rank = rt.comm_rank();
    int world_size = rt.comm_size();

#ifdef GAUXC_HAS_DEVICE
    auto exec_space = use_device ? ExecutionSpace::Device : ExecutionSpace::Host;
#else
    auto exec_space = ExecutionSpace::Host;
#endif

    // Load molecule, basis, density
    Molecule mol;
    BasisSet<double> basis;
    read_hdf5_record(mol, input_file, "/MOLECULE");
    read_hdf5_record(basis, input_file, "/BASIS");

    HighFive::File h5file(input_file, HighFive::File::ReadOnly);
    auto dset = h5file.getDataSet("/DENSITY_SCALAR");
    auto dims = dset.getDimensions();
    matrix P_s(dims[0], dims[1]), P_z(dims[0], dims[1]);
    dset.read(P_s.data());
    h5file.getDataSet("/DENSITY_Z").read(P_z.data());

    // Grid + load balancer + weights
    auto grid = MolGridFactory::create_default_molgrid(
      mol, PruningScheme::Robust, BatchSize(512),
      RadialQuad::MuraKnowles, AtomicGridSizeDefault::FineGrid);

    LoadBalancerFactory lb_factory(exec_space, "Replicated");
    auto lb = lb_factory.get_shared_instance(rt, mol, grid, basis);

    MolecularWeightsFactory mw_factory(exec_space, "Default", MolecularWeightsSettings{});
    auto mw = mw_factory.get_instance();
    mw.modify_weights(*lb);

    size_t total_npts = 0;
    for (const auto& task : lb->get_tasks()) total_npts += task.npts;
    size_t global_npts = total_npts;
#ifdef GAUXC_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_npts, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

    size_t natoms = mol.size();
    size_t nbf = basis.nbf();
    size_t nshells = basis.nshells();
    std::string exec_str = "Host";
#ifdef GAUXC_HAS_DEVICE
    if (use_device) exec_str = "Device";
#endif

    if (!world_rank && !json_output) {
      std::cout << "Benchmark: " << input_file << std::endl
                << "  Atoms    : " << natoms << std::endl
                << "  Basis fns: " << nbf << std::endl
                << "  Shells   : " << nshells << std::endl
                << "  Grid pts : " << global_npts << std::endl
                << "  MPI ranks: " << world_size << std::endl
                << "  Exec     : " << exec_str << std::endl
                << "  Warmup   : " << warmup << " iters" << std::endl
                << "  Timed    : " << iters << " iters" << std::endl
                << std::endl;
    }

    auto polar = Spin::Polarized; // Always UKS

    // Helper: create integrator for a given functional
    auto make_integrator = [&](functional_type func) {
      XCIntegratorFactory<matrix> factory(exec_space, "Replicated", "Default", "Default", "Default");
      return factory.get_instance(func, lb);
    };

    std::vector<BenchmarkResult> results;

    // --- Built-in XC functionals ---
    struct FuncSpec {
      std::string name;
      std::string type;
      std::string description;
      Functional func_enum;
      bool hybrid;
    };
    // --- Pure XC functionals ---
    std::vector<FuncSpec> func_specs = {
      {"SVWN5",  "LDA",         "XC energy + potential (eval_exc_vxc)",
       Functional::SVWN5, false},
      {"revPBE", "GGA",         "XC energy + potential (eval_exc_vxc)",
       Functional::revPBE, false},
      {"R2SCAN", "meta-GGA",    "XC energy + potential (eval_exc_vxc)",
       Functional::R2SCAN, false},
    };

    for (const auto& spec : func_specs) {
      auto func = functional_type(Backend::builtin, spec.func_enum, polar);
      auto integrator = make_integrator(func);
      auto res = run_benchmark(spec.name, [&]() {
        integrator.eval_exc_vxc(P_s, P_z);
      }, warmup, iters);
      res.type = spec.type;
      res.description = spec.description;
      results.push_back(std::move(res));
      if (!world_rank && !json_output) std::cout << "  " << spec.name << " done" << std::endl;
    }

    // --- OneDFT models (run before hybrids to avoid GPU memory pressure) ---
    for (const auto& [model_name, model_path] : models) {
      // Scope the integrator so its device data is freed between models
      try {
        functional_type empty_func;
        auto integrator = make_integrator(empty_func);
        OneDFTSettings onedft_settings;
        onedft_settings.model = model_path;

        auto res = run_benchmark(model_name, [&]() {
          integrator.eval_exc_vxc_onedft(P_s, P_z, onedft_settings);
        }, warmup, iters);
        res.type = "neural";
        res.description = "XC energy + potential via neural functional (eval_exc_vxc_onedft) [" + model_path + "]";
        results.push_back(std::move(res));

        if (!world_rank && !json_output)
          std::cout << "  " << model_name << " done" << std::endl;

        // Print internal timer breakdown for OneDFT
        if (!world_rank && !json_output) {
          std::cout << std::endl << model_name << " timer breakdown (last iteration):" << std::endl;
          auto& timings = integrator.get_timings().all_timings();
          for (const auto& [tname, dur] : timings) {
            std::cout << "  " << std::left << std::setw(45) << tname
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(10) << dur.count() << " ms" << std::endl;
          }
        }
      } catch (const std::exception& e) {
        if (!world_rank && !json_output)
          std::cout << "  " << model_name << " FAILED: " << e.what() << std::endl;
      }
#ifdef GAUXC_HAS_DEVICE
      c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    // --- Hybrid functionals (last — sn-K uses significant GPU memory) ---
    // Skip hybrids for large molecules to avoid GPU OOM from sn-K memory usage.
    if (nbf <= 2000) {
      std::vector<FuncSpec> hybrid_specs = {
        {"B3LYP",  "hybrid",      "XC potential + exact exchange matrix (eval_exc_vxc + eval_exx sn-LinK)",
         Functional::B3LYP, true},
        {"M06-2X", "meta-hybrid", "XC potential + exact exchange matrix (eval_exc_vxc + eval_exx sn-LinK)",
         Functional::M062X, true},
      };
      for (const auto& spec : hybrid_specs) {
        auto func = functional_type(Backend::builtin, spec.func_enum, polar);
        // XC part runs on the configured exec_space (GPU if --device)
        auto integrator = make_integrator(func);
        // sn-K: GPU only supports L<=2; fall back to Host for higher L
        auto exx_exec = exec_space;
        std::string desc = spec.description;
        if (basis.max_l() > 2 && use_device) {
          exx_exec = ExecutionSpace::Host;
          desc += " [sn-K on Host, GPU NYI for L>2]";
        }
        XCIntegratorFactory<matrix> exx_factory(exx_exec, "Replicated", "Default", "Default", "Default");
        auto exx_integrator = exx_factory.get_instance(func, lb);

        IntegratorSettingsSNLinK exx_settings;
        auto res = run_benchmark(spec.name, [&]() {
          integrator.eval_exc_vxc(P_s, P_z);
          exx_integrator.eval_exx(P_s, exx_settings);
        }, warmup, iters);
        res.type = spec.type;
        res.description = desc;
        results.push_back(std::move(res));
        if (!world_rank && !json_output) std::cout << "  " << spec.name << " done" << std::endl;
      }
    } else {
      if (!world_rank && !json_output) std::cout << "  Skipping hybrids (nbf=" << nbf << " > 2000)" << std::endl;
    }

    // --- Output results ---
    if (!world_rank) {
      if (json_output) {
        // JSON output — minimal hand-written JSON to avoid nlohmann dependency
        auto esc = [](const std::string& s) {
          std::string r;
          for (char c : s) {
            if (c == '"') r += "\\\"";
            else if (c == '\\') r += "\\\\";
            else r += c;
          }
          return r;
        };

        std::ostringstream js;
        js << std::setprecision(12);
        js << "{\n";
        js << "  \"input_file\": \"" << esc(input_file) << "\",\n";
        js << "  \"natoms\": " << natoms << ",\n";
        js << "  \"nbf\": " << nbf << ",\n";
        js << "  \"nshells\": " << nshells << ",\n";
        js << "  \"grid_pts\": " << global_npts << ",\n";
        js << "  \"mpi_ranks\": " << world_size << ",\n";
        js << "  \"exec_space\": \"" << exec_str << "\",\n";
        js << "  \"warmup_iters\": " << warmup << ",\n";
        js << "  \"timed_iters\": " << iters << ",\n";
        js << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
          const auto& r = results[i];
          js << "    {\n";
          js << "      \"name\": \"" << esc(r.name) << "\",\n";
          js << "      \"type\": \"" << esc(r.type) << "\",\n";
          js << "      \"description\": \"" << esc(r.description) << "\",\n";
          js << "      \"warmup_s\": " << r.warmup_s << ",\n";
          js << "      \"mean_s\": " << r.mean() << ",\n";
          js << "      \"min_s\": " << r.min() << ",\n";
          js << "      \"std_s\": " << r.stddev() << ",\n";
          js << "      \"times_s\": [";
          for (size_t j = 0; j < r.times_s.size(); ++j) {
            if (j > 0) js << ", ";
            js << r.times_s[j];
          }
          js << "]\n";
          js << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        js << "  ]\n";
        js << "}\n";
        std::cout << js.str();
      } else {
        // Table output
        std::cout << std::endl;
        std::cout << std::left << std::setw(30) << "Functional"
                  << std::right
                  << std::setw(12) << "Warmup (s)"
                  << std::setw(12) << "Mean (s)"
                  << std::setw(12) << "Std (s)"
                  << std::setw(12) << "Min (s)"
                  << std::endl;
        std::cout << std::string(78, '-') << std::endl;

        for (const auto& r : results) {
          std::cout << std::left << std::setw(30) << r.name
                    << std::right << std::fixed << std::setprecision(3)
                    << std::setw(12) << r.warmup_s
                    << std::setw(12) << r.mean()
                    << std::setw(12) << r.stddev()
                    << std::setw(12) << r.min()
                    << std::endl;
        }
      }
    }
  }
#ifdef GAUXC_HAS_MPI
  MPI_Finalize();
#endif
  return 0;
}
