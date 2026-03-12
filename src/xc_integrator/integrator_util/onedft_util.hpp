#include <gauxc/gauxc_config.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#ifdef GAUXC_HAS_CUDA
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#endif
#include <nlohmann/json.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>

using json = nlohmann::json;
using IValueList = std::vector<c10::IValue>;
using IValueMap = std::unordered_map<std::string, c10::IValue>;
using FeatureDict = c10::Dict<std::string, at::Tensor>;

namespace GauXC {

  //  custom allocator for torch::Tensor  
  // enum store onedft feature keys and our feature keys
  // TODO add laplacian ?
  void print_memory_stats(size_t device_id);
  
  enum ONEDFT_FEATURE { DEN, DDEN, TAU, POINTS, WEIGHTS, COORDS, ATOMIC_GRID_WEIGHTS, ATOMIC_GRID_SIZES, ATOMIC_GRID_SIZE_BOUND_SHAPE };

  // Mapping enums to string values
  inline const std::map<ONEDFT_FEATURE, std::string> feat_map = {
    {DEN, "density"},
    {DDEN, "grad"},
    {TAU, "kin"},
    {POINTS, "grid_coords"},
    {WEIGHTS, "grid_weights"},
    {COORDS, "coarse_0_atomic_coords"},
    {ATOMIC_GRID_WEIGHTS, "atomic_grid_weights"},
    {ATOMIC_GRID_SIZES, "atomic_grid_sizes"},
    {ATOMIC_GRID_SIZE_BOUND_SHAPE, "atomic_grid_size_bound_shape"}
  };

  inline const std::map<std::string, ONEDFT_FEATURE> reverse_feat_map = {
    {"density", DEN},
    {"grad", DDEN},
    {"kin", TAU},
    {"grid_coords", POINTS},
    {"grid_weights", WEIGHTS},
    {"coarse_0_atomic_coords", COORDS},
    {"atomic_grid_weights", ATOMIC_GRID_WEIGHTS},
    {"atomic_grid_sizes", ATOMIC_GRID_SIZES},
    {"atomic_grid_size_bound_shape", ATOMIC_GRID_SIZE_BOUND_SHAPE}
  };
  
int mpi_scatter_onedft_outputs(const FeatureDict features_dict,
                          const int world_rank, const int world_size,
                          std::vector<int> recvcounts, std::vector<int> displs,
                          const std::vector<int64_t>& atom_reorder_inv_perm,
                          std::vector<double>& den_eval, std::vector<double>& dden_eval, 
                          std::vector<double>& tau);

int mpi_gather_onedft_inputs(std::vector<double>& den_eval, std::vector<double>& dden_eval,
                          std::vector<double>& tau, std::vector<double>& grid_coords,
                          std::vector<double>& grid_weights, const int total_npts,
                          const int world_rank, const int world_size,
                          std::vector<int>& sendcounts, std::vector<int>& displs);
                          
int mpi_gather_onedft_inputs_gpu(std::vector<double>& den_eval, std::vector<double>& dden_eval,
                          std::vector<double>& tau, std::vector<double>& grid_coords,
                          std::vector<double>& grid_weights, const int total_npts,
                          const int world_rank, const int world_size,
                          std::vector<int>& recvcounts, std::vector<int>& displs) ;
  bool valueExists(const std::string& value);

  std::tuple<torch::jit::Method, std::vector<std::string>>
    load_model(const std::string filename, torch::DeviceType device);

  at::Tensor
    get_exc(torch::jit::Method exc_func, FeatureDict features);

  // Build a permutation that reorders gathered MPI data from rank-order to atom-order.
  // all_rank_atom_sizes: [world_size * natoms], row-major (rank-major).
  // sendcounts/displs: per-rank point counts and displacements from MPI_Gatherv.
  // Returns (perm, inv_perm) where perm[rank_ordered_idx] = atom_ordered_idx.
  std::pair<std::vector<int64_t>, std::vector<int64_t>>
    build_atom_reorder_perm(const std::vector<int64_t>& all_rank_atom_sizes,
                            const std::vector<int>& sendcounts,
                            const std::vector<int>& displs,
                            int natoms, int world_size);

  // Apply a point-level permutation to a strided array.
  // For each point i, copies stride elements from src[perm[i]*stride .. +stride)
  // to dst[i*stride .. +stride).
  void apply_strided_permutation(const double* src, double* dst,
                                 const std::vector<int64_t>& perm,
                                 int64_t npts, int stride);
} // namespace GauXC