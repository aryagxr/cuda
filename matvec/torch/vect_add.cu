#include <torch/extension.h>
#include <vector>


__global__ void add_vectors(const int *A, const int *B, int *C, int size){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < size){
    C[idx] = A[idx] + B[idx];
  }
}


// wrap in python callable wrapper
torch::Tensor add_vectors_cuda(torch::Tensor A, torch::Tensor B){

  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.sizes() == B.sizes(), "A and B must have same shapes");


  auto C = torch::empty_like(A);
  int size = A.numel();
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;
  add_vectors<<<blocks, threads>>>(
    A.data_ptr<int>(),
    B.data_ptr<int>(),
    C.data_ptr<int>(),
    size
  );

  return C;

}


// pybind module registration
// m.def(.cu file name, &<torch wrapper name>, desc)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("vect_add", &add_vectors_cuda, "vector addition");
}

