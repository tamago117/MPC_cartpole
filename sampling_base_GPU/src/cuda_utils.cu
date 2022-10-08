#include "cuda_utils.cuh"

unsigned int CountBlocks(unsigned int thread_num, unsigned int thread_per_block)
{
    unsigned int num;
    num = thread_num / thread_per_block;
    if (thread_num < thread_per_block || thread_num % thread_per_block > 0){
        num++;
    }
    return num;
}

__global__ void SetRandomSeed(curandState *random_seed_vec, int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &random_seed_vec[id]);
}