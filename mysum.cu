#include<stdio.h>
#include<stdint.h>
#include<time.h>
#include<stdlib.h>
#include<sys/time.h>

#define N 10000000
#define BLOCK_SIZE 256
#define BLOCKS ((N + BLOCK_SIZE -1) / BLOCK_SIZE)

__managed__ int source[N];
__managed__ int part_arry[BLOCK_SIZE];
__managed__ int final_result[1]={0};

#define KEN_CHECK(r) \
{\
    cudaError_t rr = r;   \
    if (rr != cudaSuccess)\
    {\
        fprintf(stderr, "CUDA Error %s, function: %s, line: %d\n",       \
		        cudaGetErrorString(rr), __FUNCTION__, __LINE__); \
        exit(-1);\
    }\
}
void _nanana_init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    for (int i = 0; i < count; i++) ptr[i] = rand();
}
__global__ void sumgpu(int *input,int count,int *output){
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    int allthread=gridDim.x*blockDim.x;
    __shared__ int result[BLOCK_SIZE];
    int part=0;
    for(int idx=tid;idx<count;idx+=allthread){
        part=part+input[idx];
    }
    result[threadIdx.x]=part;
    __syncthreads();
    for(int length = BLOCK_SIZE/2;length>=1;length/=2){
        if(threadIdx.x < length){
            result[threadIdx.x]=result[threadIdx.x]+result[threadIdx.x+length];
        }
        __syncthreads();
    }
    if(tid<count){
        if(threadIdx.x==0){
            output[blockIdx.x]=result[0];
        }
    }
}
int sum_cpu(int *ptr, int count)
{
    int sum = 0;
    for (int i = 0; i < count; i++)
    {
        sum += ptr[i];
    }
    return sum;
}

int main(){
    _nanana_init(source,N);
    printf("finish init!!");
    KEN_CHECK(cudaDeviceSynchronize())//steady
    sumgpu<<<BLOCKS,BLOCK_SIZE>>>(source,N,part_arry);
    sumgpu<<<1,BLOCK_SIZE>>>(part_arry,BLOCKS,final_result);
    KEN_CHECK(cudaDeviceSynchronize())
    int A = final_result[0];
    fprintf(stderr, "GPU sum: %u\n", A);
    int B = sum_cpu(source,N);
    fprintf(stderr, "CPU sum: %u\n", B);

}