#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<sys/time.h>

#define N 10000000
#define BLOCK_SIZE 256
#define BLOCK ((N+BLOCK_SIZE-1)/BLOCK_SIZE)

__managed__ int source[N];
__managed__ int partarry[2 * BLOCK_SIZE];
__managed__ int final_result[2]={0,0};

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

typedef struct
{
    int min;
    int max;
}cpu_result_t;	
	
cpu_result_t _hawk_minmax_cpu(int *ptr, int count)
{
    int YZP_min = INT_MAX;
    int YZP_max = INT_MIN;
    for (int i = 0; i < count; i++)
    {
	YZP_min = min(YZP_min, ptr[i]);
	YZP_max = max(YZP_max, ptr[i]);
    }

    cpu_result_t r;
    {
	r.min = YZP_min;
	r.max = YZP_max;
    }		
    return r;
}


__global__ void find_max_min(int* input,int n,int* output){
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int allid=gridDim.x*blockDim.x;
    __shared__ int partmax[BLOCK_SIZE];
    __shared__ int partmin[BLOCK_SIZE];
    int inmax=INT_MIN; 
    int inmin=INT_MAX;
    for(int idx=tid;idx<n;idx+=allid){
        inmax=max(input[idx],inmax);
        inmin=min(input[idx],inmin);
    }
    partmax[threadIdx.x] = inmax;
    partmin[threadIdx.x] = inmin;
    __syncthreads();
    for(int length=BLOCK_SIZE/2;length>=1;length/=2){
        if(threadIdx.x<length){
            partmax[threadIdx.x] = max(partmax[threadIdx.x],partmax[threadIdx.x+length]);
            partmin[threadIdx.x] = min(partmin[threadIdx.x],partmin[threadIdx.x+length]);
        }
        __syncthreads();
    }
    if(blockDim.x * blockIdx.x <n){
        if(threadIdx.x==0){
            output[2 * blockIdx.x + 0] = partmin[0]; 
	        output[2 * blockIdx.x + 1] = partmax[0];
        }
    }
}

int main(){
  _nanana_init(source,N);
  printf("finish init");
  cudaDeviceSynchronize();
  find_max_min<<<BLOCK,BLOCK_SIZE>>>(source,N,partarry);
  cudaDeviceSynchronize();
  find_max_min<<<1,BLOCK_SIZE>>>(partarry,2 * BLOCK,final_result);
  cudaDeviceSynchronize();
  int A0 = final_result[0];
  int A1 = final_result[1];
  fprintf(stderr, "GPU min: %d, max: %d\n", A0, A1);

  cpu_result_t B = _hawk_minmax_cpu(source, N);
  fprintf(stderr, "CPU min: %d, max: %d\n", B.min, B.max);

  return 0;
}
    

