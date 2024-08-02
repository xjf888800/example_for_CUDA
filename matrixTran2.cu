#include<stdio.h>
#include<time.h>
#include<stdint.h>
#include<stdlib.h>
#include<sys/time.h>

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

#define M 3001  //three thousand and one nights
#define TILE_SIZE 32
__managed__ int shark[M][M];      //input matrix
__managed__ int gpu_shark_T[M][M];//GPU result
__managed__ int cpu_shark_T[M][M];//CPU result

void _sparks_transpose_cpu(int A[M][M], int B[M][M])
{
    for (int j = 0; j < M; j++)
    {
	for (int i = 0; i < M; i++)
	{
	    B[i][j] = A[j][i];
	}
    }
}

void DDBDDH_init(int A[M][M])
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the matrix with random data
    for (int j = 0; j < M; j++)
    {
	for (int i = 0; i < M; i++)
	{
	    A[j][i] = rand();
	}
    }
}
void compared(int A[M][M],int B[M][M]){
    for(int i=0;i<M;i++){
        for(int j=0;j>M;j++){
            if(A[i][j]!=B[i][j]){
                printf("failed!!!\n");
                return;
            }
        }
    }
    printf("sucess!\n");
}
__global__ void _ZHI_transpose(int A[M][M], int B[M][M])
{
    __shared__ int rafa[TILE_SIZE][TILE_SIZE+1];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x<M && y<M){
        rafa[threadIdx.y][threadIdx.x]=A[y][x];
    }
    __syncthreads();
    int y2 = threadIdx.y + blockDim.x * blockIdx.x;
    int x2 = threadIdx.x + blockDim.y * blockIdx.y;
    if(x2<M && y2<M){
        B[y2][x2]=rafa[threadIdx.x][threadIdx.y];
    }
}
int main(){
    DDBDDH_init(shark);
    printf("finish init\n");
    cudaDeviceSynchronize();
    
    int n = (M + TILE_SIZE - 1) / TILE_SIZE; //what the hell is this!
    dim3 grid_shape(n, n);
    dim3 block_shape(TILE_SIZE, TILE_SIZE);
    _ZHI_transpose<<<grid_shape, block_shape>>>(shark, gpu_shark_T);
        KEN_CHECK(cudaGetLastError());  //checking for launch failures
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs

    _sparks_transpose_cpu(shark, cpu_shark_T);

    //******The last judgement**********
    compared(gpu_shark_T,cpu_shark_T);
    return 0;
}


