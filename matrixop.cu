#include<stdio.h>
#include<time.h>
#include<stdint.h>
#include<stdlib.h>
#include<sys/time.h>

#define N 3001 // for huanhuan, you know that!
#define BLOCK_SIZE 32

__managed__ int input_Matrix[N][N];
__managed__ int output_GPU[N][N];
__managed__ int output_CPU[N][N];

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

void cpuop(int intput_M[N][N], int output_CPU[N][N])
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if(j%2==0 && i%2==0)
            {
                output_CPU[i][j] = intput_M[i][j]*intput_M[i][j];
            }
            else
            {
                output_CPU[i][j] = intput_M[i][j]-1;
            }
        }
    }
}
__global__ void gpuop(int input_M[N][N], int output_M[N][N]){
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    if(x<N&&y<N){
        if(x%2==0&&y%2==0){
            output_M[x][y]=input_M[x][y]*input_M[x][y];
        }else{
            output_M[x][y]=input_M[x][y]-1;
        }
    }
}

void init(int input_Matrix[N][N]){
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) 
        {
            input_Matrix[i][j] = rand()%3001;
        }
    }
}

void compared(int A[N][N],int B[N][N]){
    for(int i=0;i<N;i++){
        for(int j=0;j>N;j++){
            if(A[i][j]!=B[i][j]){
                printf("failed!!!\n");
                return;
            }
        }
    }
    printf("sucess!\n");
}

int main(){
    init(input_Matrix);
    printf("finish init!\n");
    cudaDeviceSynchronize();
    int n=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 blockshape(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridshape(n,n);
    gpuop<<<gridshape,blockshape>>>(input_Matrix,output_GPU);
    cudaDeviceSynchronize();
    cpuop(input_Matrix,output_CPU);
    compared(output_CPU,output_GPU);
    return 0;
}
