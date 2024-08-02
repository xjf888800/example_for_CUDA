#include<bits/stdc++.h>
#include<sys/time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
using namespace std;

#define M 512 
#define K 512
#define N 512
#define BLOCK_SIZE 32 

void initial(float *array, int size){
    for(int i=0;i<size;i++){
        array[i] = (float)(rand() % 100 + 1);
    }
}

void printMatrix(float *array,int row,int col){
    float *p = array;
    for(int y=0;y<row;y++){
      for(int x=0;x<col;x++){
        printf("%10lf",p[x]);
      }
      p = p + col;
      printf("\n");
    }
}

void multiplicateMatrixonHost(float *array_A,float *array_B,float *array_C,int M_p,int K_p,int N_p){
   for(int i =0 ;i < M_p;i++){
    for(int j = 0;j < N_p;j++){
        float sum = 0;
        for(int k = 0;k < K_p;k++){
            sum += array_A[i * K_p + k] * array_B[k * N_p + j];
        }
        array_C[i * N_p + j] = sum;
    }
   }
}
void checkResult(float *array_A,float *array_B,int size){
    double epsilon = 1.0E-8;
    for(int i=0;i<size;i++){
        if(abs(array_A[i] - array_B[i]) > epsilon){
            printf("Error! Matrix[%05d]:%0.8f != %0.8f\n",i,array_A[i],array_B[i]);
            return;
        }
    }
    printf("Check result success!\n");
}

__global__ void multiplicateMatrixonDevice(float *array_A,float *array_B,float *array_C,int M_p,int K_p,int N_p){
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if(x<M_p && y<N_p){
    float sum =0;
    for(int i=0;i<K_p;i++){
      sum+=array_A[x*K_p+i]*array_B[i*N_p+y];
    }
    array_C[x*N_p+y] = sum;
  }
}

__global__ void matrixMultiplyShared(float *A, float *B, float *C, 
                                     int numARows,int numAColumns, 
                                     int numBRows,int numBColumns, 
                                     int numCRows,int numCColumns){
  __shared__ float ds_M[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float ds_N[BLOCK_SIZE][BLOCK_SIZE];
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x;
  int by=blockIdx.y;
  int rows=by*blockDim.y+ty;
  int cols=bx*blockDim.x+tx;
  int pvalue=0.0;
  for(int m=0;m<(numAColumns-1+blockDim.x)/blockDim.x;m++){
    if(rows<numARows && m*blockDim.x+tx<numAColumns){
      ds_M[ty][tx]=A[rows*numAColumns+m*blockDim.x+tx];
    }else{
      ds_M[ty][tx]=0.0;
    }
    if(cols<numBColumns && m*blockDim.y+ty<numBRows){
      ds_N[ty][tx]=B[(m*blockDim.y+ty)*numBColumns+cols];
    }else{
      ds_N[ty][tx]=0.0;
    }
    __syncthreads();
    for(int k=0;k<blockDim.x;k++){
      pvalue+=ds_M[ty][k]*ds_N[k][tx];
    }
    __syncthreads();
  }
  if(rows<numCRows && cols<numCColumns){
    C[rows*numCColumns+cols]=pvalue;
  }
}

int main(){
  float *array_A,*array_B,*array_C,*array_C_host; 
  float *d_arrayA,*d_arrayB,*d_arrayC;
  int size_A = M * K * sizeof(float);
  int size_B = K * N * sizeof(float);
  int size_C = M * N * sizeof(float);

  array_A=(float*)malloc(size_A);
  array_B=(float*)malloc(size_B);
  array_C=(float*)malloc(size_C);
  array_C_host=(float*)malloc(size_C);

  initial(array_A,M * K);
  initial(array_B,K * N);

  multiplicateMatrixonHost(array_A,array_B,array_C_host,M,K,N);

  cudaMalloc((void**)&d_arrayA,size_A);
  cudaMalloc((void**)&d_arrayB,size_B);
  cudaMalloc((void**)&d_arrayC,size_C);
  
  cudaMemcpy(d_arrayA,array_A,size_A,cudaMemcpyHostToDevice);
  cudaMemcpy(d_arrayB,array_B,size_B,cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid((N - 1)/ dimBlock.x + 1,(M - 1) / dimBlock.y + 1);

  multiplicateMatrixonDevice<<<dimGrid,dimBlock>>>(d_arrayA,d_arrayB, d_arrayC,M,K,N);
  cudaDeviceSynchronize();
  cudaMemcpy(array_C,d_arrayC,size_C,cudaMemcpyDeviceToHost);
  printf("mothed1: ");
  checkResult(array_C,array_C_host,M * N);
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      array_C[i*M+j]=0;
    }
  }
  cudaMemcpy(d_arrayC,array_C,size_C,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  matrixMultiplyShared<<<dimGrid,dimBlock>>>(d_arrayA,d_arrayB,d_arrayC,M,K,K,N,M,N);
  cudaDeviceSynchronize();
  cudaMemcpy(array_C,d_arrayC,size_C,cudaMemcpyDeviceToHost);
  printf("mothed2: ");
  checkResult(array_C,array_C_host,M * N);

  cudaFree(d_arrayA);
  cudaFree(d_arrayB);
  cudaFree(d_arrayC);
  free(array_A);
  free(array_B);
  free(array_C);
  free(array_C_host);
  return 0;
}
