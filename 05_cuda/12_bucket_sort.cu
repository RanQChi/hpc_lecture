#include <cstdio>
#include <cstdlib>
#include <vector>
//----------Change Begin----------------
__global__ void for_1(int *bucket_cu) {
	int i=blockIdx.x * blockDim.x + threadIdx.x;
	bucket_cu[i]=0;
	//printf("%d\n",i);
}

__global__ void for_2(int n,int *bucket_cu,int *key){
	int i=blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=n) return;
	atomicAdd(&bucket_cu[key[i]], 1);
}

__global__ void set_index(int range,int *index,int *bucket_cu){
	int i=blockIdx.x * blockDim.x + threadIdx.x;
	for(int j=0;j<range;j++){
		index[i+1] = bucket_cu[i] + index[i];
		__syncthreads();
	}
}

__global__ void for_3(int *index,int range,int n,int *bucket_cu,int *key){
	int i=blockIdx.x * blockDim.x + threadIdx.x;

	if(i>=n) return;
	for (int j=0;j<range;j++){
		if (i<index[j+1] && i>=index[j]){
			key[i]=j;
			return;
		}
	}
}

//---------- Change End ----------------
int main() {
  int n = 50;
  int range = 5;
  //std::vector<int> key(n);
  //----------Change Begin----------------
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  //----------Change End----------------
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  
//----------Change Begin----------------
	const int M=1024;
	int *bucket_cu;
	cudaMallocManaged(&bucket_cu, range*sizeof(int));
	int *index;
	cudaMallocManaged(&index, (1+range)*sizeof(int));
	
	for_1<<<(range+M-1)/M,M>>>(bucket_cu);
	cudaDeviceSynchronize();
	
	for_2<<<(n+M-1)/M,M>>>(n,bucket_cu,key);
	cudaDeviceSynchronize();
	
	set_index<<<(range+M-1)/M,M>>>(range,index,bucket_cu);
	cudaDeviceSynchronize();
	
	for_3<<<(n+M-1)/M,M>>>(index,range,n,bucket_cu,key);
	cudaDeviceSynchronize();
//---------- Change End ----------------

/*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
