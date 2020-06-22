#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#define M 41


using namespace std;

typedef vector<vector<float>> matrix;

__global__ void initial(float *u,float *v,float *p,float *b,int nx,int ny){
    int i = blockIdx.x / (ny / M);
    int j = threadIdx.x + blockDim.x * (blockIdx.x % (nx / M));
    
    u[ny*i+j]=0.0;
    v[ny*i+j]=0.0;
    p[ny*i+j]=0.0;
    b[ny*i+j]=0.0;
}


__global__ void build_up_b(float *b,float rho,float dt,float *u,float *v,float dx,float dy,int nx,int ny) {
    int i = blockIdx.x / (ny / M);
    int j = threadIdx.x + blockDim.x * (blockIdx.x % (nx / M));
     
    if(i>0&&i<nx-1&&j>0&&j<ny-1){
    	b[ny*i+j]=(rho*(1.0/dt*
    				((u[i*ny+j+1]-u[i*ny+j-1])/
    				 (2*dx)+(v[(i+1)*ny+j]-v[(i-1)*ny+j])/(2*dy))-
    		 		((u[i*ny+j+1]-u[i*ny+j-1])/(2*dx))*((u[i*ny+j+1]-u[i*ny+j-1])/(2*dx))-
    		 		2*((u[(i+1)*ny+j]-u[(i-1)*ny+j])/(2*dy)*
    		 		   (v[i*ny+j+1]-v[i*ny+j-1])/(2*dx))-
    		 		((v[(i+1)*ny+j]-v[(i-1)*ny+j])/(2*dy))*((v[(i+1)*ny+j]-v[(i-1)*ny+j])/(2*dy))
    			)
    		   );
    }
     __syncthreads();    
}

__global__ void pressure_poisson(float *p,float dx,float dy,float *b,int nx,int ny,float *pn) {
    int i = blockIdx.x / (ny / M);
    int j = threadIdx.x + blockDim.x * (blockIdx.x % (nx / M));
     
    if(i>0&&i<nx-1&&j>0&&j<ny-1){
        p[ny*i+j]=(((pn[i*ny+j+1] + pn[i*ny+j-1])*dy*dy+
                    (pn[(i+1)*ny+j] + pn[(i-1)*ny+j])*dx*dx)/
                    (2*(dx*dx+dy*dy))-
                   dx*dx*dy*dy/(2*(dx*dx+dy*dy))*b[i*ny+j]
                  );
    }
    __syncthreads();     
}
__global__ void pressure_poisson_2(float *p,int nx,int ny) {
    int i = blockIdx.x / (ny / M);
    int j = threadIdx.x + blockDim.x * (blockIdx.x % (nx / M));
    if(j==ny-1){
        p[ny*i+j]=p[ny*i+j-1]; 
    } 
    if(i==0){
        p[ny*i+j]=p[(i+1)*ny+j]; 
    }
    if(j==0){
        p[ny*i+j]=p[i*ny+j+1]; 
    }
    if(i==nx-1){
        p[ny*i+j] = 0.0;
    }
    __syncthreads();     
}

__global__ void cavity_flow(int nt,float *u,float *v,float dt,float dx,float dy,float*p,float rho,float nu,int nx,int ny,float *un,float *vn) {
    
    int i = blockIdx.x / (ny / M);
    int j = threadIdx.x + blockDim.x * (blockIdx.x % (nx / M));

    if(i>0&&i<nx-1&&j>0&&j<ny-1){
        u[i*ny+j]=(un[i*ny+j]-
                   un[i*ny+j]*dt/dx*
                   (un[i*ny+j]-un[i*ny+j-1])-
                   vn[i*ny+j]*dt/dy*
                   (un[i*ny+j]-un[(i-1)*ny+j])-
                   dt/(2*rho*dx)*(p[i*ny+j+1]-p[i*ny+j-1])+
                   nu*(dt/(dx*dx)*
                       (un[i*ny+j+1]-2*un[i*ny+j]+un[i*ny+j-1])+
                       dt/(dy*dy)*
                       (un[(i+1)*ny+j]-2*un[i*ny+j]+un[(i-1)*ny+j])
                      )
                  );
                  
        v[i*ny+j]=(vn[i*ny+j]-
                   un[i*ny+j]*dt/dx*
                   (vn[i*ny+j]-vn[i*ny+j-1])-
                   vn[i*ny+j]*dt/dy*
                   (vn[i*ny+j]-vn[(i-1)*ny+j])-
                   dt/(2*rho*dx)*(p[(i+1)*ny+j]-p[(i-1)*ny+j])+
                   nu*(dt/(dx*dx)*
                       (vn[i*ny+j+1]-2*vn[i*ny+j]+vn[i*ny+j-1])+
                       dt/(dy*dy)*
                       (vn[(i+1)*ny+j]-2*vn[i*ny+j]+vn[(i-1)*ny+j])
                      )
                  );
    }
    __syncthreads();
}
__global__ void cavity_flow_2(float *u,float *v,int nx,int ny) {
    int i = blockIdx.x / (ny / M);
    int j = threadIdx.x + blockDim.x * (blockIdx.x % (nx / M));
    
    if(i==0){
        u[i*ny+j]=0.0;
        v[i*ny+j]=0.0;
    } 
    if(j==0){
    	u[i*ny+j]=0.0;
    	v[i*ny+j]=0.0;
    }
    if(j==ny-1){
    	u[i*ny+j]=0.0;
    	v[i*ny+j]=0.0;
    }
    if(i==nx-1){
        u[i*ny+j]=1.0;
        v[i*ny+j]=0.0;
    }
    __syncthreads();     
}


int main() {
    
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;

    float dx = 2.0/(nx-1);
    float dy = 2.0/(ny-1);
    
    float rho = 1.0;
    float nu = 0.1;
    float dt = 0.001;
    
    int size = nx * ny * sizeof(float);
    
    float *u,*v,*p,*b;
    cudaMallocManaged(&u, size);
    cudaMallocManaged(&v, size);
    cudaMallocManaged(&p, size);
    cudaMallocManaged(&b, size);   
    
    float *pn;
    cudaMallocManaged(&pn, size);
    
    float *un,*vn;
    cudaMallocManaged(&un, size);
    cudaMallocManaged(&vn, size);
    
//-------------------nt=100----------------------------------
    nt=100;  
    initial<<<nx*ny/M,M>>>(u,v,p,b,nx,ny);
    cudaDeviceSynchronize();
    for (int nt_index=0;nt_index<nt;nt_index++){
        un=u;
        vn=v;
        
        build_up_b<<<nx*ny/M,M>>>(b,rho,dt,u,v,dx,dy,nx,ny);
        cudaDeviceSynchronize();
    
        for (int nit_index=0;nit_index<nit;nit_index++){
            pn=p;
            pressure_poisson<<<nx*ny/M,M>>>(p,dx,dy,b,nx,ny,pn);
            pressure_poisson_2<<<nx*ny/M,M>>>(p,nx,ny);
        }
        cudaDeviceSynchronize();
        
        cavity_flow<<<nx*ny/M,M>>>(nt,u,v,dt,dx,dy,p,rho,nu,nx,ny,un,vn);
        cavity_flow_2<<<nx*ny/M,M>>>(u,v,nx,ny);
        cudaDeviceSynchronize();
    
    }
    
    
   ofstream outFile_u_100,outFile_v_100,outFile_p_100;
   outFile_u_100.open("./Data/u_data_100.csv", ios::out);
   outFile_v_100.open("./Data/v_data_100.csv", ios::out);
   outFile_p_100.open("./Data/p_data_100.csv", ios::out);
   for (int i=0;i<nx;i++){
        for (int j=0;j<ny;j++){
            outFile_u_100<<u[ny*i+j]<<',';
            outFile_v_100<<v[ny*i+j]<<',';
            outFile_p_100<<p[ny*i+j]<<',';
        }
        outFile_u_100<<endl;
        outFile_v_100<<endl;
        outFile_p_100<<endl;
    }
//-------------------nt=700----------------------------------
    nt=700;  
    initial<<<nx*ny/M,M>>>(u,v,p,b,nx,ny);
    cudaDeviceSynchronize();
    for (int nt_index=0;nt_index<nt;nt_index++){
        un=u;
        vn=v;
        
        build_up_b<<<nx*ny/M,M>>>(b,rho,dt,u,v,dx,dy,nx,ny);
        cudaDeviceSynchronize();
    
        for (int nit_index=0;nit_index<nit;nit_index++){
            pn=p;
            pressure_poisson<<<nx*ny/M,M>>>(p,dx,dy,b,nx,ny,pn);
            pressure_poisson_2<<<nx*ny/M,M>>>(p,nx,ny);
        }
        cudaDeviceSynchronize();
        
        cavity_flow<<<nx*ny/M,M>>>(nt,u,v,dt,dx,dy,p,rho,nu,nx,ny,un,vn);
        cavity_flow_2<<<nx*ny/M,M>>>(u,v,nx,ny);
        cudaDeviceSynchronize();
    
    }
    
    
    ofstream outFile_u_700,outFile_v_700,outFile_p_700;
    outFile_u_700.open("./Data/u_data_700.csv", ios::out);
    outFile_v_700.open("./Data/v_data_700.csv", ios::out);
    outFile_p_700.open("./Data/p_data_700.csv", ios::out);
    for (int i=0;i<nx;i++){
        for (int j=0;j<ny;j++){
            outFile_u_700<<u[ny*i+j]<<',';
            outFile_v_700<<v[ny*i+j]<<',';
            outFile_p_700<<p[ny*i+j]<<',';
        }
        outFile_u_700<<endl;
        outFile_v_700<<endl;
        outFile_p_700<<endl;
    }
    
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    
    cudaFree(pn);
    cudaFree(un);
    cudaFree(vn);
}
