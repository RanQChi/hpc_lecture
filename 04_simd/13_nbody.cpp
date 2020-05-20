#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i]=i;
  }
/**/
//--------------Change begin-----------
  __m256 xvec=_mm256_load_ps(x);
  __m256 yvec=_mm256_load_ps(y);
  __m256 mvec=_mm256_load_ps(m);
  __m256 jvex=_mm256_load_ps(j);
  
  
  for(int i=0; i<N; i++) {
    __m256 ivex=_mm256_set1_ps(i);    
    __m256 xveci=_mm256_set1_ps(x[i]);
    __m256 yveci=_mm256_set1_ps(y[i]);
    __m256 mask =_mm256_cmp_ps(ivex, jvex,_CMP_NEQ_OQ);
    
    __m256 xvecj=_mm256_setzero_ps();
    __m256 yvecj=_mm256_setzero_ps();
    __m256 mvecj =_mm256_setzero_ps();
    
     xvecj = _mm256_blendv_ps(xvecj, xvec, mask);
     yvecj = _mm256_blendv_ps(yvecj, yvec, mask);
     mvecj = _mm256_blendv_ps(mvecj, mvec, mask);

    __m256 rxvec = _mm256_sub_ps(xveci, xvecj);
    __m256 ryvec = _mm256_sub_ps(yveci, yvecj);
    __m256 rvec=_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec,rxvec),_mm256_mul_ps(ryvec,ryvec)));
    
    __m256 fxivec=_mm256_div_ps(_mm256_div_ps(_mm256_div_ps(_mm256_mul_ps(rxvec,mvecj),rvec),rvec),rvec);
    __m256 fyivec=_mm256_div_ps(_mm256_div_ps(_mm256_div_ps(_mm256_mul_ps(ryvec,mvecj),rvec),rvec),rvec);
		
     __m256 fxvec = _mm256_permute2f128_ps(fxivec,fxivec,1);
     fxvec =-_mm256_add_ps(fxvec,fxivec);
     fxvec = _mm256_hadd_ps(fxvec,fxvec);
     fxvec = _mm256_hadd_ps(fxvec,fxvec);
     
     __m256 fyvec = _mm256_permute2f128_ps(fyivec,fyivec,1);
     fyvec =-_mm256_add_ps(fyvec,fyivec);
     fyvec = _mm256_hadd_ps(fyvec,fyvec);
     fyvec = _mm256_hadd_ps(fyvec,fyvec);
		
    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
	
  }

//--------------Change end-----------
/**/
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
     if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
 
}
