#include <cuda_runtime.h>

#include <stdio.h>

// Kernel eseguito sulla GPU.
// Ogni thread calcola un solo elemento del vettore C.
__global__ void vectorAddKernel(float *A, float *B, float *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Alcuni thread potrebbero avere un indice oltre la fine del vettore.
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  const int n = 8;
  const int size = n * sizeof(float);

  // Dati sul lato host, cioe' nella memoria della CPU.
  float A_h[n] = {0, 1, 2, 3, 4, 5, 6, 7};
  float B_h[n] = {0, 2, 4, 6, 8, 10, 12, 14};
  float C_h[n] = {0};

  // Puntatori alla memoria del device, cioe' della GPU.
  float *A_d;
  float *B_d;
  float *C_d;

  // 1. Allochiamo memoria sulla GPU.
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  // 2. Copiamo gli input dalla CPU alla GPU.
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // 3. Lanciamo il kernel.
  // Usiamo 4 thread per blocco e 2 blocchi: in totale 8 thread.
  int threadsPerBlock = 4;
  int blocks = 2;
  vectorAddKernel<<<blocks, threadsPerBlock>>>(A_d, B_d, C_d, n);

  // Aspettiamo che la GPU abbia finito prima di leggere il risultato.
  cudaDeviceSynchronize();

  // 4. Copiamo il risultato dalla GPU alla CPU.
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // 5. Stampiamo e verifichiamo il risultato.
  for (int i = 0; i < n; i++) {
    printf("%.0f + %.0f = %.0f\n", A_h[i], B_h[i], C_h[i]);
  }

  // 6. Liberiamo la memoria allocata sulla GPU.
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
