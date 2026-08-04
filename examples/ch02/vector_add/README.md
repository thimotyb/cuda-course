# Vector Add

Questo esempio verifica che la GPU esegua un kernel utile: somma due vettori elemento per elemento.

Rispetto a `simple_kernel_launch`, qui il programma:

- usa tre piccoli vettori sulla CPU;
- alloca memoria sulla GPU con `cudaMalloc`;
- copia gli input dalla CPU alla GPU con `cudaMemcpy`;
- lancia un kernel CUDA con 8 thread;
- copia il risultato dalla GPU alla CPU;
- stampa il risultato sul lato host.

## Compilazione

```bash
nvcc vector_add.cu -o vector_add
```

## Esecuzione

```bash
./vector_add
```

Output atteso, con i primi valori stampati:

```text
0 + 0 = 0
1 + 2 = 3
2 + 4 = 6
3 + 6 = 9
4 + 8 = 12
5 + 10 = 15
6 + 12 = 18
7 + 14 = 21
```

## Punti da spiegare a lezione

- `A_h`, `B_h` e `C_h` sono array nella memoria host, quindi CPU.
- `A_d`, `B_d` e `C_d` sono puntatori a memoria device, quindi GPU.
- `cudaMalloc` alloca memoria sulla GPU.
- `cudaMemcpy(..., cudaMemcpyHostToDevice)` copia dati dalla CPU alla GPU.
- Il kernel calcola l'indice globale con `blockIdx.x * blockDim.x + threadIdx.x`.
- Il controllo `if (i < n)` evita accessi fuori dai limiti quando il numero totale di thread supera la lunghezza del vettore.
- `vectorAddKernel<<<2, 4>>>(...)` lancia 2 blocchi con 4 thread ciascuno, quindi 8 thread totali.
- `cudaDeviceSynchronize()` aspetta che la GPU abbia finito prima di copiare e verificare i risultati.
- `cudaMemcpy(..., cudaMemcpyDeviceToHost)` copia il risultato dalla GPU alla CPU.
- `cudaFree` libera la memoria allocata sulla GPU.
