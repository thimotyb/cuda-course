# cuBLAS Matrix Multiplication Comparison

Questo esempio confronta tre implementazioni della stessa moltiplicazione di matrici quadrate:

- triplo `for` su CPU;
- tiled CUDA kernel didattico con shared memory;
- `cuBLAS cublasSgemm`.

Lo scopo non e' sostituire la spiegazione del tiling, ma mostrare quanto una libreria vendor ottimizzata possa guadagnare rispetto a un kernel scritto per essere leggibile.

## 1. Entrare nella cartella

```bash
cd examples/ch04/cublas_matrix_mul_compare
```

## 2. Compilare

Serve linkare cuBLAS:

```bash
nvcc cublas_matrix_mul_compare.cu -O3 -lineinfo -lcublas -o cublas_matrix_mul_compare
```

## 3. Eseguire

Run default:

```bash
./cublas_matrix_mul_compare
```

Parametri:

```text
./cublas_matrix_mul_compare [Width] [iterations] [default|pedantic]
```

- `Width`: lato delle matrici quadrate, default `1024`.
- `iterations`: ripetizioni misurate con CUDA events, default `20`.
- `default`: lascia cuBLAS nella modalita' ottimizzata predefinita.
- `pedantic`: forza una modalita' FP32 piu' stretta, utile per discutere precisione vs prestazioni.

Esempi:

```bash
./cublas_matrix_mul_compare 1024 20 default
./cublas_matrix_mul_compare 1024 20 pedantic
```

`Width` deve essere divisibile per `TILE_WIDTH`, come nell'esempio tiled senza boundary checks.

## 4. Cosa osservare

Output principale:

```text
CPU triple-loop time: ... ms
CPU triple-loop throughput: ... GFLOP/s
tiled kernel time: ... ms
tiled kernel throughput: ... GFLOP/s
cuBLAS SGEMM time: ... ms
cuBLAS SGEMM throughput: ... GFLOP/s
cuBLAS speedup vs tiled kernel: ...x
cuBLAS speedup vs CPU triple loop: ...x
```

Il confronto piu' importante per M4 e':

```text
cuBLAS speedup vs tiled kernel
```

Questo valore mostra quanto resta sul tavolo quando il kernel e' corretto e usa shared memory, ma non implementa tutte le ottimizzazioni di una libreria industriale.

Esempio misurato localmente su `NVIDIA GeForce RTX 5060 Ti` con `Width=1024`, `iterations=10`, modalita' `default`:

```text
tiled kernel time: 1.161437 ms
tiled kernel throughput: 1848.99 GFLOP/s
cuBLAS SGEMM time: 0.164579 ms
cuBLAS SGEMM throughput: 13048.33 GFLOP/s
cuBLAS speedup vs tiled kernel: 7.06x
```

Su matrici piccole il confronto puo' essere meno favorevole per cuBLAS, perche' overhead, scelta del kernel interno e modalita' matematica pesano di piu'. Per una demo in aula usare almeno `Width=1024`.

## 5. Nota su row-major e cuBLAS

Il codice del corso usa matrici row-major, come in C/C++:

```text
M[row * Width + col]
```

cuBLAS assume matrici column-major. Per calcolare lo stesso risultato senza trasporre fisicamente i dati, il programma usa la relazione:

```text
(M_row * N_row)^T = N^T * M^T
```

Quindi la chiamata cuBLAS scambia l'ordine degli operandi:

```cpp
cublasSgemm(..., device_N, ..., device_M, ..., device_P, ...)
```

La memoria e' la stessa, ma viene interpretata dal punto di vista column-major.

## 6. Nota su TF32

Sulle GPU NVIDIA moderne, la modalita' cuBLAS predefinita puo' usare percorsi ottimizzati come TF32/Tensor Core per `SGEMM`, a seconda di architettura, toolkit e impostazioni runtime.

Per questo l'esempio permette due modalita':

```bash
./cublas_matrix_mul_compare 1024 20 default
./cublas_matrix_mul_compare 1024 20 pedantic
```

La modalita' `default` rappresenta meglio "quanto va forte cuBLAS lasciato libero di ottimizzare". La modalita' `pedantic` e' piu' utile quando si vuole discutere una semantica FP32 piu' stretta.

## 7. Pulizia

```bash
rm -f cublas_matrix_mul_compare
```
