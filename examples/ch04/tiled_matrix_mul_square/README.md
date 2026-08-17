# Tiled Matrix Multiplication, Square Version

Questo esempio accompagna M4 e implementa la moltiplicazione di matrici quadrate con tiling e shared memory.

La versione e' intenzionalmente quella semplice da spiegare a lezione:

- matrici quadrate `Width x Width`;
- `Width` deve essere multiplo di `TILE_WIDTH`;
- ogni blocco CUDA calcola una tile `TILE_WIDTH x TILE_WIDTH` della matrice `P`;
- ogni thread calcola un elemento di `P`;
- non ci sono boundary checks.

La parte kernel mantiene i nomi didattici del libro: `M`, `N`, `P`, `Mds`, `Nds`, `Pvalue`, `Width`, `Row`, `Col`, `ph`.

## 1. Entrare nella cartella

```bash
cd examples/ch04/tiled_matrix_mul_square
```

## 2. Compilare

Compilazione default, con `TILE_WIDTH=32`:

```bash
nvcc tiled_matrix_mul_square.cu -O3 -lineinfo -o tiled_matrix_mul_square
```

Per usare tile piu' piccole, ricompilare impostando la macro:

```bash
nvcc tiled_matrix_mul_square.cu -O3 -lineinfo -DTILE_WIDTH=16 -o tiled_matrix_mul_square
```

## 3. Eseguire

Run di default:

```bash
./tiled_matrix_mul_square
```

Parametri:

```text
./tiled_matrix_mul_square [Width] [iterations]
```

- `Width`: lato delle matrici quadrate, default `512`.
- `iterations`: ripetizioni del kernel misurate con CUDA events, default `20`.

Esempi:

```bash
./tiled_matrix_mul_square 512 20
./tiled_matrix_mul_square 1024 10
```

`Width` deve essere divisibile per `TILE_WIDTH`. Con il default `TILE_WIDTH=32`, valori validi sono per esempio `256`, `512`, `1024`, `2048`.

## 4. Output atteso

```text
device: NVIDIA ...
Width: 512
TILE_WIDTH: 32
grid: (16, 16, 1)
block: (32, 32, 1)
shared memory per block: 8192 bytes
iterations: 20
average kernel time: ... ms
effective throughput: ... GFLOP/s
max absolute error: ...
PASS
```

Il programma verifica il risultato con una moltiplicazione CPU e stampa il throughput effettivo del kernel:

```text
2 x Width x Width x Width FLOP / tempo_kernel
```

## 5. Lettura del kernel

Ogni fase `ph` fa tre cose:

- carica una tile di `M` e una tile di `N` in shared memory;
- sincronizza il blocco con `__syncthreads()`;
- accumula `TILE_WIDTH` prodotti dentro la variabile privata `Pvalue`.

La seconda `__syncthreads()` impedisce che alcuni thread sovrascrivano `Mds` o `Nds` con la tile successiva mentre altri thread stanno ancora leggendo la tile corrente.

## 6. Profiling

Con Nsight Compute:

```bash
ncu --set full --force-overwrite -o tiled-matmul-ncu ./tiled_matrix_mul_square 1024 5
```

Aprire il report:

```bash
ncu-ui tiled-matmul-ncu.ncu-rep
```

Sezioni utili:

- `Launch Statistics`;
- `Occupancy`;
- `Memory Workload Analysis`;
- `Source Counters`;
- `GPU Speed Of Light Throughput`.

## 7. Pulizia

```bash
rm -f tiled_matrix_mul_square tiled-matmul-ncu.ncu-rep
```
