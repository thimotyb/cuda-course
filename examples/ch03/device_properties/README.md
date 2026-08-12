# Device Properties

Questo esempio rende eseguibile il programma citato in M3, sezione `2.4 Querying device properties`.

Il programma interroga il runtime CUDA con `cudaGetDeviceCount()` e `cudaGetDeviceProperties()`, poi stampa le proprieta' principali della GPU: nome, compute capability, numero di SM, warp size, limiti di thread, shared memory, registri e memoria globale.

## 1. Entrare nella cartella dell'esempio

```bash
cd examples/ch03/device_properties
```

## 2. Compilare con nvcc

```bash
nvcc device_properties.cu -o device_properties
```

## 3. Lanciare il programma

```bash
./device_properties
```

Output atteso, con valori diversi in base alla GPU installata:

```text
CUDA devices found: 1

Device 0
  name: NVIDIA ...
  compute capability: ...
  SM count: ...
  warp size: 32
  max threads per block: ...
  max threads per SM: ...
  shared memory per block: ... bytes
  shared memory per SM: ... bytes
  registers per block: ...
  registers per SM: ...
  global memory: ... GiB
```

Se il sistema non vede una GPU CUDA, il programma stampa:

```text
No CUDA-capable device found.
```

## 4. Cosa osservare

- `multiProcessorCount` indica quanti SM sono disponibili sul device.
- `warpSize` e' quasi sempre 32 sulle GPU CUDA attuali, ma e' comunque una proprieta' da leggere dal device.
- `maxThreadsPerMultiProcessor`, registri e shared memory aiutano a ragionare sull'occupancy.
- `sharedMemPerBlock` e `sharedMemPerMultiprocessor` distinguono il limite per singolo blocco dal budget complessivo dello SM.

## 5. Pulizia

Per rimuovere l'eseguibile generato:

```bash
rm device_properties
```
