# Explanation Guide: Reduction Profiling

Questa guida serve per commentare l'esempio `examples/ch03/reduction_occupancy` durante M3.

L'obiettivo non e' spiegare ogni metrica disponibile, ma dare una traccia chiara su cosa osservare nei due tool:

- Nsight Systems: timeline del programma CUDA completo.
- Nsight Compute: analisi del singolo kernel dentro gli SM.

## 1. Comandi di riferimento

Compilazione:

```bash
cd examples/ch03/reduction_occupancy
nvcc reduction.cu -O3 -lineinfo -o reduction
```

Run didattico:

```bash
./reduction 67108864 256 20
```

Profilo Nsight Systems:

```bash
nsys profile --trace=cuda --stats=true -o reduction-nsys ./reduction 67108864 256 20
nsys-ui reduction-nsys.nsys-rep
```

Profilo Nsight Compute:

```bash
ncu --set full --force-overwrite -o reduction-ncu ./reduction 67108864 256 20
ncu-ui reduction-ncu.ncu-rep
```

Se Nsight Compute e' troppo lento, usare meno iterazioni:

```bash
ncu --set full --force-overwrite -o reduction-ncu ./reduction 67108864 256 5
```

## 2. Cosa dire prima di aprire i profiler

Il programma fa una riduzione parallela di un grande array di `float`.

Ogni blocco CUDA:

- legge una porzione dell'array globale;
- copia i valori in shared memory;
- riduce i valori dentro il blocco;
- usa `__syncthreads()` tra i passi della reduction;
- scrive una somma parziale in memoria globale.

La somma finale delle partial sums viene fatta lato CPU. Questo mantiene il profilo centrato su un kernel CUDA leggibile.

Il parametro `iterations` non serve all'algoritmo. Serve alla misura:

```text
1 warmup + iterations lanci misurati
```

Con:

```bash
./reduction 67108864 256 20
```

Nsight Systems mostra:

```text
1 warmup + 20 iterazioni = 21 lanci kernel
```

La media stampata dal programma e' calcolata solo sui 20 lanci misurati, non sul warmup.

## 3. Nsight Systems: cosa guardare

Nsight Systems risponde alla domanda:

```text
Quando lavora la GPU dentro il programma completo?
```

Non e' il tool principale per occupancy, warp scheduler o register pressure.

### 3.1 Project Explorer

Aprire il report:

```text
reduction-nsys.nsys-rep
```

Commento:

```text
Questo report contiene la timeline completa del processo: lavoro CPU, chiamate CUDA API, copie memoria e kernel sulla GPU.
```

### 3.2 CPU rows

Guardare le righe CPU nere.

Cosa spiegare:

- la CPU inizializza l'array;
- calcola la somma di riferimento;
- chiama il runtime CUDA;
- alloca memoria con `cudaMalloc`;
- lancia kernel;
- aspetta sincronizzazioni;
- copia indietro le partial sums;
- verifica il risultato.

Commento utile:

```text
Tanto nero sulla CPU non significa che la GPU non venga usata. Significa che Nsight Systems mostra tutto il programma, inclusi setup, riferimento CPU e verifica.
```

### 3.3 CUDA API row

Aprire la riga:

```text
CUDA API
```

Cercare:

- `cudaMalloc`;
- `cudaMemcpy`;
- `cudaLaunchKernel`;
- `cudaEventRecord`;
- `cudaEventSynchronize`;
- `cudaFree`.

Commento:

```text
Queste sono chiamate lato host al runtime CUDA. Alcune preparano la GPU, altre lanciano lavoro, altre aspettano che il lavoro finisca.
```

### 3.4 CUDA HW rows

Aprire:

```text
CUDA HW
  Kernels
  Memory
```

Cercare:

- `reduceBlocksKernel`;
- copie host-to-device;
- copie device-to-host.

Commento:

```text
Qui vediamo il lavoro effettivo sul dispositivo: kernel e trasferimenti di memoria. Questa e' la parte che distingue l'orchestrazione CPU dall'esecuzione GPU.
```

### 3.5 I 21 lanci del kernel

Espandere:

```text
Kernels
  reduceBlocksKernel
```

Se sembra una sola riga, zoomare sulla timeline.

Commento:

```text
Vediamo 21 lanci perche' il programma fa un warmup non misurato e poi 20 iterazioni misurate. I rettangoli sono ravvicinati perche' ogni kernel dura pochi millisecondi.
```

### 3.6 Memory transfers

Guardare i trasferimenti:

- input host-to-device all'inizio;
- partial sums device-to-host alla fine.

Commento:

```text
Il kernel lavora su dati gia' residenti in GPU. Le copie fanno parte del programma completo, ma il tempo medio stampato dal programma misura solo i kernel ripetuti tra gli eventi CUDA.
```

### 3.7 Conclusione da Nsight Systems

Messaggio da portare a casa:

```text
Nsight Systems mostra che il programma ha una vera sequenza CUDA: setup, copia input, warmup, lanci kernel ripetuti, copia risultati. Ora il kernel e' abbastanza visibile da meritare un'analisi piu' dettagliata con Nsight Compute.
```

## 4. Nsight Compute: cosa guardare

Nsight Compute risponde alla domanda:

```text
Quanto bene lavora il kernel dentro gli SM?
```

Qui si guardano occupancy, warps, scheduler, memoria e sorgente.

## 4.1 Launch Statistics

Cercare:

- `Grid Size`;
- `Block Size`;
- `Threads`;
- `Registers Per Thread`;
- `Static Shared Memory Per Block`;
- `Dynamic Shared Memory Per Block`;
- `# SMs`;
- `Waves Per SM`.

Spiegazione importante:

```text
Grid Size e' il numero di blocchi, non il numero totale di thread.
```

Nel run:

```bash
./reduction 67108864 256 20
```

si ha:

```text
Grid Size = 262144 blocchi
Block Size = 256 thread per blocco
Total Threads = Grid Size x Block Size = 67108864 thread
```

Commento:

```text
A differenza di vector_add, qui non lanciamo 2 blocchi e 8 thread totali. Lanciamo centinaia di migliaia di blocchi, quindi gli SM hanno lavoro sufficiente.
```

## 4.2 Occupancy

Cercare:

- `Theoretical Occupancy`;
- `Achieved Occupancy`;
- `Theoretical Active Warps per SM`;
- `Achieved Active Warps Per SM`;
- `Block Limit Registers`;
- `Block Limit Shared Mem`;
- `Block Limit Warps`;
- `Block Limit Blocks`;
- `Block Limit SM`.

Commento:

```text
La theoretical occupancy dice quante warps potrebbero essere residenti in base a block size, registri, shared memory e limiti dello SM. La achieved occupancy dice quante warps sono state effettivamente attive durante l'esecuzione.
```

Per block size `256`:

```text
256 thread per blocco = 8 warps per blocco
```

Se uno SM supporta 64 warps residenti:

```text
64 / 8 = 8 blocchi per SM, prima di considerare altri limiti
```

Messaggio didattico:

```text
Occupancy non e' una percentuale magica da massimizzare sempre. Serve a capire se ci sono abbastanza warps residenti per nascondere latenze.
```

## 4.3 Warp State Statistics

Cercare:

- active warps;
- eligible warps;
- issued warps;
- stall reasons.

Commento:

```text
Una warp active e' residente. Una warp eligible e' pronta a emettere istruzioni. Se ci sono molte active warps ma poche eligible warps, molte warps sono ferme in attesa.
```

Messaggio:

```text
Occupancy alta aiuta solo se produce warps pronte. Warps residenti ma non pronte non tengono occupati gli scheduler.
```

## 4.4 Scheduler Statistics

Cercare:

- issue rate;
- cicli senza eligible warps;
- istruzioni emesse per scheduler.

Commento:

```text
Questa sezione mostra se gli scheduler riescono a trovare lavoro pronto. Se spesso non ci sono warps eleggibili, il kernel sta aspettando memoria, sincronizzazione o dipendenze.
```

## 4.5 Memory Workload Analysis

Cercare:

- DRAM throughput;
- L1/TEX throughput;
- L2 throughput;
- global loads;
- global stores;
- shared memory activity.

Commento:

```text
La reduction legge molti valori da global memory, poi riduce dentro il blocco in shared memory. Il traffico globale principale e' la lettura dell'input grande e la scrittura delle partial sums.
```

Messaggio:

```text
Questo kernel e' utile per vedere il passaggio tra memoria globale, shared memory e sincronizzazione intra-blocco.
```

## 4.6 Source View

Aprire la vista sorgente, se disponibile grazie a:

```bash
nvcc reduction.cu -O3 -lineinfo -o reduction
```

Guardare:

```cpp
shared[tid] = (global_index < n) ? input[global_index] : 0.0f;
__syncthreads();

for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    shared[tid] += shared[tid + stride];
  }
  __syncthreads();
}
```

Commento:

```text
La prima riga legge da global memory. Poi i thread cooperano attraverso shared memory. Ogni passo della reduction usa una barriera. L'if rende attivi sempre meno thread man mano che lo stride si dimezza.
```

Punto didattico:

```text
La reduction e' corretta e semplice, ma non e' perfetta: nelle ultime fasi molti thread del blocco sono inattivi. Questo e' un buon ponte verso ottimizzazioni successive.
```

## 5. Confronto tra block size

Eseguire:

```bash
./reduction 67108864 64 20
./reduction 67108864 128 20
./reduction 67108864 256 20
./reduction 67108864 512 20
./reduction 67108864 1024 20
```

Per profiling piu' leggero con Nsight Compute:

```bash
ncu --set full --force-overwrite -o reduction-64 ./reduction 67108864 64 5
ncu --set full --force-overwrite -o reduction-256 ./reduction 67108864 256 5
ncu --set full --force-overwrite -o reduction-1024 ./reduction 67108864 1024 5
```

Confrontare:

- tempo medio stampato dal programma;
- grid size;
- block size;
- theoretical occupancy;
- achieved occupancy;
- active warps;
- eligible warps;
- memory throughput;
- stall reasons.

Commento finale:

```text
Il block size migliore non e' necessariamente quello con occupancy piu' alta. La performance dipende dal compromesso tra parallelismo residente, scheduling, memoria, sincronizzazione e lavoro utile per thread.
```

## 6. Frase conclusiva per la lezione

```text
Nsight Systems ci dice che il programma usa davvero la GPU e ci mostra dove stanno setup, copie e kernel. Nsight Compute entra dentro il kernel e ci mostra come block size, warps, shared memory, registri e scheduler determinano occupancy e throughput.
```
