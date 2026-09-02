# Reduction Occupancy

Questo esempio aggiunge a M3 un kernel piu' interessante di `vector_add` per osservare SM, thread block, warps, shared memory, sincronizzazione e occupancy con Nsight Compute.

Il programma somma un array grande di `float`. Ogni blocco CUDA riduce una parte dell'array in shared memory e scrive una somma parziale. La somma finale delle somme parziali viene calcolata lato CPU, cosi' il profilo resta concentrato su un singolo kernel CUDA leggibile.

## 1. Entrare nella cartella dell'esempio

```bash
cd examples/ch03/reduction_occupancy
```

## 2. Compilare con nvcc

Per il profiling usare `-O3 -lineinfo`, cosi' Nsight Compute puo' collegare meglio metriche e sorgente:

```bash
nvcc reduction.cu -O3 -lineinfo -o reduction
```

## 3. Lanciare il programma

Run di default:

```bash
./reduction
```

Run con parametri espliciti:

```bash
./reduction 67108864 256 20
```

Argomenti:

```text
./reduction [N] [block_size] [iterations]
```

- `N`: numero di elementi dell'array, default `67108864`.
- `block_size`: thread per blocco, default `256`.
- `iterations`: ripetizioni del kernel misurate con CUDA events, default `20`.

Block size supportati:

```text
64, 128, 256, 512, 1024
```

Output atteso:

```text
device: NVIDIA ...
N: 67108864
block size: 256
blocks: 262144
shared memory per block: 1024 bytes
iterations: 20
CPU sum: 67108864.000000
GPU sum: 67108864.000000
absolute error: 0.000000
average kernel time: ... ms
PASS
```

## 4. Esperimenti con block size

Eseguire lo stesso problema con configurazioni diverse:

```bash
./reduction 67108864 64 20
./reduction 67108864 128 20
./reduction 67108864 256 20
./reduction 67108864 512 20
./reduction 67108864 1024 20
```

Da questi run (solo il binario, senza profiler) si osservano le metriche che il
programma stampa da solo, come mostrato in "Output atteso" al punto 3:

- numero di blocchi lanciati;
- shared memory per blocco;
- tempo kernel medio.

Le altre metriche di occupancy -- theoretical occupancy, achieved occupancy,
active warps, eligible warps, stall reasons, memory throughput -- **non
compaiono in questo output**: il programma non le calcola. Sono metriche del
profiler, visibili solo passando per Nsight Compute (sezione 6), ripetuto per
ogni block size:

```bash
ncu --set full --force-overwrite -o reduction-ncu-bs64   ./reduction 67108864 64   20
ncu --set full --force-overwrite -o reduction-ncu-bs128  ./reduction 67108864 128  20
ncu --set full --force-overwrite -o reduction-ncu-bs256  ./reduction 67108864 256  20
ncu --set full --force-overwrite -o reduction-ncu-bs512  ./reduction 67108864 512  20
ncu --set full --force-overwrite -o reduction-ncu-bs1024 ./reduction 67108864 1024 20
```

Aprire ciascun report con `ncu-ui reduction-ncu-bsNNN.ncu-rep` (o leggere
l'output testuale che `ncu` stampa comunque a schermo, anche senza GUI) e
confrontare le sezioni elencate al punto 7 tra le varie configurazioni.

## 5. Profilare con Nsight Systems

Nsight Systems mostra la timeline del programma: copie host-device, lanci kernel, sincronizzazioni e durata complessiva.

Creare un file di profilo Nsight Systems:

```bash
nsys profile --trace=cuda --stats=true -o reduction-nsys ./reduction
```

Il comando genera un report visualizzabile, normalmente:

```text
reduction-nsys.nsys-rep
```

Puo' generare anche un file SQLite di supporto:

```text
reduction-nsys.sqlite
```

Aprire il profilo nella GUI di Nsight Systems:

```bash
nsys-ui reduction-nsys.nsys-rep
```

Nella GUI osservare la timeline CUDA: chiamate runtime, copie host-device, warmup, lanci ripetuti del kernel e sincronizzazioni.

## 6. Profilare con Nsight Compute

Nsight Compute e' il tool principale per studiare il singolo kernel.

Creare un file di profilo Nsight Compute:

```bash
ncu --set full --force-overwrite -o reduction-ncu ./reduction
```

Il comando genera:

```text
reduction-ncu.ncu-rep
```

Aprire il profilo nella GUI di Nsight Compute:

```bash
ncu-ui reduction-ncu.ncu-rep
```

Questo e' il report da usare per studiare occupancy, warps, scheduler, memoria, sorgente e metriche per singolo kernel.

Se il report e' troppo lento, ridurre `N` o `iterations` durante il profiling:

```bash
ncu --set full --force-overwrite -o reduction-ncu ./reduction 16777216 256 5
```

## 7. Leggere il report Nsight Compute da console (senza GUI)

Non serve `ncu-ui`: `ncu` stampa lo stesso identico contenuto -- Launch
Statistics, Occupancy, Warp State Statistics, Scheduler Statistics, Memory
Workload Analysis, GPU Speed Of Light Throughput, Source Counters -- anche
come testo in terminale. Utile su una macchina remota via SSH senza GUI.

### Stampare direttamente durante il profiling

Basta omettere `-o`:

```bash
ncu --set full ./reduction 16777216 256 5
```

Attenzione pero': senza restrizioni, `ncu` profila **ogni** lancio del
kernel che incontra, quindi qui produrrebbe un report completo per il
warmup piu' uno per ciascuna delle `iterations` chiamate nel ciclo timed --
tanto testo ripetuto da scorrere. Per un singolo report pulito, saltare il
warmup e limitarsi a un solo lancio con `--launch-skip` / `--launch-count`
(e opzionalmente `--kernel-name` per essere espliciti su quale kernel):

```bash
ncu --set full --kernel-name reduceBlocksKernel \
    --launch-skip 1 --launch-count 1 \
    ./reduction 16777216 256 20
```

(`--launch-skip 1` salta il lancio di warmup, `--launch-count 1` profila
esattamente un lancio del ciclo timed successivo.)

### Rileggere un report gia' salvato

Se il report e' gia' stato generato con `-o` (sezione 6), lo si puo'
ristampare in console in qualsiasi momento, senza rilanciare il programma:

```bash
ncu --import reduction-ncu.ncu-rep
```

### Altri flag utili per l'output testuale

- `--page details` (default): report completo per sezione, stesso contenuto
  della pagina Details della GUI.
- `--page raw`: valori grezzi delle metriche, una riga per metrica -- comodo
  per fare `grep` su una metrica specifica.
- `--csv` (insieme a `--page raw` o `--page details`): output CSV invece che
  testo formattato, utile per script o confronti tra piu' run.
- `--print-summary per-kernel`: una riga di riepilogo per kernel invece del
  report completo -- comodo per confrontare rapidamente le 5 configurazioni
  di block size (sezione 4) senza scorrere pagine di testo per ciascuna.

## 8. Sezioni Nsight Compute da leggere

Partire da:

- `Launch Statistics`;
- `Occupancy`;
- `Warp State Statistics`;
- `Scheduler Statistics`;
- `Memory Workload Analysis`;
- `GPU Speed Of Light Throughput`;
- `Source Counters`, se disponibili.

Domande guida:

- Quanti blocchi vengono lanciati?
- Il numero di blocchi e' sufficiente per dare lavoro a tutti gli SM?
- Quale risorsa limita l'occupancy teorica?
- L'achieved occupancy segue la theoretical occupancy?
- Il kernel e' piu' limitato da memoria, scheduling, sincronizzazione o istruzioni?
- Cambiare block size migliora davvero il tempo medio?

## 9. Pulizia

Per rimuovere file generati localmente:

```bash
rm -f reduction reduction-nsys.nsys-rep reduction-nsys.sqlite reduction-ncu.ncu-rep reduction-ncu-bs*.ncu-rep
```
