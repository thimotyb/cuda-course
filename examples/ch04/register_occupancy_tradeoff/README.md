# Register Occupancy Trade-off

Questo esempio accompagna M4 e serve a misurare il rapporto tra registri per thread, occupancy teorica e tempo kernel.

Il programma compila piu' varianti dello stesso kernel. Ogni variante usa un diverso livello di pressione sui registri. A runtime si scelgono:

- numero di elementi;
- thread per blocco;
- livello di pressione sui registri;
- lavoro aritmetico per thread;
- numero di iterazioni misurate.

Il numero di registri richiesto in input non e' una promessa esatta: e' un livello di pressione. Il compilatore CUDA decide il numero reale di registri dopo ottimizzazioni, inlining, unrolling e possibili spill. Per questo il programma stampa anche `actual regs`, letto con `cudaFuncGetAttributes()`.

## 1. Entrare nella cartella dell'esempio

```bash
cd examples/ch04/register_occupancy_tradeoff
```

## 2. Compilare

Per misure leggibili e profiling usare `-O3 -lineinfo`:

```bash
nvcc register_occupancy_tradeoff.cu -O3 -lineinfo -o register_occupancy_tradeoff
```

## 3. Lanciare una configurazione singola

```bash
./register_occupancy_tradeoff
```

Parametri:

```text
./register_occupancy_tradeoff [N] [block_size] [register_level] [arithmetic_rounds] [iterations] [sweep]
```

- `N`: numero di elementi, default `16777216`.
- `block_size`: thread per blocco, default `256`.
- `register_level`: livello di pressione sui registri, default `32`.
- `arithmetic_rounds`: lavoro aritmetico per thread, default `128`.
- `iterations`: ripetizioni misurate con CUDA events, default `20`.
- `sweep`: `0` per una sola configurazione, `1` per una tabella comparativa.

Block size supportati:

```text
64, 128, 256, 512, 1024
```

Livelli register pressure supportati:

```text
8, 16, 32, 64, 96, 128
```

Esempi:

```bash
./register_occupancy_tradeoff 16777216 256 32 128 20 0
./register_occupancy_tradeoff 16777216 256 96 128 20 0
```

## 4. Confronto automatico

Per confrontare piu' configurazioni:

```bash
./register_occupancy_tradeoff 16777216 256 32 128 20 1
```

La modalita' sweep ignora `block_size` e `register_level` come singola configurazione e prova:

```text
block size:      64, 128, 256, 512, 1024
register level:  8, 16, 32, 64, 96, 128
```

Alcune combinazioni possono essere saltate. Per esempio, se `actual regs x block_size` supera il limite di registri per blocco del device, il kernel non puo' essere lanciato con quella configurazione.

Output tipico:

```text
device: NVIDIA ...
SM count: ...
warp size: 32
max threads per SM: ...
registers per block limit: ...
registers per SM: ...
N: 16777216
arithmetic rounds: 128
iterations: 20

     block      level  actual regs      blocks/SM     warps/SM  max warps      occ %       avg ms      Gelem/s
       256         32           ...             ...          ...        ...        ...        ...        ...
```

Colonne principali:

- `block`: thread per blocco.
- `level`: livello di pressione sui registri richiesto.
- `actual regs`: registri reali per thread scelti dal compilatore.
- `blocks/SM`: blocchi residenti per SM stimati dalla CUDA Occupancy API.
- `warps/SM`: warp residenti teorici.
- `occ %`: occupancy teorica.
- `avg ms`: tempo medio del kernel.
- `Gelem/s`: elementi processati al secondo.

## 5. Come leggere il risultato

Se `actual regs` aumenta, ogni thread richiede piu' registri. A parita' di block size, il numero di blocchi residenti per SM puo' diminuire, quindi diminuiscono anche warp residenti e occupancy teorica.

Questo non implica automaticamente che il kernel diventi piu' lento. Piu' registri possono evitare traffico verso local memory e rendere ogni thread piu' efficiente. Pero' se l'occupancy scende troppo, lo SM ha meno warp pronti per nascondere latenze di memoria e dipendenze.

Il confronto corretto e':

```text
register pressure -> actual regs -> blocks/SM -> occupancy -> avg ms
```

Non guardare solo `occ %`: la configurazione migliore e' quella che produce il tempo minore per quel kernel e quel problema.

## 6. Profiling con Nsight Compute

Per vedere occupancy, registri, eventuali spill e stall reasons:

```bash
ncu --set full --force-overwrite -o register-occupancy-ncu ./register_occupancy_tradeoff 16777216 256 96 128 5 0
```

Aprire il report:

```bash
ncu-ui register-occupancy-ncu.ncu-rep
```

Sezioni utili:

- `Launch Statistics`;
- `Occupancy`;
- `Source Counters`;
- `Warp State Statistics`;
- `Scheduler Statistics`.

Metriche da cercare:

- registri per thread;
- theoretical occupancy;
- achieved occupancy;
- local memory traffic, se il compilatore ha fatto spill;
- stall reasons.

Per una dimostrazione guidata completa con due report CLI e una traccia di commento, usare anche:

```bash
less DEMO.md
```

## 7. Pulizia

```bash
rm -f register_occupancy_tradeoff register-occupancy-ncu.ncu-rep
```
