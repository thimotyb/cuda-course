# Course Examples

Questa cartella contiene gli esempi CUDA usati a lezione.

Gli esempi possono essere copiati o adattati dai repository remoti di supporto, ma la versione usata nel corso deve stare qui per mantenere sito, appunti e codice nello stesso repository.

## Prerequisiti su Ubuntu

Per compilare ed eseguire gli esempi CUDA serve:

- una GPU NVIDIA compatibile con CUDA;
- un driver NVIDIA funzionante;
- NVIDIA CUDA Toolkit, che include `nvcc`;
- tool di compilazione di base come `gcc`, `g++`, `make`;
- `git`, se vuoi clonare o aggiornare il repository del corso.

Verifica che il sistema veda una GPU NVIDIA:

```bash
lspci | grep -i nvidia
```

Verifica che il driver NVIDIA sia attivo:

```bash
nvidia-smi
```

Installa gli strumenti di sviluppo di base:

```bash
sudo apt update
sudo apt install build-essential git wget
```

Installa il CUDA Toolkit dal repository NVIDIA. Scegli prima il repository corretto per la tua versione di Ubuntu dalla pagina ufficiale:

https://developer.nvidia.com/cuda-downloads

Esempio per Ubuntu 24.04 su `x86_64`:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

Se usi Ubuntu 22.04, sostituisci `ubuntu2404` con `ubuntu2204` nell'URL del repository, oppure usa i comandi generati dalla pagina ufficiale NVIDIA.

Dopo l'installazione, verifica che `nvcc` sia disponibile:

```bash
nvcc --version
```

Se `nvcc` non viene trovato ma il toolkit e' installato, aggiungi temporaneamente il percorso standard alla shell:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Per renderlo permanente, aggiungi la stessa riga al file `~/.bashrc` e riapri il terminale.

## Tool di profiling CUDA aggiornati

Per profilare gli esempi su GPU recenti conviene usare versioni aggiornate di NVIDIA Nsight Systems e NVIDIA Nsight Compute. Su Ubuntu, dopo aver configurato il repository NVIDIA CUDA come mostrato sopra, installa i pacchetti Nsight piu' recenti disponibili per la tua distribuzione.

Esempio per CUDA/Nsight 13.3:

```bash
sudo apt update
sudo apt install cuda-nsight-systems-13-3 cuda-nsight-compute-13-3
```

Se vuoi installare anche il toolkit CUDA 13.3 completo:

```bash
sudo apt install cuda-toolkit-13-3
```

Verifica le versioni disponibili:

```bash
apt-cache policy cuda-nsight-systems-13-3 cuda-nsight-compute-13-3 cuda-toolkit-13-3
```

Verifica i tool installati:

```bash
/opt/nvidia/nsight-systems/2026.1.3/bin/nsys --version
/opt/nvidia/nsight-compute/2026.2.1/ncu --version
```

Installa anche le dipendenze GUI necessarie per aprire i report con `nsys-ui`:

```bash
sudo apt install libwayland-server0
```

Se `nsys-ui` mostra un errore come:

```text
Failed to load plugin: QuadDPlugin
libwayland-server.so.0: cannot open shared object file: No such file or directory
```

significa che manca proprio `libwayland-server0`. Senza questa libreria la finestra principale di Nsight Systems puo' aprirsi ma restare vuota o senza funzioni selezionabili.

Se la shell continua a usare una vecchia versione sotto `/usr/local/cuda-12.x/bin`, controlla tutti i path visibili e metti i binari aggiornati prima nel `PATH`:

```bash
type -a nsys ncu
export PATH=/opt/nvidia/nsight-systems/2026.1.3/bin:/opt/nvidia/nsight-compute/2026.2.1:$PATH
nsys --version
ncu --version
```

Per rendere permanente questo ordine, aggiungi la riga `export PATH=...` al file `~/.bashrc`.

Comandi minimi per profilare un esempio:

```bash
cd examples/ch02/vector_add
nvcc vector_add.cu -o vector_add
nsys profile --trace=cuda --stats=true -o vector-add-nsys ./vector_add
ncu --set full -o vector-add-ncu ./vector_add
```

Se `ncu` termina con `ERR_NVGPUCTRPERM`, il tool e' installato correttamente ma l'utente corrente non ha accesso ai performance counters NVIDIA. Su una macchina Linux nativa questo si risolve abilitando l'accesso ai contatori nel driver NVIDIA o usando una sessione con privilegi adeguati. Su WSL questa limitazione puo' dipendere anche dalla versione del driver Windows e dalle policy esposte al guest Linux.

## Convenzioni

- Organizzare gli esempi per modulo: `ch02`, `ch03`, ecc.
- Usare una sottocartella per ogni esempio.
- Inserire sempre un `README.md` con compilazione, lancio e output atteso.
- Versionare sorgenti e piccoli file di input necessari.
- Non versionare binari compilati, file oggetto o output temporanei.

## Esempi disponibili

- `ch02/simple_kernel_launch`: lancio minimo di un kernel CUDA.
- `ch02/vector_add`: somma vettoriale con allocazione device, copie host-device, kernel GPU e verifica del risultato.
- `ch03/device_properties`: interrogazione delle proprieta' CUDA del device con `cudaGetDeviceProperties()`.
- `ch03/reduction_occupancy`: riduzione parallela parametrica per profilare thread block, warps, shared memory, sincronizzazione e occupancy.
- `ch04/register_occupancy_tradeoff`: kernel parametrico per confrontare thread per blocco, pressione sui registri, occupancy teorica e tempo di esecuzione.
- `ch04/tiled_matrix_mul_square`: moltiplicazione di matrici quadrate con tiling e shared memory, senza boundary checks.
