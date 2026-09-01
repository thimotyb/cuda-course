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

## Docker for the M9 vLLM exercise

The CUDA examples in the earlier modules do not require Docker. The M9 vLLM exercise uses Docker to run vLLM and its Python dependencies inside the official `vllm/vllm-openai` image, so vLLM does not need to be installed in the host Python environment.

On Ubuntu, install Docker Engine from Docker's official repository. Do not rely on an unrelated or outdated distribution package when preparing the course environment:

```bash
sudo apt update
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Allow the current user to run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
```

Log out and log in again, or start a new login session, before testing the group change. Then verify Docker:

```bash
docker run --rm hello-world
```

For the vLLM container to use the NVIDIA GPU, install and configure the NVIDIA Container Toolkit after Docker:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify that Docker can access the NVIDIA GPU:

```bash
docker run --rm --gpus all nvidia/cuda:base-ubuntu22.04 nvidia-smi
```

The test requires a working NVIDIA driver and a GPU visible to the host. If `nvidia-smi` works on the host but the container test fails, check the NVIDIA Container Toolkit configuration before troubleshooting vLLM.

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

## Requisiti per gli esercizi Python/PyTorch

Gli esempi del modulo `ch08` usano Python e PyTorch con supporto CUDA. Per evitare di installare pacchetti Python nel sistema operativo, usa una virtualenv locale del repository:

```bash
cd /home/thimoty/git/cuda-course
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy --index-url https://download.pytorch.org/whl/cu130
```

Verifica che PyTorch veda CUDA:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Il wheel `cu130` include le librerie runtime CUDA 13.0 richieste da PyTorch. Serve comunque un driver NVIDIA sufficientemente recente e una GPU visibile al sistema.

## Esempi disponibili

- `ch02/simple_kernel_launch`: lancio minimo di un kernel CUDA.
- `ch02/vector_add`: somma vettoriale con allocazione device, copie host-device, kernel GPU e verifica del risultato.
- `ch03/device_properties`: interrogazione delle proprieta' CUDA del device con `cudaGetDeviceProperties()`.
- `ch03/reduction_occupancy`: riduzione parallela parametrica per profilare thread block, warps, shared memory, sincronizzazione e occupancy.
- `ch04/register_occupancy_tradeoff`: kernel parametrico per confrontare thread per blocco, pressione sui registri, occupancy teorica e tempo di esecuzione.
- `ch04/tiled_matrix_mul_square`: moltiplicazione di matrici quadrate con tiling e shared memory, senza boundary checks.
- `ch04/cublas_matrix_mul_compare`: confronto tra triplo `for` CPU, tiled kernel didattico e `cuBLAS SGEMM`.
- `ch06/array_add_on_device`: confronto tra due mappature 2D di una somma di array per osservare accessi coalesced e strided.
- `ch06/bandwidth_bottleneck`: confronto mirato tra copy coalesced e strided per misurare il bandwidth utile e il traffico globale effettivo.
- `ch06/l2_cache`: misura del riuso in L2 con accessi normali, streaming e persisting.
- `ch06/async_overlap`: pipeline copy-kernel-copy con pinned memory e stream CUDA per osservare overlap asincrono.
- `ch08/pytorch_cuda_tensors`: introduzione ai tensori PyTorch eseguiti su CUDA, con attributi, placement, operazioni tensoriali, timing CUDA e memoria GPU.
- `ch08/torch_cuda_overview`: panoramica pratica del package `torch.cuda`, con device discovery, proprieta' GPU, memoria, stream, CUDA events e sincronizzazione.
- `ch08/pytorch_cuda_memory`: esercizio focalizzato sugli strumenti PyTorch per osservare memoria CUDA allocata, riservata, picchi, cache e memoria libera del device.
- `ch09/vllm_docker_demo`: prototipo per avviare Qwen3-0.6B in vLLM tramite Docker e inviare una richiesta da Python attraverso l'API OpenAI-compatible.
