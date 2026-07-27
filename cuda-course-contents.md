Sillabo corso
GPUs and CUDA for Parallel Computing: introduzione pratica
Obiettivi del corso
Il corso introduce i principi del calcolo parallelo su GPU e l’uso di CUDA per sviluppare semplici applicazioni ad alte prestazioni.
Al termine del percorso i partecipanti saranno in grado di comprendere il ruolo delle GPU nel calcolo eterogeneo, scrivere kernel CUDA di base, gestire la memoria tra host e device, analizzare i principali limiti prestazionali e riconoscere le tecniche fondamentali di ottimizzazione.
________________________________________
1. Introduzione al calcolo parallelo eterogeneo
Differenza tra calcolo seriale, parallelo ed eterogeneo.
Ruolo della CPU e della GPU in un’applicazione moderna.
Perché servono più velocità e più parallelismo.
Esempi di applicazioni reali accelerate tramite GPU.
Vantaggi e limiti dell’approccio parallelo.
Principali difficoltà nella programmazione parallela.
________________________________________
2. Modello di programmazione CUDA
Introduzione a CUDA come piattaforma per il calcolo parallelo su GPU.
Concetto di funzione eseguita su CPU e kernel eseguito su GPU.
Struttura di un semplice programma CUDA.
Gestione di variabili, thread, blocchi e griglie.
Uso di base della CUDA Runtime API.
Allocazione, trasferimento e rilascio della memoria.
Primo esempio pratico: esecuzione parallela di un semplice calcolo vettoriale.
________________________________________
3. Architettura di calcolo della GPU
Struttura generale di una GPU moderna.
Streaming Multiprocessor, core CUDA e organizzazione delle risorse.
Scheduling dei blocchi di thread.
Sincronizzazione tra thread dello stesso blocco.
Scalabilità trasparente dei kernel CUDA.
Warps e modello SIMD.
Divergenza del flusso di controllo.
Scheduling dei warp e tolleranza alla latenza.
Occupancy e partizionamento delle risorse.
________________________________________
4. Architettura della memoria e località dei dati
La banda di memoria come limite prestazionale.
Gerarchia della memoria in CUDA.
Memoria globale, condivisa, locale, costante e texture memory.
Differenza tra memoria host e memoria device.
Accesso efficiente ai dati.
Località dei dati e riduzione del traffico di memoria.
Tecnica del tiling.
Esempio pratico: moltiplicazione di matrici con tiling.
Controlli sui bordi nei kernel.
Impatto dell’uso della memoria sull’occupancy.
________________________________________
5. Misurazione delle prestazioni
Metriche principali per valutare le prestazioni.
Misurazione dei tempi lato CPU.
Uso dei timer CUDA lato GPU.
Differenza tra tempo di esecuzione complessivo e tempo effettivo del kernel.
Analisi dei trasferimenti tra host e device.
Individuazione dei colli di bottiglia.
________________________________________
6. Ottimizzazione della memoria
Ottimizzazione dei trasferimenti tra host e device.
Uso della memoria pinned.
Trasferimenti asincroni.
Sovrapposizione tra computazione e trasferimento dati.
Uso corretto degli stream CUDA.
Scelta degli spazi di memoria più adatti.
Riduzione degli accessi alla memoria globale.
Utilizzo della shared memory per migliorare le prestazioni.
________________________________________
7. Ottimizzazione della configurazione CUDA
Scelta della dimensione dei blocchi.
Relazione tra thread, blocchi, warp e occupancy.
Bilanciamento tra registri, shared memory e numero di thread attivi.
Effetti della configurazione del kernel sulle prestazioni.
Strategie pratiche per testare configurazioni diverse.
________________________________________
8. CUDA e PyTorch
Introduzione al supporto GPU in PyTorch.
Uso del package torch.cuda.
Verifica della disponibilità della GPU.
Spostamento di tensori e modelli su GPU.
Differenza tra calcolo su CPU e calcolo su GPU.
Gestione base della memoria GPU in PyTorch.
Esempi pratici di utilizzo CUDA con tensori PyTorch.
