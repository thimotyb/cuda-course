# Simple Kernel Launch

Questo esempio serve a verificare il ciclo minimo di lavoro con CUDA: scrivere un kernel, compilarlo con `nvcc`, lanciarlo e osservare l'output del programma.

Questa e' la copia mantenuta nel repository principale del corso. Gli esempi usati a lezione vivono sotto `examples/` per tenere insieme materiali, sito e codice compilabile.

## 1. Entrare nella cartella dell'esempio

```bash
cd examples/ch02/simple_kernel_launch
```

## 2. Compilare con nvcc

```bash
nvcc simple_kernel_launch.cu -o simple_kernel_launch
```

Punti da spiegare:

- `nvcc` e' il compilatore CUDA.
- Il file sorgente usa estensione `.cu`.
- `-o simple_kernel_launch` sceglie il nome dell'eseguibile.
- Durante la compilazione, `nvcc` gestisce sia il codice host CPU sia il codice device GPU.

## 3. Lanciare il programma

```bash
./simple_kernel_launch
```

Output atteso:

```text
Hello World!
```

## 4. Cosa succede nel codice

```cpp
__global__ void mykernel(void) {}
```

`__global__` dichiara una funzione CUDA kernel. Un kernel viene chiamato dal codice host ed eseguito sul device. In questo esempio il corpo e' vuoto, quindi il thread GPU non produce output e non modifica dati.

```cpp
mykernel<<<1, 1>>>();
```

La sintassi `<<<1, 1>>>` configura il lancio del kernel:

- il primo valore indica il numero di blocchi nella griglia;
- il secondo valore indica il numero di thread per blocco.

Qui viene lanciata una griglia composta da un solo blocco con un solo thread.

```cpp
printf("Hello World!\n");
```

Questa stampa avviene sul lato host, cioe' sulla CPU. L'esempio non dimostra ancora calcolo parallelo utile: dimostra solo che il programma CUDA compila, che il runtime accetta un kernel launch e che l'eseguibile parte correttamente.

## 5. Pulizia

Per rimuovere l'eseguibile generato:

```bash
rm simple_kernel_launch
```
