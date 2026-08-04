# CUDA Course

Repository principale del corso pratico su GPU e CUDA.

## Struttura

- `site/`: sito statico del corso, con una pagina per modulo.
- `site/chapters/`: contenuti HTML dei moduli.
- `site/assets/`: CSS, JavaScript e immagini usate dal sito.
- `resources/`: fonti e materiali di riferimento locali.
- `examples/`: esempi CUDA usati a lezione e in laboratorio.
- `tests/non-regression/`: lock e controlli per proteggere i moduli gia' preparati.

## Esempi CUDA

Gli esempi usati durante il corso vanno copiati e mantenuti in questo repository, sotto `examples/`, anche quando derivano dai repository remoti di supporto.

Convenzione consigliata:

```text
examples/chXX/nome_esempio/
```

Ogni esempio dovrebbe contenere:

- uno o piu' file sorgente, per esempio `.cu`, `.cpp`, `.h`;
- un `README.md` con i comandi di compilazione ed esecuzione;
- eventuali dati piccoli necessari per riprodurre l'esercizio.

Non vanno versionati gli eseguibili generati da `nvcc`, i file oggetto o altri output di build locali.

Esempio gia' presente:

```bash
cd examples/ch02/simple_kernel_launch
nvcc simple_kernel_launch.cu -o simple_kernel_launch
./simple_kernel_launch
```

Output atteso:

```text
Hello World!
```

## Controlli

Dopo modifiche ai contenuti del sito:

```bash
python3 scripts/non_regression_guard.py check
```

Quando una modifica intenzionale aggiorna un modulo gia' bloccato, rigenerare il lock del modulo interessato prima del controllo finale.
