# Course Examples

Questa cartella contiene gli esempi CUDA usati a lezione.

Gli esempi possono essere copiati o adattati dai repository remoti di supporto, ma la versione usata nel corso deve stare qui per mantenere sito, appunti e codice nello stesso repository.

## Convenzioni

- Organizzare gli esempi per modulo: `ch02`, `ch03`, ecc.
- Usare una sottocartella per ogni esempio.
- Inserire sempre un `README.md` con compilazione, lancio e output atteso.
- Versionare sorgenti e piccoli file di input necessari.
- Non versionare binari compilati, file oggetto o output temporanei.

## Esempi disponibili

- `ch02/simple_kernel_launch`: lancio minimo di un kernel CUDA.
- `ch02/vector_add`: somma vettoriale con allocazione device, copie host-device, kernel GPU e verifica del risultato.
