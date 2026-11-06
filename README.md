# swot2argo

Code and data for *Reconstructing 3D upper-ocean density from SWOT's wide-swath sea surface height using machine
learning* (Champenois, Ali and Bodner).

| folder | content |
|---|---|
| [code/](code/) | the Python package; four scripts (model selection, training, evaluation, figures) that produce every figure and table of the paper and its Supporting Information; the trained network, its predictions and the tables; notebooks that draw the figures |
| [data/](data/) | the inputs: Argo training pairs, the pre-2023 Argo record, the test sets, the gridded products at the test samples, the hourly mooring records, the SWOT overpasses and crossover statistics of the figures |

Quick start, from this folder:

```
cd code
conda env create -f environment.yml && conda activate swot2argo   # or: pip install -r requirements.txt
pip install -e .
python scripts/03_evaluate.py && python scripts/04_figures.py       # every table and figure from the shipped results (about 3 min)
```

The full pipeline (model selection and training, which need a GPU) and the notebooks are described in
[code/README.md](code/README.md).

The code is under the MIT license ([LICENSE](LICENSE)); the data are under CC BY 4.0, with the credits of the products
they come from listed in [data/README.md](data/README.md). To cite, see [CITATION.cff](CITATION.cff).
