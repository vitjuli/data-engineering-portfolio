# Data Engineering & Data Structures – Research Portfolio

This repository presents a set of research-driven projects in **Data Engineering** and **Computer Science** by **Iuliia Vitiugova**.
The focus is on designing efficient data pipelines, experimenting with data structures, and applying advanced transformations in a reproducible, professional manner.

The work includes both **Jupyter Notebooks** for transparent exploration and **Python modules** for reusable code.

---

## Repository Structure

```
data-engineering-portfolio/
├── notebooks/        # Research notebooks (cleaned: outputs cleared, uniform headers)
├── src/              # Reusable Python code auto-extracted from notebooks
├── requirements.txt  # Minimal environment (auto-detected from imports)
├── README.md
└── LICENSE
```

### Notebooks
- **TP1_Data_Engineering.ipynb** — Prototype data pipeline: loading, cleaning, validation.
- **TP2_Data_Engineering.ipynb** — Transformations at scale and performance profiling.
- **TP2_Data_Structures_Dynamic_Tables.ipynb** — Dynamic tables from scratch; amortized analysis.
- **TP3_Data_Engineering.ipynb** — Storage strategies, efficient joins, memory profiling.
- **Project_Data_Engineering.ipynb** — End-to-end pipeline from ingestion to reporting.

> All notebooks are standardized with cover pages, reproducibility notes, and cleared outputs.

### Source Code
Auto-extracted top-level functions/classes consolidated in **`src/common.py`** for reuse.

---

## Installation

```bash
pip install -r requirements.txt
```

## ▶ Usage

Run any notebook from the `notebooks/` folder via Jupyter Lab/Notebook.
You can also import utilities:

```python
from src.common import *
```

---

## Research Scope & Highlights
- Modular **data pipelines**: ingestion → validation → preprocessing → transformation → analysis.
- Efficient **data structures** and memory-aware operations.
- **Reproducibility**: clean outputs, fixed kernel metadata, minimal dependencies.
- Clear **documentation**: cover pages, section templates, and conclusions.

---

---

## 📜 License
MIT License. See `LICENSE` for details.
