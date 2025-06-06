<img src="https://github.com/user-attachments/assets/40310b7a-a864-4ed2-a2dd-c488231aee6b" width="300"/>

---
### 📦 Dependencies

| Package       | Version  | Badges |
|---------------|----------|--------|
| **Python**    | 3.11+    | ![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg) |
| **torch**     | 2.6.0    | [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) |
| **pynvml**    | 12.0.0   | [![PyPI - pynvml](https://img.shields.io/pypi/v/pynvml.svg)](https://pypi.org/project/pynvml/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/pynvml.svg)](https://pypi.org/project/pynvml/) |
| **pandas**    | 2.3.0    | [![PyPI - pandas](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.org/project/pandas/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/pandas.svg)](https://pypi.org/project/pandas/) |
| **matplotlib**| 3.10.3    | [![PyPI - matplotlib](https://img.shields.io/pypi/v/matplotlib.svg)](https://pypi.org/project/matplotlib/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/matplotlib.svg)](https://pypi.org/project/matplotlib/) |
| **tqdm**      | 4.68.1   | [![PyPI - tqdm](https://img.shields.io/pypi/v/tqdm.svg)](https://pypi.org/project/tqdm/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/tqdm.svg)](https://pypi.org/project/tqdm/) |

---
### 🧠 What is PruneEnergyAnalyzer?

**PruneEnergyAnalyzer** is a Python-based toolkit designed to analyze the energy consumption of pruned deep learning models. It allows you to:
- Evaluate the energy efficiency of different pruning strategies.
- Visualize how pruning affects energy, FLOPs, parameters, and FPS.
- Support Green AI initiatives by making energy-aware design decisions based on real consumption data.

---

### ⚙️ Installation

```bash
git clone https://github.com/DEEP-CGPS/PruneEnergyAnalyzer.git
cd PruneEnergyAnalyzer
pip install -r requirements.txt
```

---

### 🚀 Quick Example

```python
import sys
import os

# Add the library path (assuming the library is one folder above this script)
lib_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if lib_path not in sys.path:
    sys.path.append(lib_path)

import torch
import pandas as pd
from PruneEnergyAnalizer import ExperimentRunner

# Define the model directory (adjust as needed)
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "models"))

# Define experiment parameters
batch_sizes = [1, 8, 16, 32, 64]  # Adjust according to GPU memory
input_channels = 3
input_height = 224
input_width = 224

# Initialize and run the experiment
experiment = ExperimentRunner(
    model_dir = model_dir,             # Path to the directory containing all pruned models
    batch_sizes = batch_sizes,         # List of batch sizes to be tested (e.g., [1, 8, 16, 32, 64])
    input_channels = input_channels,   # Number of channels in the input images (e.g., 3 for RGB)
    input_height = input_height,       # Height of the input images (e.g., 224 for 224x224 images)
    input_width = input_width,         # Width of the input images (e.g., 224 for 224x224 images)
    filename = "results.csv",          # Output filename for the experiment results (CSV)
)

results_df = experiment.run_experiment()

# Display the results DataFrame
results_df
```

---

### 📚 How to cite

If you use this software for research or application purposes, please use the following citation:

```bibtex
@article{PACHON2025pruneenergyanalyzer,
  title     = {PruneEnergyAnalyzer: A Toolkit for Measuring Energy Efficiency in Pruned Deep Learning Models},
  journal   = {},
  volume    = {},
  pages     = {},
  year      = {2025},
  issn      = {},
  doi       = {},
  url       = {https://github.com/DEEP-CGPS/PruneEnergyAnalyzer},
  author    = {Cesar G. Pachon and Cesar Pedraza and Dora Ballesteros},
  keywords  = {Energy consumption, Deep learning, CNN pruning, Green AI, Python toolkit}
}
```



