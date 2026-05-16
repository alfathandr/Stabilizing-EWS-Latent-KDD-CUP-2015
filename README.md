# Stabilizing Dynamic Early Warning Systems via Latent Feature Representation (KDD Cup 2015 Pipeline)

Official repository for the research paper: **"Stabilizing Dynamic Early Warning Systems via Latent Feature Representation in AE-LSTM and AE-Transformer"**.

This folder contains the complete, standalone implementation pipeline optimized for the **KDD Cup 2015 MOOC Dataset**. The framework targets early student dropout prediction by transforming high-dimensional user interaction event logs into dense temporal representations, specifically focusing on enforcing neural network training stabilization across brief observation windows.

---

## 🛠️ Pipeline Architecture & Cell Breakdown

The notebook is structurally modularized into 11 self-contained logical cells to protect the computational runtime from connection timeout data resets and reference missing states:

### 1. Environment & Determinism (`CELL 1`)
Loads primary computing packages for neural infrastructure operations. Configures global random generator backends to seed locks (`seed=42`) across PyTorch and NumPy, neutralizing initialization biases and guaranteeing stable replication properties.

### 2. Ingestion & Storage Integration (`CELL 2`)
Mounts cloud storage paths to construct operational data pathways. Ingests raw platform records files, including transaction tracking sheets (`log_train.csv`), enrollment profiles (`enrollment_train.csv`), and output target metrics (`truth_train.csv`).

### 3. Exploratory Data Analysis (`CELL 3`)
Profiles missing value volumes and statistical descriptive distribution metrics. Synchronizes data columns into clean retention parameters (`0` for Stay/Safe, `1` for Dropout/At-Risk) and maps target balance properties via standalone embedded visualization modules.

### 4. Leakage-Free Feature Engineering (`CELL 4`)
Partitions student records into Train, Validation, and Test branches (60:20:20 ratio) based strictly on unique enrollment IDs. Computes dynamic relative week variables tracking individual user interaction paths from day zero. Vectorizes event transactions into an unclipped raw 3D sequence tensor matrix layout:
$$\mathbf{X} \in \mathbb{R}^{\text{Students} \times \text{Weeks} \times \text{Event Attributes}}$$
The operational matrix structure is tightly boxed into a 4-week longitudinal sequence length comprising 7 unique feature attributes.

### 5. Deep Learning Model Architectures (`CELL 5`)
Declares baseline deep sequential frameworks (`Vanilla-LSTM`, `Vanilla-Transformer`) alongside the proposed hibrida frameworks (`AE-LSTM`, `AE-Transformer`). Integrates an information bottleneck compression layer restricted to a 32-unit volume layer to filter out zero-value transactional sparsity.

### 6. Multi-Horizon Joint Optimization Pipeline (`CELL 6`)
Tracks system adaptive prediction powers at sequential checkpoints (Weeks 1, 2, 3, and 4). Embeds local normalizer data processing and Scikit-Learn evaluation functions directly within the loop to maintain standalone execution integrity. Hybrid parameters are optimized using a strict Multi-Objective Joint Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + 0.05 \cdot \mathcal{L}_{\text{reconstruction}}$$

### 7. Quantitative Recapitulation & Performance Trends (`CELL 7`)
Structures multi-horizon model score tracking entries into styled evaluation tables. Generates multi-panel line charts mapping performance trend lines over week steps, clustered bar graphics at week 4, and cumulative area-under-the-curve ROC diagnostic plots.

### 8. Learning Optimization Dynamics & Confusion Matrix (`CELL 8`)
Deploys local metrics and rendering dependencies to diagram neural optimization behaviors. Plots validation accuracy alongside training loss lines across 50 epochs to capture optimization paths, highlighting vanilla loss spikes vs the smooth convergence path of the proposed model. Renders discrete 2x2 confusion matrix grids.

### 9. Post-Hoc Model Explainability via SHAP (`CELL 9`)
Dismantles machine learning "black-box" limitations by compiling `GradientExplainer` post-hoc accountability algorithms. Integrates execution hooks to temporarily bypass cuDNN state limitations during backward graph routing passes, generating dots summary graphics illustrating the feature attribution impacts of the 7 MOOC event types.

### 10. Operational Hardware Profiling (`CELL 10`)
Measures standalone hardware footprint allocations within physical CUDA/GPU environments. Profiles specific processing runtime per forward-backward pass iteration (`Step Time`), max graphic adapter memory storage reserves (`Peak VRAM`), and systemic loading parameters (`System RAM`).

### 11. Empirical Latent Space Seed Ablation & Significance Testing (`CELL 11`)
Validates structural choice integrity via double empirical testing. Executes a bottleneck capacity ablation suite ($z \in \{16, 32, 64\}$) to justify the choice of a 32-unit latent constraint layer. Conducts independent two-sample Student's t-tests over training histories to provide absolute mathematical proof of significant stabilization gains ($p < 0.05$).

---

## 🚀 Execution & Requirements

### Technical Dependencies
- Python >= 3.8
- PyTorch >= 2.0
- Scikit-Learn >= 1.0
- SHAP >= 0.40
- SciPy, Psutil, Pandas, NumPy, Matplotlib, Seaborn

### Running the Notebook
1. Put your raw KDD Cup 2015 files (`log_train.csv`, `enrollment_train.csv`, `truth_train.csv`) inside your Google Drive storage path.
2. Open the file `stabilizing_ews_latent_kdd2015.ipynb` inside Google Colab.
3. Activate a hardware accelerated GPU environment (`Runtime` -> `Change runtime type` -> `T4 GPU`).
4. Execute the cells sequentially from `CELL 1` to `CELL 11`. All generated evaluation metric outputs, performance graphics, and model weights checkpoints (`.pth`) will save automatically to your Drive results directory.
