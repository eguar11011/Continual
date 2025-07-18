## 🌱 Continual Neural Network Training

This repository enables training a neural network under **continual learning** scenarios. It also provides tools to **compute the similarity of internal representations** between two trained models.

### 🚀 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 🧠 Training

Train a neural network with continual learning:

```bash
python trainer.py --config path_to_config
```

Supported learning strategies:

- `finetune`
- `replay`
- `ewc`

### 📐 Representation Similarity

Compute the similarity of internal representations between two trained models:

```bash
python subspace_similarity.py --config path_to_config
```
## 🌱 Continual Neural Network Training

This repository enables training a neural network under **continual learning** scenarios. It also provides tools to **compute the similarity of internal representations** between two trained models.

### 🚀 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 🧠 Training

Train a neural network with continual learning:

```bash
python trainer.py --config path_to_config
```

Supported learning strategies:

- `finetune`
- `replay`
- `ewc`

### 📐 Representation Similarity

Compute the similarity of internal representations between two trained models:

```bash
python subspace_similarity.py --config path_to_config
```

---

### 🖥️ Recommended Compute

All models in this repository were trained on **RunPod** using RTX 4000 Ada GPUs.  
<https://console.runpod.io/>.
