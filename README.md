---
title: vae-mnist-playground
colorFrom: purple
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🧠 VAE · MNIST Playground</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=7C3AED&center=true&vCenter=true&width=700&lines=Train+a+Variational+Autoencoder+in+your+browser;Explore+the+2-D+latent+manifold+visually;Generate+new+digits+by+sampling+latent+space;Deep+learning+made+interactive+%26+accessible" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/VAE-Playground)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/VAE-Playground)

<br/>

**🧠 VAE · MNIST Playground** — An interactive web application for training, visualising, and experimenting with Variational Autoencoders on the MNIST handwritten digit dataset, directly in your browser with zero setup.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>⚡ <b>Live Training Dashboard</b></td>
    <td>Configure hyperparameters (epochs, batch size, learning rate, hidden & latent dims) and launch training with a real-time animated progress bar and loss curve</td>
  </tr>
  <tr>
    <td>🌐 <b>Latent Space Visualisation</b></td>
    <td>Scatter-plot the 2-D encoded representations of 10 000 MNIST samples, colour-coded by digit class, revealing the learned manifold structure</td>
  </tr>
  <tr>
    <td>🔁 <b>Reconstruction Comparison</b></td>
    <td>Side-by-side view of original MNIST digits and their VAE reconstructions, updating on every click with a freshly sampled random batch</td>
  </tr>
  <tr>
    <td>✨ <b>Interactive Generation</b></td>
    <td>Two latent-space sliders let you navigate the learned manifold in real time and decode novel digit-like images on the fly; a full 15×15 grid view is also available</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerized Deployment</b></td>
    <td>Docker-first architecture, cloud-ready and scalable</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VAE-Playground                         │
│                                                                 │
│  ┌────────────┐    ┌─────────────────┐    ┌───────────────┐   │
│  │   MNIST    │───▶│  VAE (PyTorch)  │───▶│   Flask API   │   │
│  │  Dataset   │    │  Encoder/Decoder│    │   Backend     │   │
│  └────────────┘    └─────────────────┘    └───────┬───────┘   │
│                                                    │           │
│                                           ┌────────▼────────┐  │
│                                           │  Vanilla JS     │  │
│                                           │  + Matplotlib   │  │
│                                           │   Dashboard     │  │
│                                           └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**VAE Data Flow:**
```
Input (784-D)
    │
    ▼  Encoder FC (ReLU)
    │
    ├──▶ μ head ──┐
    └──▶ σ² head ─┤  z = μ + σ·ε   (reparameterization trick)
                  │
                  ▼
           Latent z (2-D / n-D)
                  │
                  ▼  Decoder FC (ReLU → Sigmoid)
                  │
           Output (784-D)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/VAE-Playground.git
cd VAE-Playground

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash

docker build -t vae-playground .

docker run -p 7860:7860 vae-playground
```

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| ⚡ Training Dashboard | Configure hyperparameters, launch training, watch real-time loss curve and progress bar | ✅ Live |
| 🏗️ Architecture Viewer | Interactive diagram of the full VAE topology with the ELBO loss formula explained | ✅ Live |
| 🌐 Latent Space Explorer | 2-D scatter plot of encoded MNIST digits, colour-coded by class | ✅ Live |
| 🔁 Reconstruction Viewer | Random-batch side-by-side comparison of originals vs VAE reconstructions | ✅ Live |
| ✨ Generation Console | Latent-vector sliders for real-time digit generation + 15×15 manifold grid | ✅ Live |
| 📦 Model Export | Download trained weights as a `.pt` checkpoint | 🗓️ Planned |

---

## 🧠 ML Models

```python
# Core Models Used in VAE-Playground
models = {
    "architecture":       "Variational Autoencoder (VAE)",
    "encoder":            "FC 784 → hidden_dim → (μ, log σ²)",
    "decoder":            "FC latent_dim → hidden_dim → 784",
    "loss_function":      "ELBO = BCE Reconstruction + KL Divergence",
    "reparameterization": "z = μ + σ · ε,  ε ~ N(0, I)"
}
```

**Configurable hyperparameters at runtime:**

| Parameter | Default | Range |
|-----------|---------|-------|
| Epochs | 30 | 1 – 200 |
| Batch size | 128 | 32 / 64 / 128 / 256 |
| Learning rate | 1e-3 | 1e-4 / 1e-3 / 1e-2 |
| Hidden dimension | 400 | 200 / 400 / 512 |
| Latent dimension | 2 | 2 / 5 / 10 / 20 |

---

## 📁 Project Structure

```
VAE-Playground/
│
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 app.py                     # Flask app, VAE model, all routes & HTML template
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available (MNIST is a public domain dataset) — no real user data is stored. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c3aed,100:06b6d4&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![Star on GitHub](https://img.shields.io/badge/%E2%AD%90_Star-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/VAE-Playground/stargazers)
[![Fork on GitHub](https://img.shields.io/badge/%F0%9F%8D%B4_Fork-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/VAE-Playground/fork)

<sub>The name "VAE · MNIST Playground" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
