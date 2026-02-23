from flask import Flask, render_template_string, jsonify, request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import os

app = Flask(__name__)

# ── Global training state ──────────────────────────────────────────────────────
training_state = {
    'is_training': False, 'progress': 0, 'current_epoch': 0,
    'total_epochs': 0, 'losses': [], 'trained': False, 'current_loss': 0
}

# ── VAE Architecture ───────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc_mu    = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar= nn.Linear(hidden_dim, latent_dim)
        self.fc3      = nn.Linear(latent_dim, hidden_dim)
        self.fc4      = nn.Linear(hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# ── Data loading ───────────────────────────────────────────────────────────────
def load_mnist_data():
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
    idx = torch.randperm(len(ds))[:10000]
    imgs, lbls = [], []
    for i in idx:
        img, lbl = ds[i]
        imgs.append(img.view(-1).numpy())
        lbls.append(lbl)
    return np.array(imgs), np.array(lbls)

print("Loading MNIST …")
vae = None
data, labels = load_mnist_data()
data_tensor = torch.FloatTensor(data)
print(f"Loaded {len(data)} samples")

# ── Training thread ────────────────────────────────────────────────────────────
def train_vae_thread(epochs, batch_size, lr, hidden_dim, latent_dim):
    global vae, training_state
    training_state.update(is_training=True, progress=0, current_epoch=0,
                           total_epochs=epochs, losses=[])
    vae = VAE(784, hidden_dim, latent_dim)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    dl  = DataLoader(torch.utils.data.TensorDataset(data_tensor),
                     batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        vae.train(); total = 0
        for (x,) in dl:
            opt.zero_grad()
            rx, mu, lv = vae(x)
            loss, _, _ = vae_loss(rx, x, mu, lv)
            loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(dl.dataset)
        training_state['losses'].append(avg)
        training_state.update(current_epoch=ep+1, current_loss=avg,
                               progress=int((ep+1)/epochs*100))
    training_state.update(is_training=False, trained=True)
    print("Training complete!")

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return s

# ── HTML Template ──────────────────────────────────────────────────────────────
HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VAE · MNIST Playground</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  /* ── Reset & Base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0f0f1a;
    --surface: #16162a;
    --card:    #1e1e35;
    --border:  #2e2e52;
    --accent:  #7c3aed;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --danger:  #ef4444;
    --success: #22c55e;
    --warn:    #f59e0b;
    --radius:  14px;
    --shadow:  0 8px 32px rgba(0,0,0,.4);
  }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Animated background ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: -1;
    background:
      radial-gradient(ellipse 80% 60% at 20% 20%, rgba(124,58,237,.18) 0%, transparent 60%),
      radial-gradient(ellipse 60% 80% at 80% 80%, rgba(6,182,212,.12) 0%, transparent 60%),
      var(--bg);
    animation: bgPulse 10s ease-in-out infinite alternate;
  }
  @keyframes bgPulse {
    from { opacity: 1; }
    to   { opacity: .85; }
  }

  /* ── Layout ── */
  .app-shell {
    display: grid;
    grid-template-rows: auto 1fr auto;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    padding: 24px 32px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
  }
  .logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px;
    display: grid; place-items: center;
    font-size: 24px;
    flex-shrink: 0;
    box-shadow: 0 4px 20px rgba(124,58,237,.4);
  }
  .header-text h1 {
    font-size: clamp(1.3rem, 3vw, 2rem);
    font-weight: 700;
    background: linear-gradient(90deg, #c4b5fd, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .header-text p { color: var(--muted); font-size: .9rem; margin-top: 2px; }
  .header-badges { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
  .badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .75rem; font-weight: 600;
    border: 1px solid;
  }
  .badge-purple { border-color: #7c3aed44; background: #7c3aed22; color: #c4b5fd; }
  .badge-cyan   { border-color: #06b6d444; background: #06b6d422; color: #67e8f9; }
  .badge-amber  { border-color: #f59e0b44; background: #f59e0b22; color: #fcd34d; }

  /* ── Main ── */
  main { padding: 28px 32px; max-width: 1440px; margin: 0 auto; width: 100%; }

  /* ── Tabs ── */
  .tabs {
    display: flex; gap: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px;
    overflow-x: auto;
    margin-bottom: 28px;
    scrollbar-width: none;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tab {
    flex-shrink: 0;
    display: flex; align-items: center; gap: 7px;
    padding: 9px 18px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--muted);
    font-family: inherit; font-size: .88rem; font-weight: 500;
    cursor: pointer;
    transition: all .2s;
    white-space: nowrap;
  }
  .tab:hover { color: var(--text); background: var(--card); }
  .tab.active {
    background: linear-gradient(135deg, var(--accent), #5b21b6);
    color: #fff;
    box-shadow: 0 4px 16px rgba(124,58,237,.35);
  }
  .tab-icon { font-size: 1rem; }

  /* ── Tab content ── */
  .tab-content { display: none; animation: fadeIn .25s ease; }
  .tab-content.active { display: block; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
  }
  .card-title {
    font-size: 1rem; font-weight: 600; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }
  .card-title .icon { font-size: 1.15rem; }

  /* ── Grid helpers ── */
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 16px; }
  .mt { margin-top: 20px; }

  /* ── Stat chips (training dashboard) ── */
  .stat-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    display: flex; align-items: center; gap: 14px;
  }
  .stat-chip .sc-icon {
    width: 44px; height: 44px;
    border-radius: 10px;
    display: grid; place-items: center;
    font-size: 1.3rem;
    flex-shrink: 0;
  }
  .sc-purple { background: rgba(124,58,237,.2); }
  .sc-cyan   { background: rgba(6,182,212,.2);  }
  .sc-amber  { background: rgba(245,158,11,.2); }
  .sc-green  { background: rgba(34,197,94,.2);  }
  .stat-chip .sc-label { font-size: .78rem; color: var(--muted); }
  .stat-chip .sc-val   { font-size: 1.2rem; font-weight: 700; margin-top: 2px; }

  /* ── Form controls ── */
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px,1fr)); gap: 14px; margin-bottom: 20px; }
  .field label {
    display: block; font-size: .8rem; font-weight: 500; color: var(--muted);
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px;
  }
  .field select, .field input[type=number] {
    width: 100%; padding: 10px 14px;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: .88rem;
    transition: border-color .2s;
    -webkit-appearance: none; appearance: none;
  }
  .field select:focus, .field input:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(124,58,237,.2);
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 10px 22px;
    border-radius: 8px; border: none;
    font-family: inherit; font-size: .9rem; font-weight: 600;
    cursor: pointer; transition: all .2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #5b21b6);
    color: #fff; box-shadow: 0 4px 14px rgba(124,58,237,.4);
  }
  .btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,.55);
  }
  .btn-ghost {
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover:not(:disabled) { background: var(--card); border-color: var(--accent); }
  .btn:disabled { opacity: .45; cursor: not-allowed; transform: none !important; }
  .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }

  /* ── Progress bar ── */
  .prog-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; height: 36px; overflow: hidden; position: relative;
    margin: 16px 0;
  }
  .prog-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 8px;
    transition: width .4s ease;
    display: flex; align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 700; color: #fff; min-width: 2rem;
    box-shadow: 0 0 20px rgba(124,58,237,.4);
    position: relative; overflow: hidden;
  }
  .prog-bar::after {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,.15) 50%, transparent 100%);
    animation: shimmer 1.5s infinite;
  }
  @keyframes shimmer {
    from { transform: translateX(-100%); }
    to   { transform: translateX(200%); }
  }

  /* ── Status badge ── */
  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px;
    font-size: .8rem; font-weight: 600;
  }
  .status-pill::before {
    content: ''; width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
  }
  .s-idle    { background: rgba(148,163,184,.15); color: var(--muted);   border: 1px solid rgba(148,163,184,.25); }
  .s-idle::before    { background: var(--muted); }
  .s-training{ background: rgba(245,158,11,.15);  color: #fcd34d; border: 1px solid rgba(245,158,11,.3); }
  .s-training::before{ background: var(--warn); animation: pulse 1s infinite; }
  .s-ready   { background: rgba(34,197,94,.15);   color: #86efac; border: 1px solid rgba(34,197,94,.3); }
  .s-ready::before   { background: var(--success); }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.3;} }

  /* ── Sliders ── */
  .slider-wrap { margin: 12px 0; }
  .slider-label {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
    font-size: .85rem; color: var(--muted); font-weight: 500;
  }
  .slider-val {
    background: rgba(124,58,237,.25); color: #c4b5fd;
    border: 1px solid rgba(124,58,237,.4);
    padding: 2px 10px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: .8rem;
  }
  input[type=range] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 6px;
    background: var(--surface); border-radius: 3px;
    border: 1px solid var(--border); outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 18px; height: 18px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    cursor: pointer; box-shadow: 0 2px 8px rgba(124,58,237,.5);
    border: 2px solid rgba(255,255,255,.2);
    transition: transform .15s;
  }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }

  /* ── Image output ── */
  .img-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; min-height: 200px;
    display: flex; align-items: center; justify-content: center;
    margin-top: 14px; overflow: hidden;
  }
  .img-box img { width: 100%; height: auto; border-radius: 10px; display: block; }
  .img-box.small img { max-height: 300px; width: auto; max-width: 100%; margin: 0 auto; }
  .img-box .placeholder {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--muted); font-size: .88rem; padding: 30px; text-align: center;
  }
  .img-box .placeholder .ph-icon { font-size: 2.2rem; opacity: .5; }

  /* ── Architecture diagram ── */
  .arch-flow { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .arch-block {
    width: 100%; max-width: 460px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 20px; text-align: center;
  }
  .arch-block .ab-title { font-weight: 600; font-size: .95rem; }
  .arch-block .ab-sub   { color: var(--muted); font-size: .8rem; margin-top: 3px; }
  .arch-block.enc { border-left: 3px solid var(--accent);  }
  .arch-block.lat { border-left: 3px solid var(--accent2); background: rgba(6,182,212,.06); }
  .arch-block.dec { border-left: 3px solid var(--accent3); }
  .arch-arrow { color: var(--muted); font-size: 1.1rem; line-height: 1; }
  .arch-label {
    background: rgba(6,182,212,.1); border: 1px solid rgba(6,182,212,.25);
    color: #67e8f9; border-radius: 6px;
    padding: 3px 12px; font-size: .78rem; font-weight: 600;
  }
  .formula-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 22px; margin-top: 14px;
  }
  .formula-box p { font-size: .88rem; color: var(--muted); line-height: 1.7; }
  .formula-box code {
    font-family: 'JetBrains Mono', monospace; font-size: .85rem;
    color: #c4b5fd; background: rgba(124,58,237,.15);
    padding: 2px 6px; border-radius: 4px;
  }

  /* ── Info alert ── */
  .alert {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 18px;
    font-size: .88rem; line-height: 1.5;
  }
  .alert-warn { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); color: #fcd34d; }
  .alert-info { background: rgba(6,182,212,.1);  border: 1px solid rgba(6,182,212,.3);  color: #67e8f9; }
  .alert .a-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

  /* ── Footer ── */
  footer {
    padding: 20px 32px;
    border-top: 1px solid var(--border);
    box-shadow: 0 -1px 0 0 rgba(124,58,237,.15);
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; flex-wrap: wrap;
  }
  .footer-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .footer-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    border: 2px solid var(--border);
    object-fit: cover; flex-shrink: 0;
  }
  .footer-name { font-size: .88rem; font-weight: 600; color: var(--text); }
  .footer-sub  { font-size: .76rem; color: var(--muted); margin-top: 1px; }
  .footer-links { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .footer-link {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 12px; border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--muted); font-size: .78rem; font-weight: 500;
    text-decoration: none;
    transition: all .2s;
  }
  .footer-link:hover {
    border-color: var(--accent);
    color: #c4b5fd;
    background: rgba(124,58,237,.12);
  }

  /* ── Spinner ── */
  .spinner {
    width: 28px; height: 28px; border-radius: 50%;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="app-shell">

  <!-- Header -->
  <header>
    <div class="logo">🧠</div>
    <div class="header-text">
      <h1>VAE · MNIST Playground</h1>
      <p>Interactive Variational Autoencoder — train, explore &amp; generate</p>
    </div>
    <div class="header-badges">
      <span class="badge badge-purple">PyTorch</span>
      <span class="badge badge-cyan">MNIST</span>
      <span class="badge badge-amber">Generative AI</span>
    </div>
  </header>

  <!-- Main -->
  <main>
    <!-- Tabs -->
    <nav class="tabs" id="tabs">
      <button class="tab active" onclick="switchTab('training', this)">
        <span class="tab-icon">⚡</span> Training
      </button>
      <button class="tab" onclick="switchTab('architecture', this)">
        <span class="tab-icon">🏗️</span> Architecture
      </button>
      <button class="tab" onclick="switchTab('latent', this)">
        <span class="tab-icon">🌐</span> Latent Space
      </button>
      <button class="tab" onclick="switchTab('reconstruction', this)">
        <span class="tab-icon">🔁</span> Reconstruction
      </button>
      <button class="tab" onclick="switchTab('generation', this)">
        <span class="tab-icon">✨</span> Generation
      </button>
    </nav>

    <!-- ── TAB: Training ── -->
    <div id="training" class="tab-content active">

      <!-- Stat chips -->
      <div class="grid-3" style="margin-bottom:20px;">
        <div class="stat-chip">
          <div class="sc-icon sc-purple">🎯</div>
          <div>
            <div class="sc-label">Status</div>
            <div class="sc-val" id="status-chip">
              <span class="status-pill s-idle" id="status-pill">Not Trained</span>
            </div>
          </div>
        </div>
        <div class="stat-chip">
          <div class="sc-icon sc-cyan">📈</div>
          <div>
            <div class="sc-label">Epoch</div>
            <div class="sc-val" id="epoch-val">0 / 0</div>
          </div>
        </div>
        <div class="stat-chip">
          <div class="sc-icon sc-amber">⚡</div>
          <div>
            <div class="sc-label">Current Loss</div>
            <div class="sc-val" id="loss-val">—</div>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Config card -->
        <div class="card">
          <div class="card-title"><span class="icon">⚙️</span> Hyperparameters</div>
          <div class="form-grid">
            <div class="field">
              <label>Epochs</label>
              <input type="number" id="epochs" value="30" min="1" max="200"/>
            </div>
            <div class="field">
              <label>Batch Size</label>
              <select id="batch_size">
                <option>32</option>
                <option>64</option>
                <option selected>128</option>
                <option>256</option>
              </select>
            </div>
            <div class="field">
              <label>Learning Rate</label>
              <select id="learning_rate">
                <option value="0.0001">1e-4</option>
                <option value="0.001" selected>1e-3</option>
                <option value="0.01">1e-2</option>
              </select>
            </div>
            <div class="field">
              <label>Hidden Dim</label>
              <select id="hidden_dim">
                <option>200</option>
                <option selected>400</option>
                <option>512</option>
              </select>
            </div>
            <div class="field">
              <label>Latent Dim</label>
              <select id="latent_dim">
                <option value="2" selected>2 (viz)</option>
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="20">20</option>
              </select>
            </div>
          </div>
          <div class="btn-row">
            <button class="btn btn-primary" id="train-btn" onclick="startTraining()">
              🚀 Start Training
            </button>
            <button class="btn btn-ghost" onclick="resetModel()">
              🔄 Reset
            </button>
          </div>
        </div>

        <!-- Loss curve card -->
        <div class="card">
          <div class="card-title"><span class="icon">📉</span> Training Loss</div>
          <div class="prog-wrap" id="prog-wrap" style="display:none;">
            <div class="prog-bar" id="prog-bar" style="width:0%">0%</div>
          </div>
          <div class="img-box" id="loss-box" style="min-height:220px;">
            <div class="placeholder">
              <span class="ph-icon">📊</span>
              Loss curve appears here after training starts
            </div>
          </div>
          <div style="margin-top:10px;">
            <button class="btn btn-ghost" style="font-size:.8rem;padding:7px 14px;" onclick="updateLossCurve()">
              🔄 Refresh
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- ── TAB: Architecture ── -->
    <div id="architecture" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span class="icon">🏗️</span> Network Topology</div>
          <div class="arch-flow">
            <div class="arch-block">
              <div class="ab-title">Input · 784-D</div>
              <div class="ab-sub">28 × 28 flattened pixel values</div>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-block enc">
              <div class="ab-title">Encoder FC · <span id="arch-hidden">400</span>-D</div>
              <div class="ab-sub">ReLU activation</div>
            </div>
            <div class="arch-arrow">↓ split ↓</div>
            <div class="arch-block enc">
              <div class="ab-title">μ head &nbsp;&amp;&nbsp; log σ² head · <span id="arch-latent">2</span>-D</div>
              <div class="ab-sub">Two parallel linear layers</div>
            </div>
            <div class="arch-arrow">↓ reparameterize ↓</div>
            <div class="arch-block lat">
              <div class="ab-title">Latent vector z · <span id="arch-latent2">2</span>-D</div>
              <div class="ab-sub">z = μ + σ · ε &nbsp;&nbsp;ε ∼ 𝒩(0, I)</div>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-block dec">
              <div class="ab-title">Decoder FC · <span id="arch-hidden2">400</span>-D</div>
              <div class="ab-sub">ReLU activation</div>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-block">
              <div class="ab-title">Output · 784-D</div>
              <div class="ab-sub">Sigmoid → reconstructed image</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span class="icon">📐</span> Loss Function</div>
          <div class="formula-box">
            <p><strong style="color:var(--text);">ELBO Loss</strong></p>
            <p style="margin-top:8px;">
              <code>ℒ = ℒ_recon + β · KL</code>
            </p>
            <p style="margin-top:14px;"><strong style="color:var(--text);">Reconstruction Loss</strong> (Binary Cross-Entropy)</p>
            <p><code>ℒ_recon = −Σ [ x·log(x̂) + (1−x)·log(1−x̂) ]</code></p>
            <p style="margin-top:14px;"><strong style="color:var(--text);">KL Divergence</strong> (closed form)</p>
            <p><code>KL = −½ Σ [ 1 + log σ² − μ² − σ² ]</code></p>
          </div>
          <div class="formula-box" style="margin-top:14px;">
            <p><strong style="color:var(--text);">Reparameterization Trick</strong></p>
            <p style="margin-top:6px;">The trick decouples stochasticity from parameters, making
               backpropagation through sampling possible:</p>
            <p style="margin-top:8px;"><code>z = μ + σ ⊙ ε</code> &nbsp; where &nbsp;<code>ε ∼ 𝒩(0, I)</code></p>
          </div>
          <div class="formula-box" style="margin-top:14px;">
            <p><strong style="color:var(--text);">Why 2-D Latent Space?</strong></p>
            <p style="margin-top:6px;">A 2-D latent space lets us visualise the full manifold as a
               2-D scatter plot. Higher dimensions improve quality but lose direct interpretability.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- ── TAB: Latent Space ── -->
    <div id="latent" class="tab-content">
      <div class="alert alert-warn">
        <span class="a-icon">⚠️</span>
        Train the model first (Training tab). Latent space visualisation requires a 2-D latent dimension.
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">🌐</span> 2-D Latent Space — MNIST</div>
        <p style="color:var(--muted); font-size:.88rem; margin-bottom:14px;">
          Each point is an MNIST digit encoded via the VAE's encoder. Colour = digit class (0–9).
          Well-separated clusters indicate a structured latent manifold.
        </p>
        <button class="btn btn-primary" onclick="loadLatentSpace()">🔄 Generate Plot</button>
        <div class="img-box small" id="latent-box" style="min-height:160px; margin-top:16px;">
          <div class="placeholder">
            <span class="ph-icon">🌐</span>
            Train the model, then click Generate Plot
          </div>
        </div>
      </div>
    </div>

    <!-- ── TAB: Reconstruction ── -->
    <div id="reconstruction" class="tab-content">
      <div class="alert alert-warn">
        <span class="a-icon">⚠️</span>
        Train the model first (Training tab) to enable reconstruction.
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">🔁</span> Original vs Reconstructed Digits</div>
        <p style="color:var(--muted); font-size:.88rem; margin-bottom:14px;">
          <strong style="color:var(--text);">Row 1:</strong> original MNIST samples &nbsp;|&nbsp;
          <strong style="color:var(--text);">Row 2:</strong> VAE reconstructions.
          Blurriness reflects the smoothing nature of the reconstruction loss.
        </p>
        <button class="btn btn-primary" onclick="loadReconstruction()">🎲 Random Batch</button>
        <div class="img-box" id="recon-box" style="min-height:260px; margin-top:16px;">
          <div class="placeholder">
            <span class="ph-icon">🔁</span>
            Train then click Random Batch
          </div>
        </div>
      </div>
    </div>

    <!-- ── TAB: Generation ── -->
    <div id="generation" class="tab-content">
      <div class="alert alert-info">
        <span class="a-icon">ℹ️</span>
        Adjust the latent sliders to navigate the learned manifold and decode new digit-like images.
        Grid view requires a 2-D latent space.
      </div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span class="icon">🎛️</span> Latent Controls</div>
          <div class="slider-wrap">
            <div class="slider-label">
              <span>Z<sub>1</sub> — Latent Dim 1</span>
              <span class="slider-val" id="z1-val">0.00</span>
            </div>
            <input type="range" id="z1" min="-3" max="3" step="0.05" value="0"
                   oninput="updateSlider('z1')"/>
          </div>
          <div class="slider-wrap">
            <div class="slider-label">
              <span>Z<sub>2</sub> — Latent Dim 2</span>
              <span class="slider-val" id="z2-val">0.00</span>
            </div>
            <input type="range" id="z2" min="-3" max="3" step="0.05" value="0"
                   oninput="updateSlider('z2')"/>
          </div>
          <div class="btn-row" style="margin-top:18px;">
            <button class="btn btn-primary" onclick="generateSample()">✨ Generate</button>
            <button class="btn btn-ghost" onclick="randomSample()">🎲 Random</button>
            <button class="btn btn-ghost" onclick="generateGrid()">📐 Grid</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span class="icon">🖼️</span> Generated Image</div>
          <div class="img-box small" id="gen-box" style="min-height:160px;">
            <div class="placeholder">
              <span class="ph-icon">✨</span>
              Train the model then click Generate
            </div>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer>
    <div class="footer-left">
      <img class="footer-avatar"
           src="https://avatars.githubusercontent.com/mnoorchenar"
           alt="Mohammad Noorchenarboo"/>
      <div>
        <div class="footer-name">Mohammad Noorchenarboo</div>
        <div class="footer-sub">Data Scientist · AI Researcher · Biostatistician</div>
      </div>
    </div>
    <div class="footer-links">
      <a class="footer-link" href="https://www.linkedin.com/in/mnoorchenar" target="_blank">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.07 2.07 0 1 1 0-4.14 2.07 2.07 0 0 1 0 4.14zM7.12 20.45H3.55V9h3.57v11.45zM22.23 0H1.77C.79 0 0 .77 0 1.72v20.56C0 23.23.79 24 1.77 24h20.46C23.21 24 24 23.23 24 22.28V1.72C24 .77 23.21 0 22.23 0z"/></svg>
        LinkedIn
      </a>
      <a class="footer-link" href="https://github.com/mnoorchenar" target="_blank">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0C17.3 4.7 18.3 5 18.3 5c.6 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3z"/></svg>
        GitHub
      </a>
      <a class="footer-link" href="https://mnoorchenar.github.io/" target="_blank">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
        Website
      </a>
      <a class="footer-link" href="https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en" target="_blank">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3L1 9l4 2.18V17c0 .96.5 1.84 1.26 2.33C7.93 20.5 9.87 21 12 21s4.07-.5 5.74-1.67C18.5 18.84 19 18 19 17v-5.82L21 10V17h2V9L12 3zm5 13.08c0 .41-.25.8-.68 1.09C15.08 17.96 13.6 18.5 12 18.5s-3.08-.54-4.32-1.33c-.43-.29-.68-.68-.68-1.09v-4.57l5 2.73 5-2.73v4.57zM12 12.72L3.97 9 12 5.28 20.03 9 12 12.72z"/></svg>
        Scholar
      </a>
    </div>
  </footer>

</div><!-- .app-shell -->

<script>
  /* ── Tab switching ── */
  function switchTab(id, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(id).classList.add('active');
  }

  /* ── Slider helper ── */
  function updateSlider(id) {
    const v = parseFloat(document.getElementById(id).value).toFixed(2);
    document.getElementById(id + '-val').textContent = v;
  }

  /* ── Img box helper ── */
  function showImg(boxId, b64) {
    document.getElementById(boxId).innerHTML =
      `<img src="data:image/png;base64,${b64}" alt="plot"/>`;
  }
  function showSpinner(boxId) {
    document.getElementById(boxId).innerHTML =
      `<div class="placeholder"><div class="spinner"></div><span>Generating…</span></div>`;
  }
  function showErr(boxId, msg) {
    document.getElementById(boxId).innerHTML =
      `<div class="placeholder" style="color:#f87171;"><span class="ph-icon">❌</span>${msg}</div>`;
  }

  /* ── Training ── */
  let pollInterval = null;

  async function startTraining() {
    const params = {
      epochs:        +document.getElementById('epochs').value,
      batch_size:    +document.getElementById('batch_size').value,
      learning_rate: +document.getElementById('learning_rate').value,
      hidden_dim:    +document.getElementById('hidden_dim').value,
      latent_dim:    +document.getElementById('latent_dim').value,
    };
    // Sync arch labels
    ['arch-hidden','arch-hidden2'].forEach(id => document.getElementById(id).textContent = params.hidden_dim);
    ['arch-latent','arch-latent2'].forEach(id => document.getElementById(id).textContent = params.latent_dim);

    document.getElementById('train-btn').disabled = true;
    document.getElementById('prog-wrap').style.display = 'block';

    const res = await fetch('/start_training', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(params)
    });
    const d = await res.json();
    if (d.status === 'started') {
      pollInterval = setInterval(pollProgress, 600);
    }
  }

  async function pollProgress() {
    const d = await (await fetch('/training_progress')).json();
    const bar = document.getElementById('prog-bar');
    bar.style.width = d.progress + '%';
    bar.textContent  = d.progress + '%';
    document.getElementById('epoch-val').textContent = `${d.current_epoch} / ${d.total_epochs}`;
    document.getElementById('loss-val').textContent  = d.current_loss ? d.current_loss.toFixed(4) : '—';

    const pill = document.getElementById('status-pill');
    if (d.is_training) {
      pill.className = 'status-pill s-training'; pill.textContent = 'Training…';
    } else if (d.trained) {
      pill.className = 'status-pill s-ready'; pill.textContent = 'Ready';
      document.getElementById('train-btn').disabled = false;
      clearInterval(pollInterval);
      updateLossCurve();
    } else {
      pill.className = 'status-pill s-idle'; pill.textContent = 'Not Trained';
    }
    if (d.current_epoch > 0) updateLossCurve();
  }

  async function updateLossCurve() {
    const d = await (await fetch('/training_curve')).json();
    if (d.image) showImg('loss-box', d.image);
  }

  async function resetModel() {
    if (!confirm('Reset model? All training progress will be lost.')) return;
    await fetch('/reset_model', {method:'POST'});
    location.reload();
  }

  /* ── Latent Space ── */
  async function loadLatentSpace() {
    showSpinner('latent-box');
    const d = await (await fetch('/latent_space')).json();
    d.error ? showErr('latent-box', d.error) : showImg('latent-box', d.image);
  }

  /* ── Reconstruction ── */
  async function loadReconstruction() {
    showSpinner('recon-box');
    const d = await (await fetch('/reconstruction')).json();
    d.error ? showErr('recon-box', d.error) : showImg('recon-box', d.image);
  }

  /* ── Generation ── */
  async function generateSample() {
    showSpinner('gen-box');
    const d = await (await fetch('/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        z1: +document.getElementById('z1').value,
        z2: +document.getElementById('z2').value
      })
    })).json();
    d.error ? showErr('gen-box', d.error) : showImg('gen-box', d.image);
  }

  async function randomSample() {
    const rnd = () => (Math.random()*6 - 3).toFixed(2);
    const z1 = rnd(), z2 = rnd();
    document.getElementById('z1').value = z1; updateSlider('z1');
    document.getElementById('z2').value = z2; updateSlider('z2');
    await generateSample();
  }

  async function generateGrid() {
    showSpinner('gen-box');
    const d = await (await fetch('/generate_grid')).json();
    d.error ? showErr('gen-box', d.error) : showImg('gen-box', d.image);
  }

  // Initial poll to reflect any pre-existing server state
  pollProgress();
</script>
</body>
</html>
'''

# ── Routes (unchanged from original) ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/start_training', methods=['POST'])
def start_training():
    if training_state['is_training']:
        return jsonify({'status': 'already_training'})
    p = request.json
    threading.Thread(
        target=train_vae_thread, daemon=True,
        args=(p.get('epochs',30), p.get('batch_size',128),
              p.get('learning_rate',1e-3), p.get('hidden_dim',400),
              p.get('latent_dim',2))
    ).start()
    return jsonify({'status': 'started'})

@app.route('/training_progress')
def training_progress():
    return jsonify({k: training_state[k] for k in
                    ('is_training','progress','current_epoch','total_epochs',
                     'current_loss','trained')})

@app.route('/reset_model', methods=['POST'])
def reset_model():
    global vae, training_state
    vae = None
    training_state = {
        'is_training':False,'progress':0,'current_epoch':0,
        'total_epochs':0,'losses':[],'trained':False,'current_loss':0
    }
    return jsonify({'status': 'reset'})

@app.route('/latent_space')
def latent_space():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet.'})
    if vae.latent_dim != 2:
        return jsonify({'error': 'Latent space visualisation requires 2-D latent dimension.'})
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(data_tensor)
        mu_np = mu.numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#16162a')
    ax.set_facecolor('#0f0f1a')
    sc = ax.scatter(mu_np[:,0], mu_np[:,1], c=labels, cmap='tab10',
                    alpha=.65, s=22, linewidths=0)
    ax.set_xlabel('z₁', fontsize=12, color='#94a3b8')
    ax.set_ylabel('z₂', fontsize=12, color='#94a3b8')
    ax.set_title('VAE Latent Space  ·  MNIST Digits', fontsize=14,
                 fontweight='bold', color='#e2e8f0', pad=14)
    ax.tick_params(colors='#64748b')
    for sp in ax.spines.values(): sp.set_color('#2e2e52')
    ax.grid(True, alpha=.15, color='#ffffff')
    cb = plt.colorbar(sc, ax=ax, ticks=range(10))
    cb.ax.yaxis.set_tick_params(color='#94a3b8')
    cb.ax.set_yticklabels([str(i) for i in range(10)], color='#94a3b8')
    cb.outline.set_edgecolor('#2e2e52')
    plt.tight_layout()
    return jsonify({'image': fig_to_b64(fig)})

@app.route('/reconstruction')
def reconstruction():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet.'})
    n = 10
    idx = np.random.choice(len(data), n, replace=False)
    vae.eval()
    with torch.no_grad():
        orig = data_tensor[idx]
        recon, _, _ = vae(orig)
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    fig.patch.set_facecolor('#16162a')
    for i in range(n):
        for row, arr, ttl in [(0, orig[i].numpy(), f'Original\n({labels[idx[i]]})'),
                               (1, recon[i].numpy(), 'Recon')]:
            axes[row,i].imshow(arr.reshape(28,28), cmap='gray')
            axes[row,i].set_title(ttl, fontsize=8, color='#94a3b8')
            axes[row,i].axis('off')
            axes[row,i].set_facecolor('#0f0f1a')
    fig.suptitle('Reconstruction Comparison', fontsize=13, fontweight='bold',
                 color='#e2e8f0', y=1.01)
    plt.tight_layout()
    return jsonify({'image': fig_to_b64(fig)})

@app.route('/generate', methods=['POST'])
def generate():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet.'})
    d = request.json
    z = torch.zeros(1, vae.latent_dim)
    z[0,0], z[0,1] = d['z1'], d['z2']
    vae.eval()
    with torch.no_grad():
        gen = vae.decode(z)
    fig, ax = plt.subplots(figsize=(5,5))
    fig.patch.set_facecolor('#16162a')
    ax.set_facecolor('#0f0f1a')
    ax.imshow(gen.numpy().reshape(28,28), cmap='gray')
    ax.set_title(f'z₁={d["z1"]:.2f}  z₂={d["z2"]:.2f}',
                 fontsize=12, color='#94a3b8')
    ax.axis('off')
    return jsonify({'image': fig_to_b64(fig)})

@app.route('/generate_grid')
def generate_grid():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet.'})
    if vae.latent_dim != 2:
        return jsonify({'error': 'Grid requires 2-D latent dimension.'})
    n = 15
    gx = np.linspace(-3, 3, n)
    gy = np.linspace(-3, 3, n)
    fig, axes = plt.subplots(n, n, figsize=(15,15))
    fig.patch.set_facecolor('#16162a')
    vae.eval()
    with torch.no_grad():
        for i, yi in enumerate(gy):
            for j, xi in enumerate(gx):
                g = vae.decode(torch.FloatTensor([[xi, yi]]))
                axes[i,j].imshow(g.numpy().reshape(28,28), cmap='gray')
                axes[i,j].axis('off')
    fig.suptitle('Latent Space Manifold  ·  15×15 Grid', fontsize=16,
                 fontweight='bold', color='#e2e8f0')
    plt.tight_layout()
    return jsonify({'image': fig_to_b64(fig)})

@app.route('/training_curve')
def training_curve():
    if not training_state['losses']:
        return jsonify({'error': 'No data yet.'})
    losses = training_state['losses']
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#16162a')
    ax.set_facecolor('#0f0f1a')
    ax.plot(losses, lw=2.5, color='#7c3aed')
    ax.fill_between(range(len(losses)), losses, alpha=.25, color='#7c3aed')
    ax.set_xlabel('Epoch', fontsize=11, color='#94a3b8')
    ax.set_ylabel('Loss', fontsize=11, color='#94a3b8')
    ax.set_title('VAE Training Loss', fontsize=13, fontweight='bold',
                 color='#e2e8f0', pad=12)
    ax.tick_params(colors='#64748b')
    for sp in ax.spines.values(): sp.set_color('#2e2e52')
    ax.grid(True, alpha=.15, color='#ffffff')
    plt.tight_layout()
    return jsonify({'image': fig_to_b64(fig)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)