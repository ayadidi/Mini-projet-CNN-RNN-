"""
Image Captioning — Flask Web App
=================================
Détecte automatiquement la structure du dossier :

    Flickr8k_Output/
    ├── vocabulary.pkl
    ├── models/           ← cherché en priorité ici
    │     ├── q1_lstm_CNN_epoch_best.pkl
    │     └── ...
    └── Flickr8k_Output/  ← fallback si models/ est imbriqué
         └── models/ ...

Lancement :
    pip install flask torch torchvision pillow
    python app.py
→ http://127.0.0.1:5000
"""

import io, math, pickle, warnings, json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
OUTPUT_DIR = Path("Flickr8k_Output")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 25


# ── Détection automatique des sous-dossiers ───────────────────────
def _find_subdir(base: Path, name: str) -> Path:
    """Cherche name/ dans base/ puis dans base/base.name/ (double imbrication)."""
    for candidate in [base / name, base / base.name / name]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Dossier '{name}/' introuvable dans {base}. "
        f"Vérifiez que OUTPUT_DIR pointe vers le bon répertoire.")


def _find_vocab(base: Path) -> Path:
    for candidate in [base / "vocabulary.pkl",
                      base / base.name / "vocabulary.pkl"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"vocabulary.pkl introuvable dans {base}.")


MODELS_DIR = _find_subdir(OUTPUT_DIR, "models")
LOGS_DIR   = OUTPUT_DIR / "logs"
VOCAB_PATH = _find_vocab(OUTPUT_DIR)

# ══════════════════════════════════════════════════════════════════
# VOCABULAIRE
# ══════════════════════════════════════════════════════════════════
with open(VOCAB_PATH, "rb") as f:
    _v = pickle.load(f)

word2idx   = _v["word2idx"]
idx2word   = _v["idx2word"]
vocab_size = len(word2idx)
PAD_IDX    = word2idx.get("<pad>",     0)
SOS_IDX    = word2idx.get("startseq", 1)
EOS_IDX    = word2idx.get("endseq",   2)

print(f"✅ Vocabulaire  : {vocab_size:,} tokens")
print(f"✅ Models dir   : {MODELS_DIR.resolve()}")
print(f"✅ Device       : {DEVICE}")


# ══════════════════════════════════════════════════════════════════
# ARCHITECTURES  (noms d'attributs identiques au notebook)
# ══════════════════════════════════════════════════════════════════
class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, backbone: str = "resnet50"):
        super().__init__()
        _wmap = {"resnet18": "ResNet18_Weights",
                 "resnet50": "ResNet50_Weights",
                 "resnet101":"ResNet101_Weights"}
        if backbone in _wmap:
            wc     = getattr(models, _wmap[backbone])
            resnet = getattr(models, backbone)(weights=wc.IMAGENET1K_V1)
        else:
            resnet = getattr(models, backbone)(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad_(False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed  = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn     = nn.BatchNorm1d(embed_size)

    def forward(self, x):
        f = self.resnet(x).view(x.size(0), -1)
        return self.bn(self.embed(f))


class DecoderRNN(nn.Module):
    # ⚠️  self.lstm (pas self.rnn) — correspond aux clés des .pkl sauvegardés
    def __init__(self, embed_size, hidden_size, vocab_size,
                 num_layers=1, dropout=0.0, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embed    = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.drop     = nn.Dropout(dropout)
        rnn_cls       = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.lstm     = rnn_cls(embed_size, hidden_size,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0,
                                batch_first=True)
        self.linear   = nn.Linear(hidden_size, vocab_size)

    @torch.no_grad()
    def sample(self, inputs, states=None, max_len=MAX_LEN):
        self.eval()
        res = []
        for _ in range(max_len):
            out, states = self.lstm(inputs, states)
            _, pred     = self.linear(out.squeeze(1)).max(dim=1)
            idx         = pred.item()
            res.append(idx)
            if idx == EOS_IDX:
                break
            inputs = self.drop(self.embed(pred)).unsqueeze(1)
        return res


class _PE(nn.Module):
    def __init__(self, d, maxlen=512, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size,
                 num_heads=8, num_layers=3, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.pos_enc = _PE(embed_size, drop=dropout)
        layer        = nn.TransformerDecoderLayer(embed_size, num_heads,
                                                   ff_dim, dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.fc          = nn.Linear(embed_size, vocab_size)

    def _mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()

    @torch.no_grad()
    def sample(self, feature, max_len=MAX_LEN):
        self.eval()
        tokens = torch.tensor([[SOS_IDX]], device=DEVICE)
        memory = feature.unsqueeze(1)
        res    = []
        for _ in range(max_len):
            tgt = self.pos_enc(self.embed(tokens))
            out = self.transformer(tgt, memory, tgt_mask=self._mask(tgt.size(1)))
            nxt = self.fc(out[:, -1]).argmax(dim=-1, keepdim=True)
            idx = nxt.item()
            if idx == EOS_IDX:
                break
            res.append(idx)
            tokens = torch.cat([tokens, nxt], dim=1)
        return res


# ══════════════════════════════════════════════════════════════════
# CATALOGUE — scan automatique des best .pkl disponibles
# ══════════════════════════════════════════════════════════════════
_KNOWN = [
    dict(tag="q1_lstm",           label="LSTM — emb 256",      group="Q1 · LSTM vs GRU",    rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q1_gru",            label="GRU  — emb 256",      group="Q1 · LSTM vs GRU",    rnn_type="gru",         embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q2_emb128",         label="LSTM — emb 128",      group="Q2 · Embedding",      rnn_type="lstm",        embed_size=128,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q2_emb256",         label="LSTM — emb 256",      group="Q2 · Embedding",      rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q2_emb512",         label="LSTM — emb 512",      group="Q2 · Embedding",      rnn_type="lstm",        embed_size=512,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q2_emb1024",        label="LSTM — emb 1024",     group="Q2 · Embedding",      rnn_type="lstm",        embed_size=1024, hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q3_resnet18",       label="LSTM — ResNet18",     group="Q3 · CNN backbone",   rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet18"),
    dict(tag="q3_resnet50",       label="LSTM — ResNet50",     group="Q3 · CNN backbone",   rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q4_drop0",          label="LSTM — dropout 0.0",  group="Q4 · Dropout",        rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q4_drop3",          label="LSTM — dropout 0.3",  group="Q4 · Dropout",        rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.3, backbone="resnet50"),
    dict(tag="q4_drop5",          label="LSTM — dropout 0.5",  group="Q4 · Dropout",        rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.5, backbone="resnet50"),
    dict(tag="q5_layers1",        label="LSTM — 1 couche",     group="Q5 · Nb couches",     rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q5_layers2",        label="LSTM — 2 couches",    group="Q5 · Nb couches",     rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=2, dropout=0.3, backbone="resnet50"),
    dict(tag="q5_layers3",        label="LSTM — 3 couches",    group="Q5 · Nb couches",     rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=3, dropout=0.3, backbone="resnet50"),
    dict(tag="q6_adam",           label="LSTM — Adam",         group="Q6 · Optimiseur",     rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="q6_sgd",            label="LSTM — SGD",          group="Q6 · Optimiseur",     rnn_type="lstm",        embed_size=256,  hidden_size=512, num_layers=1, dropout=0.0, backbone="resnet50"),
    dict(tag="bonus_transformer", label="Transformer",         group="🎁 Bonus",            rnn_type="transformer", embed_size=256,  hidden_size=512, num_layers=1, dropout=0.1, backbone="resnet50"),
]

AVAILABLE = [
    c for c in _KNOWN
    if (MODELS_DIR / f"{c['tag']}_CNN_epoch_best.pkl").exists()
    and (MODELS_DIR / f"{c['tag']}_RNN_epoch_best.pkl").exists()
]

# best = q1_lstm s'il existe, sinon premier disponible
BEST_TAG = next(
    (c["tag"] for c in AVAILABLE if c["tag"] == "q1_lstm"),
    AVAILABLE[0]["tag"] if AVAILABLE else None
)

if not AVAILABLE:
    raise RuntimeError(
        f"Aucun fichier *_epoch_best.pkl trouvé dans {MODELS_DIR}.\n"
        "Vérifiez que OUTPUT_DIR pointe vers votre Flickr8k_Output/ local."
    )

print(f"✅ {len(AVAILABLE)} modèles disponibles  (best → {BEST_TAG})")
for m in AVAILABLE:
    mark = " ★" if m["tag"] == BEST_TAG else ""
    print(f"   • {m['tag']}{mark}")


# ══════════════════════════════════════════════════════════════════
# CACHE + INFÉRENCE
# ══════════════════════════════════════════════════════════════════
_cache: dict = {}


def load_model(tag: str):
    if tag in _cache:
        return _cache[tag]
    cfg = next(c for c in AVAILABLE if c["tag"] == tag)
    print(f"  → Chargement {tag} …", flush=True)
    cnn = EncoderCNN(cfg["embed_size"], cfg["backbone"]).to(DEVICE)
    cnn.load_state_dict(torch.load(
        MODELS_DIR / f"{tag}_CNN_epoch_best.pkl", map_location=DEVICE))
    cnn.eval()
    if cfg["rnn_type"] == "transformer":
        rnn = TransformerCaptionDecoder(
            cfg["embed_size"], vocab_size, dropout=cfg["dropout"]).to(DEVICE)
    else:
        rnn = DecoderRNN(
            cfg["embed_size"], cfg["hidden_size"], vocab_size,
            cfg["num_layers"], cfg["dropout"], cfg["rnn_type"]).to(DEVICE)
    rnn.load_state_dict(torch.load(
        MODELS_DIR / f"{tag}_RNN_epoch_best.pkl", map_location=DEVICE))
    rnn.eval()
    _cache[tag] = (cnn, rnn, cfg["rnn_type"])
    print(f"  ✅ {tag} prêt")
    return _cache[tag]


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def ids_to_caption(ids: list) -> str:
    words = []
    for i in ids:
        w = idx2word.get(i, "")
        if w in ("endseq", "<pad>"):
            break
        if w != "startseq":
            words.append(w)
    return " ".join(words)


def predict(pil_image: Image.Image, tag: str) -> str:
    cnn, rnn, rnn_type = load_model(tag)
    t = _transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = cnn(t)
    ids = rnn.sample(feat) if rnn_type == "transformer" else rnn.sample(feat.unsqueeze(1))
    return ids_to_caption(ids)


def _load_history(tag: str):
    p = LOGS_DIR / f"{tag}_history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════
# HTML (embarqué)
# ══════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Image Captioning</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090c;--surf:#0e1218;--card:#131920;--border:#1c2530;--border2:#263347;
  --accent:#39e8a0;--accent-dim:rgba(57,232,160,.1);
  --gold:#f5c842;--gold-dim:rgba(245,200,66,.08);
  --red:#f0544f;--text:#c9d8e8;--muted:#4a5a6a;
  --mono:'DM Mono',monospace;--sans:'Syne',sans-serif;--r:14px;
}
body{background:var(--bg);color:var(--text);font-family:var(--mono);
  min-height:100vh;display:flex;flex-direction:column;align-items:center;
  padding:44px 16px 100px}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='.025'/%3E%3C/svg%3E")}
.wrap{position:relative;z-index:1;width:100%;max-width:920px}

/* HEADER */
.hdr{text-align:center;margin-bottom:38px}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:.6rem;
  letter-spacing:3px;text-transform:uppercase;color:var(--accent);
  border:1px solid rgba(57,232,160,.3);border-radius:20px;padding:4px 14px;margin-bottom:14px}
.pill::before{content:'';width:6px;height:6px;border-radius:50%;
  background:var(--accent);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.hdr h1{font-family:var(--sans);font-size:clamp(2rem,6vw,3.4rem);
  font-weight:800;letter-spacing:-2px;color:#fff;line-height:1}
.hdr h1 em{font-style:normal;color:var(--accent)}
.hdr p{margin-top:9px;color:var(--muted);font-size:.7rem;letter-spacing:.5px}

/* TABS */
.tabs{display:flex;gap:4px;background:var(--surf);border:1px solid var(--border);
  border-radius:12px;padding:4px;margin-bottom:18px}
.tab{flex:1;padding:8px;border:none;border-radius:8px;cursor:pointer;
  font-family:var(--mono);font-size:.7rem;color:var(--muted);
  background:transparent;transition:all .2s;letter-spacing:.2px}
.tab.active{background:var(--card);color:var(--text);border:1px solid var(--border2)}
.tab-panel{display:none}.tab-panel.active{display:block}

/* CARD */
.card{background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);padding:20px;transition:border-color .2s}
.card:hover{border-color:var(--border2)}
.clabel{font-size:.57rem;letter-spacing:2.5px;text-transform:uppercase;
  color:var(--muted);margin-bottom:11px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:580px){.g2{grid-template-columns:1fr}}

/* DROP ZONE */
#drop{border:2px dashed var(--border2);border-radius:10px;min-height:205px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;transition:border-color .2s,background .2s;
  position:relative;overflow:hidden}
#drop.over{border-color:var(--accent);background:var(--accent-dim)}
.hint{text-align:center;pointer-events:none}
.hint svg{color:var(--muted);margin-bottom:10px}
.hint p{font-size:.7rem;color:var(--muted);line-height:1.9}
.hint strong{color:var(--text);font-size:.75rem}
#prev{position:absolute;inset:0;width:100%;height:100%;
  object-fit:cover;border-radius:8px;display:none}
#drop.has-img .hint{display:none}
#drop.has-img #prev{display:block}
#chg{display:none;position:absolute;bottom:8px;right:8px;
  background:rgba(7,9,12,.82);border:1px solid var(--border2);border-radius:6px;
  color:var(--text);font-family:var(--mono);font-size:.6rem;
  padding:4px 10px;cursor:pointer;backdrop-filter:blur(6px)}
#drop.has-img #chg{display:block}
#fi{display:none}

/* MODEL LIST */
.mlist{display:flex;flex-direction:column;gap:5px;
  max-height:250px;overflow-y:auto;padding-right:2px;
  scrollbar-width:thin;scrollbar-color:var(--border2) transparent}
.msep{font-size:.55rem;letter-spacing:2px;text-transform:uppercase;
  color:var(--muted);padding:9px 2px 3px;border-top:1px solid var(--border);
  margin-top:2px;opacity:.7}
.msep:first-child{border-top:none;margin-top:0;padding-top:0}
.mopt{display:flex;align-items:center;gap:9px;padding:8px 11px;
  border:1px solid var(--border);border-radius:8px;cursor:pointer;
  transition:border-color .15s,background .15s;font-size:.69rem}
.mopt:hover{border-color:var(--border2)}
.mopt.sel{border-color:var(--accent);background:var(--accent-dim)}
.mopt.is-best{border-color:rgba(245,200,66,.4);background:var(--gold-dim)}
.mopt.is-best.sel{border-color:var(--gold)}
.mopt input{display:none}
.mdot{width:7px;height:7px;border-radius:50%;border:2px solid var(--muted);flex-shrink:0}
.mopt.sel .mdot{border-color:var(--accent);background:var(--accent)}
.mopt.is-best .mdot{border-color:rgba(245,200,66,.6)}
.mopt.is-best.sel .mdot{background:var(--gold);border-color:var(--gold)}
.mname{flex:1;color:var(--text);font-weight:500}
.mtag{color:var(--muted);font-size:.58rem}
.mbadge{font-size:.52rem;letter-spacing:.8px;text-transform:uppercase;
  padding:2px 6px;border-radius:8px;flex-shrink:0}
.mbadge-best{background:rgba(245,200,66,.14);color:var(--gold);border:1px solid rgba(245,200,66,.3)}
.mbadge-type{background:rgba(255,255,255,.04);color:var(--muted);border:1px solid var(--border)}

/* BUTTONS */
.brow{display:flex;gap:10px;margin-top:13px}
.btn{flex:1;padding:12px;border:none;border-radius:10px;cursor:pointer;
  font-family:var(--mono);font-size:.73rem;font-weight:500;letter-spacing:.2px;
  transition:opacity .15s,transform .1s;
  display:flex;align-items:center;justify-content:center;gap:7px}
.btn:active{transform:scale(.97)}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-gen{background:var(--accent);color:#021a0e}
.btn-gen.gold-mode{background:var(--gold);color:#1a1200}
.btn-cmp{background:var(--surf);color:var(--text);border:1px solid var(--border2)}
.btn-cmp:hover:not(:disabled){border-color:rgba(57,232,160,.35)}
.spin{width:13px;height:13px;border:2px solid rgba(0,0,0,.2);
  border-top-color:currentColor;border-radius:50%;
  animation:rot .55s linear infinite;display:none}
@keyframes rot{to{transform:rotate(360deg)}}

/* RESULT */
#res{background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);padding:20px;margin-top:14px;
  min-height:76px;transition:border-color .3s}
#res.ok{border-color:var(--accent)}
#res.gold{border-color:var(--gold)}
#res.err{border-color:var(--red)}
.cap-text{font-family:var(--sans);font-size:1.5rem;font-weight:700;
  color:#fff;letter-spacing:-.4px;line-height:1.3}
.cap-text::first-letter{text-transform:uppercase}
.cap-meta{margin-top:9px;font-size:.6rem;color:var(--muted);
  display:flex;flex-wrap:wrap;gap:10px}
.cap-meta span{color:var(--text)}
.cap-meta .gspan{color:var(--gold)}
.empty{color:var(--muted);font-size:.73rem;line-height:1.9}

/* COMPARE */
#cmp-wrap{display:none;margin-top:14px}
#cmp-wrap.vis{display:block}
.cmp-card{background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);overflow:hidden}
.cmp-hdr{display:grid;grid-template-columns:158px 1fr;gap:12px;
  padding:9px 18px;background:var(--surf);border-bottom:1px solid var(--border);
  font-size:.56rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted)}
.cmp-row{display:grid;grid-template-columns:158px 1fr;gap:12px;
  align-items:center;padding:11px 18px;border-bottom:1px solid var(--border);
  transition:background .15s}
.cmp-row:last-child{border-bottom:none}
.cmp-row:hover{background:rgba(255,255,255,.02)}
.cmp-row.best-row{background:var(--gold-dim);border-left:3px solid var(--gold)}
.cmp-tag{font-size:.62rem;color:var(--accent);display:flex;align-items:center;gap:5px}
.cmp-tag .star{color:var(--gold);font-size:.8rem}
.cmp-cap{font-size:.78rem;color:var(--text)}

/* DASHBOARD */
.dash-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));
  gap:12px;margin-top:12px}
.dcard{background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);padding:16px}
.dcard.is-best{border-color:rgba(245,200,66,.4)}
.dtitle{font-family:var(--sans);font-size:.82rem;font-weight:600;
  color:var(--text);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.dstar{color:var(--gold)}
.drow{display:flex;justify-content:space-between;padding:5px 0;
  border-bottom:1px solid var(--border);font-size:.65rem}
.drow:last-of-type{border-bottom:none}
.dkey{color:var(--muted)}.dval{color:var(--text);font-weight:500}
.dval.ac{color:var(--accent)}.dval.go{color:var(--gold)}
.bar-bg{height:3px;background:var(--border2);border-radius:2px;margin-top:8px}
.bar{height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--accent) 0%,#8bffcc 100%)}

/* STATUS */
#sb{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);
  background:var(--card);border:1px solid var(--border2);border-radius:30px;
  padding:7px 18px;font-size:.65rem;color:var(--muted);
  opacity:0;transition:opacity .3s;pointer-events:none;z-index:1000;white-space:nowrap}
#sb.show{opacity:1}
#sb.ok{color:var(--accent);border-color:rgba(57,232,160,.4)}
#sb.go{color:var(--gold);border-color:rgba(245,200,66,.4)}
#sb.err{color:var(--red);border-color:rgba(240,84,79,.4)}
</style>
</head>
<body>
<div class="wrap">

<header class="hdr">
  <div class="pill">TP Deep Learning · Flickr8k</div>
  <h1>Image <em>Captioning</em></h1>
  <p>{{ n_models }} modèles disponibles &nbsp;·&nbsp; best model : <strong style="color:var(--gold)">{{ best_tag }}</strong></p>
</header>

<div class="tabs">
  <button class="tab active" onclick="switchTab('infer',this)">🖼 &nbsp;Inférence</button>
  <button class="tab"        onclick="switchTab('compare',this)">⚡ &nbsp;Comparer</button>
  <button class="tab"        onclick="switchTab('dash',this)">📊 &nbsp;Dashboard</button>
</div>

<!-- ═══ TAB INFÉRENCE ═══ -->
<div id="tab-infer" class="tab-panel active">
  <div class="g2">

    <div class="card">
      <div class="clabel">01 — Image</div>
      <div id="drop">
        <div class="hint">
          <svg width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <p><strong>Glissez une image ici</strong><br/>ou cliquez pour parcourir<br/>JPG · PNG · WEBP</p>
        </div>
        <img id="prev" alt="preview"/>
        <button id="chg" type="button">↺ Changer</button>
      </div>
      <input id="fi" type="file" accept="image/*"/>
    </div>

    <div class="card">
      <div class="clabel">02 — Modèle (★ = best)</div>
      <div class="mlist" id="mlist">
        {% set ns = namespace(lg='') %}
        {% for m in models %}
          {% if m.group != ns.lg %}
            <div class="msep">{{ m.group }}</div>
            {% set ns.lg = m.group %}
          {% endif %}
          <label class="mopt{% if m.tag==best_tag %} is-best{% endif %}{% if loop.first %} sel{% endif %}">
            <input type="radio" name="model" value="{{ m.tag }}"{% if loop.first %} checked{% endif %}/>
            <div class="mdot"></div>
            <span class="mname">{{ m.label }}</span>
            {% if m.tag == best_tag %}
              <span class="mbadge mbadge-best">★ best</span>
            {% else %}
              <span class="mbadge mbadge-type">{{ m.rnn_type }}</span>
            {% endif %}
          </label>
        {% endfor %}
      </div>
      <div class="brow">
        <button class="btn btn-gen" id="gen-btn" disabled>
          <div class="spin" id="gspin"></div>
          <span id="gtxt">▶ &nbsp;Générer</span>
        </button>
      </div>
    </div>
  </div>

  <div id="res">
    <div class="clabel">03 — Résultat</div>
    <div class="empty">Chargez une image et cliquez sur <strong>▶ Générer</strong>.<br/>
    Le modèle <span style="color:var(--gold)">★ {{ best_tag }}</span> est sélectionné par défaut.</div>
  </div>
</div>

<!-- ═══ TAB COMPARER ═══ -->
<div id="tab-compare" class="tab-panel">
  <div class="g2" style="margin-bottom:14px">
    <div class="card">
      <div class="clabel">Image à comparer</div>
      <img id="prev2" style="width:100%;max-height:175px;object-fit:cover;
           border-radius:8px;display:none;margin-bottom:10px"/>
      <div id="hint2" style="font-size:.7rem;color:var(--muted);
           text-align:center;padding:16px 0;line-height:1.8">
        Utilisez l'image de l'onglet Inférence<br/>ou chargez-en une nouvelle.
      </div>
      <input id="fi2" type="file" accept="image/*" style="display:none"/>
      <button onclick="document.getElementById('fi2').click()"
        style="width:100%;padding:8px;background:var(--surf);
               border:1px solid var(--border2);border-radius:8px;
               color:var(--text);font-family:var(--mono);font-size:.68rem;cursor:pointer;margin-top:4px">
        📂 Charger une autre image
      </button>
    </div>
    <div class="card" style="display:flex;flex-direction:column;justify-content:space-between">
      <div>
        <div class="clabel">À propos</div>
        <div style="font-size:.68rem;color:var(--muted);line-height:2">
          Lance <strong style="color:var(--text)">{{ n_models }} modèles</strong> en parallèle.<br/>
          Le modèle <span style="color:var(--gold)">★ {{ best_tag }}</span> est mis en évidence.<br/>
          Les modèles sont mis en cache après le 1er appel.
        </div>
      </div>
      <button class="btn btn-cmp" id="cmp-btn" style="margin-top:14px" disabled>
        <div class="spin" id="cspin"></div>
        <span id="ctxt">⚡ &nbsp;Comparer tous les modèles</span>
      </button>
    </div>
  </div>
  <div id="cmp-wrap">
    <div class="cmp-card">
      <div class="cmp-hdr"><div>Modèle</div><div>Caption générée</div></div>
      <div id="cmp-body"></div>
    </div>
  </div>
</div>

<!-- ═══ TAB DASHBOARD ═══ -->
<div id="tab-dash" class="tab-panel">
  <div style="margin-top:4px">
    <div class="clabel">Résultats d'entraînement — tous les modèles</div>
    <div class="dash-grid">
      {% for m in models %}
      <div class="dcard{% if m.tag==best_tag %} is-best{% endif %}">
        <div class="dtitle">
          {% if m.tag==best_tag %}<span class="dstar">★</span>{% endif %}
          {{ m.tag }}
        </div>
        {% if m.history %}
        <div class="drow"><span class="dkey">Best val loss</span>
          <span class="dval {% if m.tag==best_tag %}go{% else %}ac{% endif %}">
            {{ "%.4f"|format(m.history.val_loss|min) }}</span></div>
        <div class="drow"><span class="dkey">Best val PPL</span>
          <span class="dval">{{ "%.2f"|format(m.history.val_ppl|min) }}</span></div>
        <div class="drow"><span class="dkey">Epochs</span>
          <span class="dval">{{ m.history.val_loss|length }}</span></div>
        <div class="drow"><span class="dkey">Type</span>
          <span class="dval">{{ m.rnn_type.upper() }} · {{ m.backbone }}</span></div>
        <div class="bar-bg">
          <div class="bar" style="width:{{ [100-(m.history.val_loss|min/5.5*100),4]|max }}%"></div>
        </div>
        {% else %}
        <div class="drow"><span class="dkey" style="font-style:italic">Pas d'historique</span></div>
        {% endif %}
      </div>
      {% endfor %}
    </div>
  </div>
</div>

</div><!-- /wrap -->
<div id="sb"></div>

<script>
const BEST = "{{ best_tag }}";
let file1=null, file2=null, sbTimer=null;

function status(msg,cls='',dur=2800){
  clearTimeout(sbTimer);
  const el=document.getElementById('sb');
  el.textContent=msg; el.className='show '+cls;
  sbTimer=setTimeout(()=>el.className='',dur);
}
function switchTab(n,btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+n).classList.add('active'); btn.classList.add('active');
}

// ── drop zone tab1 ──
const drop=document.getElementById('drop');
drop.addEventListener('click',e=>{if(e.target!==document.getElementById('chg'))document.getElementById('fi').click()});
document.getElementById('chg').addEventListener('click',e=>{e.stopPropagation();document.getElementById('fi').click()});
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',e=>setFile1(e.dataTransfer.files[0]));
document.getElementById('fi').addEventListener('change',e=>setFile1(e.target.files[0]));

function setFile1(f){
  if(!f||!f.type.startsWith('image/'))return status('Format non supporté','err');
  file1=f; file2=f;
  const url=URL.createObjectURL(f);
  document.getElementById('prev').src=url; drop.classList.add('has-img');
  // sync tab2
  const p2=document.getElementById('prev2');
  p2.src=url; p2.style.display='block';
  document.getElementById('hint2').style.display='none';
  document.getElementById('gen-btn').disabled=false;
  document.getElementById('cmp-btn').disabled=false;
  document.getElementById('res').className='';
  document.getElementById('res').innerHTML=`<div class="clabel">03 — Résultat</div>
    <div class="empty">Image prête · cliquez sur <strong>▶ Générer</strong>.</div>`;
  document.getElementById('cmp-wrap').classList.remove('vis');
  updateGenStyle(); status('Image chargée ✓','ok');
}
document.getElementById('fi2').addEventListener('change',e=>{
  const f=e.target.files[0]; if(!f)return;
  file2=f; const url=URL.createObjectURL(f);
  const p2=document.getElementById('prev2');
  p2.src=url; p2.style.display='block';
  document.getElementById('hint2').style.display='none';
  document.getElementById('cmp-btn').disabled=false;
  document.getElementById('cmp-wrap').classList.remove('vis');
  status('Image chargée ✓','ok');
});

// ── model select ──
document.getElementById('mlist').addEventListener('click',e=>{
  const opt=e.target.closest('.mopt'); if(!opt)return;
  document.querySelectorAll('.mopt').forEach(o=>o.classList.remove('sel'));
  opt.classList.add('sel'); opt.querySelector('input').checked=true;
  updateGenStyle();
});
function selModel(){const r=document.querySelector('.mopt.sel input');return r?r.value:BEST}
function updateGenStyle(){
  const btn=document.getElementById('gen-btn');
  const txt=document.getElementById('gtxt');
  if(selModel()===BEST){btn.classList.add('gold-mode');txt.textContent='★  Générer (best model)'}
  else{btn.classList.remove('gold-mode');txt.textContent='▶  Générer'}
}
updateGenStyle();

// ── POST ──
async function post(url,file,model){
  const fd=new FormData(); fd.append('image',file);
  if(model)fd.append('model',model);
  return (await fetch(url,{method:'POST',body:fd})).json();
}

// ── GENERATE ──
document.getElementById('gen-btn').addEventListener('click',async()=>{
  if(!file1)return;
  const btn=document.getElementById('gen-btn'),spin=document.getElementById('gspin');
  const txt=document.getElementById('gtxt'),res=document.getElementById('res');
  btn.disabled=true; spin.style.display='block'; txt.textContent='Génération…';
  res.className=''; res.innerHTML='<div class="clabel">03 — Résultat</div><div class="empty">⏳ Inférence…</div>';
  try{
    const d=await post('/predict',file1,selModel());
    if(d.error){
      res.className='err';
      res.innerHTML=`<div class="clabel">Erreur</div><div class="cap-text" style="color:var(--red);font-size:.95rem">${d.error}</div>`;
      status('Erreur : '+d.error,'err',5000);
    }else{
      const isB=d.tag===BEST;
      res.className=isB?'gold':'ok';
      res.innerHTML=`
        <div class="clabel">03 — Résultat</div>
        <div class="cap-text">${d.caption}</div>
        <div class="cap-meta">
          <div>Modèle <span class="${isB?'gspan':''}">${isB?'★ ':''}${d.tag}</span></div>
          <div>Type <span>${d.rnn_type.toUpperCase()}</span></div>
          <div>Backbone <span>${d.backbone}</span></div>
          <div>Emb <span>${d.embed_size}</span></div>
        </div>`;
      status(isB?'★ Caption (best model)':'Caption générée ✓',isB?'go':'ok');
    }
  }catch(e){res.innerHTML='<div class="clabel">Erreur réseau</div><div class="empty">'+e+'</div>';status('Erreur réseau','err')}
  finally{btn.disabled=false;spin.style.display='none';updateGenStyle()}
});

// ── COMPARE ──
document.getElementById('cmp-btn').addEventListener('click',async()=>{
  const f=file2||file1; if(!f)return;
  const btn=document.getElementById('cmp-btn'),spin=document.getElementById('cspin');
  const txt=document.getElementById('ctxt');
  btn.disabled=true; spin.style.display='block'; txt.textContent='Comparaison…';
  document.getElementById('cmp-wrap').classList.remove('vis');
  try{
    const d=await post('/compare',f,null);
    if(d.error){status(d.error,'err',4000);return}
    const body=document.getElementById('cmp-body'); body.innerHTML='';
    d.results.forEach(r=>{
      const isB=r.tag===BEST;
      const row=document.createElement('div');
      row.className='cmp-row'+(isB?' best-row':'');
      row.innerHTML=`<div class="cmp-tag">${isB?'<span class="star">★</span>':'<span style="color:var(--muted)">·</span>'} ${r.tag}</div>
                     <div class="cmp-cap">${r.caption}</div>`;
      body.appendChild(row);
    });
    document.getElementById('cmp-wrap').classList.add('vis');
    status(`${d.results.length} modèles comparés ✓`,'ok');
  }catch(e){status('Erreur réseau','err')}
  finally{btn.disabled=false;spin.style.display='none';txt.textContent='⚡  Comparer tous les modèles'}
});
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)


@app.route("/")
def index():
    enriched = []
    for c in AVAILABLE:
        e = dict(c)
        e["history"] = _load_history(c["tag"])
        enriched.append(e)
    return render_template_string(
        HTML, models=enriched, best_tag=BEST_TAG, n_models=len(AVAILABLE))


@app.route("/predict", methods=["POST"])
def route_predict():
    file = request.files.get("image")
    tag  = request.form.get("model", BEST_TAG)
    if not file:
        return jsonify({"error": "Aucune image reçue."}), 400
    try:
        img     = Image.open(io.BytesIO(file.read())).convert("RGB")
        caption = predict(img, tag)
        cfg     = next(c for c in AVAILABLE if c["tag"] == tag)
        return jsonify({"caption": caption, "tag": tag,
                        "rnn_type": cfg["rnn_type"], "embed_size": cfg["embed_size"],
                        "backbone": cfg["backbone"], "is_best": tag == BEST_TAG})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def route_compare():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Aucune image reçue."}), 400
    try:
        raw     = file.read()
        results = []
        for cfg in AVAILABLE:
            img     = Image.open(io.BytesIO(raw)).convert("RGB")
            caption = predict(img, cfg["tag"])
            results.append({"tag": cfg["tag"], "caption": caption,
                             "rnn_type": cfg["rnn_type"],
                             "is_best": cfg["tag"] == BEST_TAG})
        # best model toujours en premier
        results.sort(key=lambda r: (0 if r["is_best"] else 1, r["tag"]))
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 54)
    print("  🖼  Image Captioning — Flask App")
    print("═" * 54)
    print(f"  Output dir  : {OUTPUT_DIR.resolve()}")
    print(f"  Models dir  : {MODELS_DIR.resolve()}")
    print(f"  Vocab       : {VOCAB_PATH.resolve()}")
    print(f"  Modèles     : {len(AVAILABLE)}")
    print(f"  Best model  : {BEST_TAG}")
    print(f"  Device      : {DEVICE}")
    print("═" * 54)
    print("  → http://127.0.0.1:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
