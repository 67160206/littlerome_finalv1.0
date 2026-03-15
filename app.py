"""
Littlerome AI Vision — Streamlit Dashboard
YOLOv11 · Pipeline Detection · best.pt
Classes: pipeline_legit, water_leak, pipe_crack, flame, corrosion
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import datetime
from zoneinfo import ZoneInfo

TZ_TH = ZoneInfo('Asia/Bangkok')
def now_th(): return datetime.datetime.now(TZ_TH)
import os
import io
import time
import tempfile
import pandas as pd
from collections import Counter

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Littlerome AI Vision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────────────────────────────────────────
# CSS — dark industrial theme matching the original HTML
# ───────────────────────────────────────────────────────────────
_CSS_INJECT = """
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:     #0d1117;
  --bg2:    #161b22;
  --bg3:    #1c2230;
  --bg4:    #21262d;
  --border: #30363d;
  --border2:#3a4049;
  --text:   #e6edf3;
  --text2:  #8b949e;
  --text3:  #484f58;
  --blue:   #2188ff;
  --blue-g: rgba(33,136,255,.15);
  --red:    #f85149;
  --red-g:  rgba(248,81,73,.15);
  --green:  #3fb950;
  --green-g:rgba(63,185,80,.15);
  --orange: #e3b341;
  --orng-g: rgba(227,179,65,.15);
  --purple: #bc8cff;
  --purp-g: rgba(188,140,255,.15);
  --font: 'Syne', sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --r:  10px;
  --rs: 6px;
}

/* ── GLOBAL ── */
html, body, [class*="css"], [class*="stMarkdown"],
[class*="stText"], button, input, select, label {
  font-family: var(--font) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── HIDE DEFAULT STREAMLIT HEADER PADDING ── */
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 8px;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--text2);
  font-family: var(--font) !important;
  font-size: 13px;
  font-weight: 600;
  padding: 10px 16px;
  border: none;
  border-radius: 0;
  border-bottom: 2px solid transparent;
  transition: all .15s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); }
.stTabs [aria-selected="true"] {
  color: var(--blue) !important;
  border-bottom: 2px solid var(--blue) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--bg);
  padding: 24px 0;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── METRICS ── */
[data-testid="metric-container"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 18px 20px !important;
  position: relative;
  overflow: hidden;
}
[data-testid="metric-container"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent, var(--blue));
}
[data-testid="stMetricLabel"] {
  color: var(--text2) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: .5px !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 28px !important;
  font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

/* ── BUTTONS ── */
.stButton > button {
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  border-radius: var(--rs) !important;
  transition: all .15s !important;
  border: none !important;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button:active { transform: scale(.97) !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--bg2);
  border: 2px dashed var(--border2);
  border-radius: var(--r);
}
[data-testid="stFileUploader"]:hover { border-color: var(--blue); }

/* ── INPUTS / SLIDERS ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [data-testid="stTickBar"] { display: none; }
input[type="range"] { accent-color: var(--blue) !important; }

.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  border-radius: var(--rs) !important;
}

/* ── CAMERA INPUT ── */
[data-testid="stCameraInput"] { background: #000 !important; border-radius: var(--r) !important; }
[data-testid="stCameraInput"] > div { border-radius: var(--r) !important; }

/* ── DATAFRAME / TABLE ── */
[data-testid="stDataFrame"] { background: var(--bg2) !important; }
.dvn-scroller { background: var(--bg2) !important; }

/* ── EXPANDER ── */
details { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }
summary { color: var(--text) !important; font-weight: 600 !important; padding: 10px 16px !important; }

/* ── CUSTOM COMPONENTS ── */
.lv-header {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 10px 24px; margin: 0 -1rem 0 -1rem;
}
.lv-logo {
  display: flex; align-items: center; gap: 10px;
  font-size: 16px; font-weight: 800; letter-spacing: .5px;
}
.lv-logo-icon {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, #2188ff, #1a5fb4);
  border-radius: 8px; display: flex; align-items: center;
  justify-content: center; font-size: 16px;
}
.lv-conn {
  display: flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 12px; color: var(--text2);
}
.lv-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); box-shadow: 0 0 8px var(--green);
  animation: pulse-g 2s infinite;
}
@keyframes pulse-g { 0%,100%{opacity:1} 50%{opacity:.4} }
.lv-time { font-family: var(--mono); font-size: 12px; color: var(--text2); }
.lv-ver  { font-size: 11px; color: var(--text3); margin-right: 8px; }
.lv-avatar {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, #2188ff, #7c3aed);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: #fff;
}

.kpi-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--r); padding: 18px 20px;
  position: relative; overflow: hidden;
}
.kpi-card::before {
  content:''; position:absolute; top:0;left:0;right:0; height:2px;
  background: var(--kpi-accent, var(--blue));
}
.kpi-label { font-size:11px;font-weight:700;color:var(--text2);letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px; }
.kpi-val   { font-size:30px;font-weight:800;font-family:var(--mono); }
.kpi-sub   { font-size:11px;color:var(--text2);margin-top:4px; }
.kpi-icon  { position:absolute;right:16px;top:50%;transform:translateY(-50%);opacity:.12;font-size:42px; }

.card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--r); padding: 20px;
}
.card-title {
  font-size: 13px; font-weight: 700; letter-spacing: .3px;
  color: var(--text); margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.card-row {
  display:flex;align-items:center;justify-content:space-between;
  padding: 8px 0; border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.card-row:last-child { border-bottom: none; }

.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 700; letter-spacing: .3px;
}
.badge-ok     { background:var(--green-g);color:var(--green);border:1px solid rgba(63,185,80,.3); }
.badge-fault  { background:var(--red-g);  color:var(--red);  border:1px solid rgba(248,81,73,.3); animation: blink .8s infinite; }
.badge-warn   { background:var(--orng-g); color:var(--orange); }
.badge-info   { background:var(--blue-g); color:var(--blue); }
.badge-idle   { background:var(--bg3);    color:var(--text2); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.55} }

.det-item {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: var(--rs); padding: 10px 14px; margin-bottom: 8px;
}
.det-item-header { display:flex;align-items:center;justify-content:space-between;margin-bottom:4px; }
.det-name  { font-size:13px; font-weight:700; }
.det-conf  { font-family:var(--mono); font-size:12px; color:var(--text2); }
.conf-bar  { height:3px; background:var(--border); border-radius:2px; }
.conf-fill { height:100%; border-radius:2px; }

.sys-row {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 0; border-bottom:1px solid var(--border); font-size:13px;
}
.sys-row:last-child { border:none; }
.sdot { width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px; }

.empty-state { text-align:center;padding:48px 20px;color:var(--text3); }
.empty-icon  { font-size:40px;margin-bottom:12px;opacity:.4; }
.empty-text  { font-size:14px;color:var(--text2); }
.empty-sub   { font-size:12px;margin-top:4px;color:var(--text3); }

.hist-row {
  display:flex;align-items:center;gap:12px;
  padding:10px 0; border-bottom:1px solid var(--border);
  font-size:13px; cursor:pointer;
}
.hist-row:hover { background: var(--bg3); border-radius: var(--rs); }
.hist-row:last-child { border:none; }

.alert-box {
  background:var(--red-g); border:1px solid rgba(248,81,73,.4);
  border-radius:var(--r); padding:16px 20px; margin-bottom:16px;
  display:flex; align-items:flex-start; gap:12px;
}
.alert-icon { font-size:24px; animation: shake .4s infinite; }
@keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-3px)} 75%{transform:translateX(3px)} }

.prog-wrap  { background:var(--bg3);border-radius:4px;height:8px;overflow:hidden; }
.prog-fill  { height:100%;background:linear-gradient(90deg,var(--blue),#7c3aed);border-radius:4px;transition:width .3s; }

.scan-img { border-radius:var(--r);overflow:hidden;border:2px solid var(--border); }
.text-mono { font-family:var(--mono); }
.text-muted{ color:var(--text2); }
.text-blue { color:var(--blue); }
.text-red  { color:var(--red);  }
.text-green{ color:var(--green);}
.text-orange{color:var(--orange);}
.text-purple{color:var(--purple);}

.mt-8  { margin-top:8px; }
.mt-12 { margin-top:12px; }
.mt-16 { margin-top:16px; }
.mt-24 { margin-top:24px; }
</style>
"""
# Inject CSS — use st.html (Streamlit >=1.31) which handles <style> correctly;
# fall back to st.markdown for older versions.
try:
    st.html(_CSS_INJECT)
except AttributeError:
    st.markdown(_CSS_INJECT, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────────────────────
defaults = {
    "history": [],
    "total_scans": 0,
    "total_faults": 0,
    "conf_threshold": 0.15,
    "alert_enabled": True,
    "autosave": True,
    "cam_continuous": False,
    "settings_threshold": 15,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────────────────────────────────────────
# CONSTANTS — class colours (keyword-match approach)
# ───────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "legit":     "#3fb950",   # green  — pipeline OK
    "pipeline":  "#3fb950",
    "water":     "#2188ff",   # blue   — water leak
    "leak":      "#2188ff",
    "crack":     "#e3b341",   # orange — pipe crack
    "pipe_crack":"#e3b341",
    "flame":     "#f85149",   # red    — flame / fire
    "fire":      "#f85149",
    "corrosion": "#bc8cff",   # purple — corrosion / rust
    "rust":      "#e05c00",
}

DEFECT_EMOJI = {
    "legit":     "✅",
    "pipeline":  "✅",
    "water":     "💧",
    "leak":      "💧",
    "crack":     "🔧",
    "pipe_crack":"🔧",
    "flame":     "🔥",
    "fire":      "🔥",
    "corrosion": "⚠️",
    "rust":      "🟫",
}

def class_color(name: str) -> str:
    nl = name.lower().replace(" ", "_")
    for k, v in CLASS_COLORS.items():
        if k in nl:
            return v
    return "#8b949e"

def class_emoji(name: str) -> str:
    nl = name.lower().replace(" ", "_")
    for k, v in DEFECT_EMOJI.items():
        if k in nl:
            return v
    return "🔍"

def is_fault(name: str) -> bool:
    nl = name.lower()
    return not ("legit" in nl)

# ───────────────────────────────────────────────────────────────
# ENSEMBLE — 6 dedicated models, one per class
# ───────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))

PIPELINE_MODELS = {
    "pipeline_legit": ["pipeline_legit"],
    "water_leak":     ["water_leak"],
    "pipe_crack":     ["pipecrack", "pipe_crack"],
    "flame":          ["flame"],
    "corrosion":      ["corrosion"],
    "rust":           ["rust"],
}

def _find_model_file(keywords):
    base_dirs = [_script_dir, os.getcwd()]
    mount_src = "/mount/src"
    if os.path.isdir(mount_src):
        base_dirs.append(mount_src)
        for sub in os.listdir(mount_src):
            sub_path = os.path.join(mount_src, sub)
            if os.path.isdir(sub_path):
                base_dirs.append(sub_path)
    for d in base_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for f in os.listdir(d):
                fl = f.lower().replace("-", "_")
                if f.endswith(".pt") and any(k in fl for k in keywords):
                    return os.path.join(d, f)
        except Exception:
            continue
    return None

@st.cache_resource(show_spinner=False)
def load_single_model(path):
    return YOLO(path)

@st.cache_resource(show_spinner=False)
def load_all_models():
    loaded = {}
    missing = []
    for cls_name, keywords in PIPELINE_MODELS.items():
        path = _find_model_file(keywords)
        if path:
            try:
                loaded[cls_name] = load_single_model(path)
            except Exception as e:
                missing.append(f"{cls_name}: {e}")
        else:
            missing.append(cls_name)
    return loaded, missing

_ensemble, _missing_models = load_all_models()
model_ok    = len(_ensemble) > 0
model       = None
class_names = list(_ensemble.keys()) if _ensemble else list(PIPELINE_MODELS.keys())

if _missing_models:
    st.warning(f"⚠️ Model ที่ยังไม่พบ: {', '.join(_missing_models)}")
if not _ensemble:
    st.error("❌ ไม่พบไฟล์ .pt เลย — กรุณา push model ทั้ง 6 ไฟล์ขึ้น GitHub")

with st.expander("🔍 Debug: ตรวจสอบ Model Paths", expanded=False):
    for cls_name, keywords in PIPELINE_MODELS.items():
        p = _find_model_file(keywords)
        if p:
            st.markdown(f"✅ **{cls_name}** → `{p}`")
        else:
            st.markdown(f"❌ **{cls_name}** → ไม่พบ (ค้นหา: {keywords})")
    st.markdown(f"**script_dir:** `{_script_dir}`")
    st.markdown(f"**cwd:** `{os.getcwd()}`")
    if _ensemble:
        st.markdown(f"**โหลดแล้ว:** {len(_ensemble)}/6 models")

# ───────────────────────────────────────────────────────────────
# INFERENCE HELPER — runs all 6 ensemble models
# ───────────────────────────────────────────────────────────────
def run_inference(img_pil, conf=0.15):
    if not model_ok or not _ensemble:
        return img_pil, []

    img_np    = np.array(img_pil.convert("RGB"))
    annotated = img_np.copy()
    detections = []

    for cls_name, mdl in _ensemble.items():
        try:
            results = mdl(img_np, conf=conf, verbose=False)[0]
        except Exception:
            continue
        for box in results.boxes:
            conf_val = float(box.conf[0])
            name     = cls_name
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            hex_c = class_color(name)
            r = int(hex_c[1:3], 16)
            g = int(hex_c[3:5], 16)
            b = int(hex_c[5:7], 16)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (r, g, b), 2)
            label = f"{name}  {conf_val:.0%}"
            lw    = max(len(label) * 8, 60)
            cv2.rectangle(annotated, (x1, y1 - 24), (x1 + lw, y1), (r, g, b), -1)
            cv2.putText(
                annotated, label,
                (x1 + 4, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
            )
            detections.append({
                "class":      name,
                "confidence": round(conf_val * 100, 1),
                "bbox":       [x1, y1, x2, y2],
                "color":      hex_c,
                "fault":      is_fault(name),
                "emoji":      class_emoji(name),
            })

    return Image.fromarray(annotated), detections


def save_to_history(source: str, img_pil: Image.Image | None, detections: list):
    if not st.session_state.autosave:
        return
    thumb = None
    if img_pil:
        thumb = img_pil.copy()
        thumb.thumbnail((120, 90))
    has_fault = any(d["fault"] for d in detections)
    st.session_state.history.insert(
        0,
        {
            "ts":         now_th().strftime("%Y-%m-%d %H:%M:%S"),
            "source":     source,
            "status":     "FAULT" if has_fault else "OK",
            "detections": detections,
            "thumbnail":  thumb,
        },
    )
    st.session_state.total_scans += 1
    if has_fault:
        st.session_state.total_faults += 1

# ───────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────
now_str = now_th().strftime("%H:%M:%S · %d %b %Y")
_active_model_label = f"{len(_ensemble)}/6 models"
st.markdown(
    f"""
<div class="lv-header">
  <div style="display:flex;align-items:center;gap:20px">
    <div class="lv-logo">
      <div class="lv-logo-icon">👁</div>
      Littlerome<span style="color:#2188ff">AI</span>&nbsp;Vision
    </div>
    <div class="lv-conn">
      <div class="lv-dot"></div> CONNECTED
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="lv-time">{now_str}</span>
    <span style="font-family:var(--mono);font-size:11px;color:var(--blue);
      background:rgba(33,136,255,.12);padding:3px 10px;border-radius:20px;
      border:1px solid rgba(33,136,255,.3)">🧠 {_active_model_label}</span>
    <span class="lv-ver">v1.0.0</span>
    <div class="lv-avatar">OP</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────
# MAIN TABS
# ───────────────────────────────────────────────────────────────
tab_dash, tab_cam, tab_img, tab_vid, tab_hist, tab_settings = st.tabs(
    ["📊 Dashboard", "📷 Live Camera", "🖼 Upload Image", "🎬 Upload Video", "📋 History", "⚙️ Settings"]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab_dash:
    total   = st.session_state.total_scans
    faults  = st.session_state.total_faults
    ok_cnt  = total - faults
    pass_rt = round(ok_cnt / total * 100) if total > 0 else 100

    all_dets   = [d for h in st.session_state.history for d in h.get("detections", [])]
    avg_conf   = round(sum(d["confidence"] for d in all_dets) / len(all_dets), 1) if all_dets else 0
    defect_cnt = Counter(d["class"] for d in all_dets if d["fault"])

    # KPI row
    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px">
  <div class="kpi-card" style="--kpi-accent:var(--blue)">
    <div class="kpi-label">Total Inspected</div>
    <div class="kpi-val text-blue">{total}</div>
    <div class="kpi-sub">All time scans</div>
    <div class="kpi-icon">🔍</div>
  </div>
  <div class="kpi-card" style="--kpi-accent:var(--red)">
    <div class="kpi-label">Defects Found</div>
    <div class="kpi-val text-red">{faults}</div>
    <div class="kpi-sub">Fault rate: {100-pass_rt}%</div>
    <div class="kpi-icon">⚠️</div>
  </div>
  <div class="kpi-card" style="--kpi-accent:var(--green)">
    <div class="kpi-label">Passed (OK)</div>
    <div class="kpi-val text-green">{ok_cnt}</div>
    <div class="kpi-sub">No defects · {pass_rt}% pass rate</div>
    <div class="kpi-icon">✅</div>
  </div>
  <div class="kpi-card" style="--kpi-accent:var(--orange)">
    <div class="kpi-label">Avg Confidence</div>
    <div class="kpi-val text-orange">{avg_conf}%</div>
    <div class="kpi-sub">AI detection confidence</div>
    <div class="kpi-icon">📊</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="medium")

    # Defect breakdown
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Defect Breakdown</div>', unsafe_allow_html=True)
        if defect_cnt:
            for cls_name, count in defect_cnt.most_common():
                color = class_color(cls_name)
                emoji = class_emoji(cls_name)
                pct   = round(count / sum(defect_cnt.values()) * 100)
                st.markdown(
                    f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="font-size:13px;font-weight:600;color:{color}">{emoji} {cls_name}</span>
    <span class="text-mono" style="font-size:12px;color:var(--text2)">{count} · {pct}%</span>
  </div>
  <div class="prog-wrap">
    <div class="prog-fill" style="width:{pct}%;background:{color}"></div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">📊</div>'
                '<div class="empty-text">No defects recorded yet</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # System status
    with col_b:
        model_status = ("🟢", "Online", "var(--green)") if model_ok else ("🔴", "Not Loaded", "var(--red)")
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">🖥 System Status</div>
  <div class="sys-row">
    <span>AI Engine (YOLOv11)</span>
    <span class="text-mono" style="color:{model_status[2]};font-size:12px">{model_status[0]} {model_status[1]}</span>
  </div>
  <div class="sys-row">
    <span>Model File</span>
    <span class="text-mono" style="font-size:11px;color:var(--text2)">best.pt</span>
  </div>
  <div class="sys-row">
    <span>Detection Classes</span>
    <span class="text-mono" style="font-size:11px;color:var(--blue)">{len(class_names)} classes</span>
  </div>
  <div class="sys-row">
    <span>Confidence Threshold</span>
    <span class="text-mono" style="font-size:12px;color:var(--orange)">{int(st.session_state.conf_threshold*100)}%</span>
  </div>
  <div class="sys-row" style="border:none">
    <span>Alert System</span>
    <span class="text-mono" style="font-size:12px;color:var(--orange)">
      {"🟡 Armed" if st.session_state.alert_enabled else "⚫ Disabled"}
    </span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Recent inspections
    st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">'
        '<div class="card-title" style="margin-bottom:0">📋 Recent Inspections</div></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.history:
        rows_html = ""
        for h in st.session_state.history[:8]:
            s_color = "var(--red)" if h["status"] == "FAULT" else "var(--green)"
            s_badge = (
                '<span class="badge badge-fault">FAULT</span>'
                if h["status"] == "FAULT"
                else '<span class="badge badge-ok">OK</span>'
            )
            det_names = ", ".join(d["class"] for d in h["detections"][:3]) or "—"
            rows_html += f"""
<div class="hist-row">
  <span style="font-family:var(--mono);font-size:11px;color:var(--text3);min-width:140px">{h['ts']}</span>
  <span class="badge badge-idle">{h['source']}</span>
  {s_badge}
  <span style="font-size:12px;color:var(--text2);flex:1;padding-left:8px">{det_names}</span>
</div>
"""
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">📋</div>'
            '<div class="empty-text">No inspections yet</div>'
            '<div class="empty-sub">Start with Live Camera or Upload Image</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — LIVE CAMERA
# ═══════════════════════════════════════════════════════════════
with tab_cam:
    col_cam, col_panel = st.columns([3, 1], gap="medium")

    with col_panel:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚙️ Camera Controls</div>', unsafe_allow_html=True)
        cam_conf = st.slider(
            "Confidence Threshold", 5, 95,
            int(st.session_state.conf_threshold * 100), 5,
            format="%d%%", key="cam_conf_slider",
        )
        st.session_state.conf_threshold = cam_conf / 100

        continuous = st.checkbox("🔄 Continuous Detection", value=st.session_state.cam_continuous, key="cam_cont_chk")
        st.session_state.cam_continuous = continuous
        st.markdown("</div>", unsafe_allow_html=True)

        # Live detections panel (updated after inference)
        det_placeholder = st.empty()

        # Session stats
        st.markdown('<div class="card mt-16">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Session Stats</div>', unsafe_allow_html=True)
        cam_scans  = sum(1 for h in st.session_state.history if h["source"] == "Camera")
        cam_faults = sum(1 for h in st.session_state.history if h["source"] == "Camera" and h["status"] == "FAULT")
        cam_pr     = round((cam_scans - cam_faults) / cam_scans * 100) if cam_scans > 0 else 100
        st.markdown(
            f"""
<div class="card-row"><span class="text-muted">Scanned</span><span class="text-mono">{cam_scans}</span></div>
<div class="card-row"><span class="text-muted">Faults</span><span class="text-mono text-red">{cam_faults}</span></div>
<div class="card-row" style="border:none"><span class="text-muted">Pass Rate</span>
  <span class="text-mono text-green">{cam_pr}%</span></div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cam:
        st.markdown(
            '<div style="font-size:22px;font-weight:800;margin-bottom:4px">📷 Live Camera</div>'
            '<div style="font-size:13px;color:var(--text2);margin-bottom:16px">Real-time defect detection via webcam</div>',
            unsafe_allow_html=True,
        )

        cam_frame = st.camera_input(
            "Camera Feed — capture a frame for instant detection",
            key="cam_input",
            label_visibility="collapsed",
        )

        result_placeholder = st.empty()

        if cam_frame is not None:
            img_pil = Image.open(cam_frame).convert("RGB")
            with st.spinner("🔍 Analyzing frame…"):
                annotated, detections = run_inference(img_pil, conf=st.session_state.conf_threshold)

            result_placeholder.image(annotated, width='stretch', caption="Detection Result")

            has_fault = any(d["fault"] for d in detections)

            # Alert banner
            if has_fault and st.session_state.alert_enabled:
                fault_names = ", ".join(d["class"] for d in detections if d["fault"])
                st.markdown(
                    f"""<div class="alert-box">
  <div class="alert-icon">🚨</div>
  <div>
    <div style="font-weight:800;font-size:16px;color:var(--red)">FAULT DETECTED</div>
    <div style="font-size:13px;color:var(--text2);margin-top:2px">{fault_names}</div>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:var(--green-g);border:1px solid rgba(63,185,80,.3);'
                    'border-radius:var(--r);padding:12px 16px;color:var(--green);'
                    'font-weight:700;font-size:14px;margin-bottom:8px">✅ No defects detected — Pipeline OK</div>',
                    unsafe_allow_html=True,
                )

            # Detection list in panel
            if detections:
                dets_html = '<div class="card"><div class="card-title">🎯 Live Detections</div>'
                for d in detections:
                    pct = int(d["confidence"])
                    dets_html += f"""
<div class="det-item">
  <div class="det-item-header">
    <span class="det-name" style="color:{d['color']}">{d['emoji']} {d['class']}</span>
    <span class="det-conf">{pct}%</span>
  </div>
  <div class="conf-bar">
    <div class="conf-fill" style="width:{pct}%;background:{d['color']}"></div>
  </div>
</div>
"""
                dets_html += "</div>"
                det_placeholder.markdown(dets_html, unsafe_allow_html=True)
            else:
                det_placeholder.markdown(
                    '<div class="card"><div class="card-title">🎯 Live Detections</div>'
                    '<div style="font-size:12px;color:var(--text3);padding:8px 0">No objects detected</div></div>',
                    unsafe_allow_html=True,
                )

            save_to_history("Camera", img_pil, detections)

            if continuous:
                time.sleep(0.5)
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 3 — UPLOAD IMAGE
# ═══════════════════════════════════════════════════════════════
with tab_img:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;margin-bottom:4px">🖼 Upload Image</div>'
        '<div style="font-size:13px;color:var(--text2);margin-bottom:20px">Analyze a single image for pipeline defects</div>',
        unsafe_allow_html=True,
    )
    col_upload, col_result = st.columns(2, gap="medium")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 Upload</div>', unsafe_allow_html=True)
        img_conf = st.slider(
            "Confidence Threshold", 5, 95,
            int(st.session_state.conf_threshold * 100), 5,
            format="%d%%", key="img_conf_slider",
        )
        uploaded = st.file_uploader(
            "Drop image here · JPG, PNG, WEBP",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, caption="Preview", width='stretch')
        analyze_btn = st.button(
            "🔍  Analyze Image",
            type="primary",
            disabled=(uploaded is None),
            width='stretch',
            key="img_analyze_btn",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">👁 Detection Result</div>', unsafe_allow_html=True)

        if analyze_btn and uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            with st.spinner("🔍 Running YOLOv11 inference…"):
                annotated, detections = run_inference(img_pil, conf=img_conf / 100)

            st.image(annotated, width='stretch')

            has_fault = any(d["fault"] for d in detections)
            status_badge = (
                '<span class="badge badge-fault">⚠️ FAULT</span>'
                if has_fault
                else '<span class="badge badge-ok">✅ OK</span>'
            )
            st.markdown(
                f'<div style="margin:12px 0 8px;display:flex;align-items:center;gap:8px">'
                f'{status_badge}<span style="font-size:13px;color:var(--text2)">· {len(detections)} detection(s)</span></div>',
                unsafe_allow_html=True,
            )

            if detections:
                for d in detections:
                    pct = int(d["confidence"])
                    st.markdown(
                        f"""
<div class="det-item">
  <div class="det-item-header">
    <span class="det-name" style="color:{d['color']}">{d['emoji']} {d['class']}</span>
    <span class="det-conf">{pct}%</span>
  </div>
  <div class="conf-bar"><div class="conf-fill" style="width:{pct}%;background:{d['color']}"></div></div>
  <div style="font-size:11px;color:var(--text3);margin-top:4px;font-family:var(--mono)">
    bbox [{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="empty-state"><div class="empty-icon">🔍</div>'
                    '<div class="empty-text">No objects detected</div>'
                    '<div class="empty-sub">Try lowering the confidence threshold</div></div>',
                    unsafe_allow_html=True,
                )

            save_to_history("Image", img_pil, detections)

            # Download annotated
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button(
                "⬇️  Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"lv_result_{now_th().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                width='stretch',
            )
        else:
            st.markdown(
                '<div class="empty-state" style="padding:60px 20px">'
                '<div class="empty-icon">📷</div>'
                '<div class="empty-text">Upload an image to begin</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — UPLOAD VIDEO
# ═══════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;margin-bottom:4px">🎬 Upload Video</div>'
        '<div style="font-size:13px;color:var(--text2);margin-bottom:20px">Process video file frame-by-frame for defect detection</div>',
        unsafe_allow_html=True,
    )
    col_vup, col_vres = st.columns(2, gap="medium")

    with col_vup:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 Upload Video</div>', unsafe_allow_html=True)
        vid_conf     = st.slider("Confidence", 10, 95, 50, 5, format="%d%%", key="vid_conf")
        max_frames   = st.select_slider(
            "Frames to Analyze",
            options=[5, 10, 15, 20, 30, 50],
            value=10, key="vid_frames",
        )
        vid_file = st.file_uploader(
            "Drop video here · MP4, MOV, AVI, WEBM",
            type=["mp4", "mov", "avi", "webm", "mkv"],
            key="vid_uploader",
            label_visibility="collapsed",
        )
        process_btn = st.button(
            "▶  Process Video",
            type="primary",
            disabled=(vid_file is None),
            width='stretch',
            key="vid_process_btn",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_vres:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Video Analysis</div>', unsafe_allow_html=True)

        if process_btn and vid_file:
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            duration   = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            progress_bar  = st.progress(0.0, text="Extracting frames…")
            frame_results = []
            all_vid_dets  = []

            cap = cv2.VideoCapture(tmp_path)
            intervals = np.linspace(0, max(total_frames - 1, 1), num=max_frames, dtype=int)

            for i, frame_idx in enumerate(intervals):
                pct = (i + 1) / max_frames
                progress_bar.progress(pct, text=f"Analyzing frame {i+1}/{max_frames}…")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame_bgr = cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil   = Image.fromarray(frame_rgb)
                _, dets   = run_inference(img_pil, conf=vid_conf / 100)
                frame_results.append(
                    {
                        "frame":       i + 1,
                        "frame_idx":   int(frame_idx),
                        "time_sec":    round(frame_idx / max(cap.get(cv2.CAP_PROP_FPS), 1), 1),
                        "detections":  dets,
                        "has_fault":   any(d["fault"] for d in dets),
                    }
                )
                all_vid_dets.extend(dets)

            cap.release()
            os.unlink(tmp_path)
            progress_bar.progress(1.0, text="✅ Complete!")
            time.sleep(0.8)
            progress_bar.empty()

            # Summary
            fault_frames = [r for r in frame_results if r["has_fault"]]
            ok_frames    = [r for r in frame_results if not r["has_fault"]]
            vid_defects  = Counter(d["class"] for d in all_vid_dets if d["fault"])
            pass_rate_v  = round(len(ok_frames) / len(frame_results) * 100) if frame_results else 100

            st.markdown(
                f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
  <div class="card-row"><span class="text-muted">Frames Analyzed</span>
    <span class="text-mono">{len(frame_results)}</span></div>
  <div class="card-row"><span class="text-muted">Fault Frames</span>
    <span class="text-mono text-red">{len(fault_frames)}</span></div>
  <div class="card-row"><span class="text-muted">OK Frames</span>
    <span class="text-mono text-green">{len(ok_frames)}</span></div>
  <div class="card-row"><span class="text-muted">Pass Rate</span>
    <span class="text-mono text-green">{pass_rate_v}%</span></div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Defect breakdown
            if vid_defects:
                st.markdown("**Defect Breakdown**")
                for cls_n, cnt in vid_defects.most_common():
                    col = class_color(cls_n)
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">'
                        f'<span style="color:{col}">{class_emoji(cls_n)} {cls_n}</span>'
                        f'<span class="text-mono">{cnt}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:var(--green);font-weight:700;margin-bottom:12px">✅ No defects found in video</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<hr style='border:1px solid var(--border);margin:12px 0'>", unsafe_allow_html=True)

            # Frame timeline
            st.markdown("**Frame Timeline**")
            chips = ""
            for r in frame_results:
                c = "var(--red)" if r["has_fault"] else "var(--green)"
                label = "⚠️" if r["has_fault"] else "✅"
                chips += (
                    f'<div style="background:var(--bg3);border:1px solid {c};border-radius:var(--rs);'
                    f'padding:6px 10px;text-align:center;min-width:60px;display:inline-block;margin:3px">'
                    f'<div style="font-size:10px;color:var(--text2)">F{r["frame"]}</div>'
                    f'<div style="font-size:12px;color:{c}">{label}</div>'
                    f'<div style="font-size:10px;color:var(--text3)">{r["time_sec"]}s</div></div>'
                )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:4px">{chips}</div>',
                unsafe_allow_html=True,
            )

            save_to_history("Video", None, all_vid_dets)

        else:
            st.markdown(
                '<div class="empty-state" style="padding:60px 20px">'
                '<div class="empty-icon">🎬</div>'
                '<div class="empty-text">Upload a video to begin</div>'
                '<div class="empty-sub">Frames will be analyzed across the full duration</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — HISTORY
# ═══════════════════════════════════════════════════════════════
with tab_hist:
    col_ht, col_hb = st.columns([3, 1])
    with col_ht:
        st.markdown(
            '<div style="font-size:22px;font-weight:800;margin-bottom:4px">📋 Inspection History</div>'
            '<div style="font-size:13px;color:var(--text2);margin-bottom:20px">All past inspections in this session</div>',
            unsafe_allow_html=True,
        )
    with col_hb:
        if st.button("🗑  Clear History", key="clear_hist", width='stretch'):
            st.session_state.history      = []
            st.session_state.total_scans  = 0
            st.session_state.total_faults = 0
            st.rerun()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.session_state.history:
        # ── HTML table (dark-theme safe) ──
        rows_html = """
<style>
.hist-table { width:100%; border-collapse:collapse; font-size:13px; }
.hist-table th {
  font-size:11px; font-weight:700; color:var(--text2);
  letter-spacing:.5px; text-transform:uppercase;
  padding:10px 14px; border-bottom:1px solid var(--border);
  text-align:left; background:var(--bg2);
}
.hist-table td { padding:10px 14px; color:var(--text2); border-bottom:1px solid var(--border); }
.hist-table td:first-child { color:var(--text); font-family:var(--mono); font-size:12px; }
.hist-table tr:hover td { background:var(--bg3); }
.hist-table tr:last-child td { border-bottom:none; }
.st-fault { color:#f85149; font-weight:700; }
.st-ok    { color:#3fb950; font-weight:700; }
</style>
<table class="hist-table">
<thead><tr>
  <th>Timestamp</th><th>Source</th><th>Status</th>
  <th>Detections</th><th>Avg Conf</th>
</tr></thead><tbody>
"""
        csv_rows = []
        for h in st.session_state.history:
            det_str  = ", ".join(d["class"] for d in h["detections"][:3]) or "—"
            conf_avg = (
                round(sum(d["confidence"] for d in h["detections"]) / len(h["detections"]), 1)
                if h["detections"] else 0
            )
            s_class = "st-fault" if h["status"] == "FAULT" else "st-ok"
            rows_html += (
                f"<tr>"
                f"<td>{h['ts']}</td>"
                f"<td><span class='badge badge-idle'>{h['source']}</span></td>"
                f"<td><span class='{s_class}'>{h['status']}</span></td>"
                f"<td>{det_str}</td>"
                f"<td style='font-family:var(--mono)'>{conf_avg}%</td>"
                f"</tr>"
            )
            csv_rows.append({
                "Timestamp":  h["ts"],
                "Source":     h["source"],
                "Status":     h["status"],
                "Detections": det_str,
                "Avg Conf %": conf_avg,
            })
        rows_html += "</tbody></table>"
        st.markdown(rows_html, unsafe_allow_html=True)

        # Export CSV
        csv_buf = pd.DataFrame(csv_rows).to_csv(index=False).encode()
        st.download_button(
            "⬇️  Export CSV",
            data=csv_buf,
            file_name=f"lv_history_{now_th().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="hist_export",
        )
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">📋</div>'
            '<div class="empty-text">No inspection history yet</div>'
            '<div class="empty-sub">Results will appear here after inspections</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 6 — SETTINGS
# ═══════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;margin-bottom:4px">⚙️ Settings</div>'
        '<div style="font-size:13px;color:var(--text2);margin-bottom:20px">Configure AI detection parameters and preferences</div>',
        unsafe_allow_html=True,
    )
    col_s1, col_s2 = st.columns(2, gap="medium")

    with col_s1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🤖 AI Configuration</div>', unsafe_allow_html=True)
        new_conf = st.slider(
            "Global Confidence Threshold",
            5, 95,
            int(st.session_state.conf_threshold * 100), 5,
            format="%d%%",
            help="Minimum confidence for a detection to be reported",
            key="global_conf_setting",
        )
        if new_conf != int(st.session_state.conf_threshold * 100):
            st.session_state.conf_threshold = new_conf / 100

        st.markdown("<br>**Active Detection Classes**", unsafe_allow_html=True)
        for cn in class_names:
            col = class_color(cn)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{col}"></div>'
                f'<span style="font-size:13px;color:var(--text)">{class_emoji(cn)} {cn}</span></div>',
                unsafe_allow_html=True,
            )
        if not class_names:
            st.markdown(
                '<div style="font-size:12px;color:var(--text3)">No class names loaded — check best.pt</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card mt-16">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧠 Ensemble Models (5 ตัว)</div>', unsafe_allow_html=True)
        _model_rows = ""
        for cls_name in PIPELINE_MODELS.keys():
            loaded = cls_name in _ensemble
            dot_color = "var(--green)" if loaded else "var(--red)"
            path_found = _find_model_file(PIPELINE_MODELS[cls_name])
            status = "โหลดแล้ว" if loaded else (f"พบที่ {os.path.basename(path_found)} แต่โหลดไม่ได้" if path_found else "ไม่พบไฟล์")
            col       = class_color(cls_name)
            emoji     = class_emoji(cls_name)
            _model_rows += (
                f'<div class="card-row">' 
                f'<span style="color:{col}">{emoji} {cls_name}</span>' 
                f'<span class="text-mono" style="color:{dot_color};font-size:12px">● {status}</span></div>'
            )
        st.markdown(_model_rows, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card mt-16">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ℹ️ Model Info</div>', unsafe_allow_html=True)
        active_label = f"{len(_ensemble)}/6 models loaded"
        st.markdown(
            f'''
<div class="card-row"><span class="text-muted">Models โหลดแล้ว</span><span class="text-mono text-blue">{len(_ensemble)}/6</span></div>
<div class="card-row"><span class="text-muted">Classes</span><span class="text-mono">{len(class_names)}</span></div>
<div class="card-row" style="border:none"><span class="text-muted">Framework</span><span class="text-mono">YOLOv11 · Ultralytics</span></div>
''',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_s2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔔 Alerts & Behaviour</div>', unsafe_allow_html=True)

        alert_en = st.toggle(
            "Fault Alert Banner",
            value=st.session_state.alert_enabled,
            help="Show alert banner when defects are detected",
            key="setting_alert",
        )
        st.session_state.alert_enabled = alert_en

        autosave = st.toggle(
            "Auto-save to History",
            value=st.session_state.autosave,
            help="Automatically save each inspection to history",
            key="setting_autosave",
        )
        st.session_state.autosave = autosave

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card mt-16">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗄️ Data Management</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🗑  Clear All History", width='stretch', key="clear_hist_settings"):
                st.session_state.history      = []
                st.session_state.total_scans  = 0
                st.session_state.total_faults = 0
                st.success("History cleared!")
        with col_r2:
            if st.button("🔄  Reset Counters", width='stretch', key="reset_counters"):
                st.session_state.total_scans  = 0
                st.session_state.total_faults = 0
                st.success("Counters reset!")

        st.markdown(
            f"""
<div style="margin-top:16px">
<div class="card-row"><span class="text-muted">Total Records</span><span class="text-mono">{len(st.session_state.history)}</span></div>
<div class="card-row" style="border:none"><span class="text-muted">Session Memory</span>
  <span class="text-mono text-blue">In-Memory Only</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card mt-16">', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card-title">👁 About</div>
<div class="card-row"><span class="text-muted">App</span><span class="text-mono">Littlelome AI Vision</span></div>
<div class="card-row"><span class="text-muted">Version</span><span class="text-mono">1.0.0</span></div>
<div class="card-row"><span class="text-muted">Engine</span><span class="text-mono">YOLOv11 · Ultralytics</span></div>
<div class="card-row" style="border:none"><span class="text-muted">Framework</span><span class="text-mono">Streamlit</span></div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
