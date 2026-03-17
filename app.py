"""
CamSecure — 3D Modern Gate Security Dashboard
===============================================
Files needed in same folder:
  camsecure_logo.png  — your logo
  camsecure_cam.jpg   — camera photo
  step3_run.py        — detection script

Run : python -m streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import subprocess, sys, time, base64
from pathlib import Path
from PIL import Image
import datetime, io

st.set_page_config(
    page_title="CamSecure · Your Security Matters",
    page_icon="🔒", layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCREENSHOTS_DIR = Path("screenshots")
LOG_FILE        = Path("detection_log.txt")
RUN_SCRIPT      = Path("run.py")


# ── Load + shrink images → tiny base64 (must stay <35KB each) ─────────────────
def _b64(path: Path, fmt: str, max_px: int) -> str:
    if not path.exists():
        return ""
    try:
        img = Image.open(path).convert("RGBA" if fmt == "PNG" else "RGB")
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        kw = {"quality": 65} if fmt == "JPEG" else {"optimize": True}
        img.save(buf, format=fmt, **kw)
        b = base64.b64encode(buf.getvalue()).decode()
        return b
    except Exception:
        return ""


# Keep each image tiny so no single st.markdown call exceeds ~40 KB
LOGO_B64 = _b64(Path("CamSecure.png"), "PNG",  200)  # ~32 KB
CAM_B64  = _b64(Path("CamSecure_logo.jpg"),  "JPEG", 240)  # ~7 KB

LOGO_SRC = f"data:image/png;base64,{LOGO_B64}"   if LOGO_B64 else ""
CAM_SRC  = f"data:image/jpeg;base64,{CAM_B64}"   if CAM_B64  else ""


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "process": None, "cam_running": False,
    "start_time": None, "delete_confirm": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_screenshots():
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    return sorted(SCREENSHOTS_DIR.glob("ALERT_*.jpg"),
                  key=lambda f: f.stat().st_mtime, reverse=True)

def count_log_events():
    s, u = 0, 0
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text(errors="ignore").splitlines():
            if "STUDENT" in line:   s += 1
            elif "UNKNOWN" in line: u += 1
    return s, u

def parse_recent_alerts(n=6):
    alerts = []
    if LOG_FILE.exists():
        for line in reversed(LOG_FILE.read_text(errors="ignore").splitlines()):
            if "UNKNOWN" in line: alerts.append(line)
            if len(alerts) >= n:  break
    return alerts

def is_running():
    if st.session_state.process is None: return False
    return st.session_state.process.poll() is None

def uptime_str():
    if not st.session_state.start_time: return "00:00:00"
    d = datetime.datetime.now() - st.session_state.start_time
    return (f"{int(d.total_seconds()//3600):02d}:"
            f"{int((d.total_seconds()%3600)//60):02d}:"
            f"{int(d.total_seconds()%60):02d}")

def screenshot_to_b64(path):
    try:
        img = Image.open(path); img.thumbnail((700, 500))
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception: return None

def fmt_ts(stem):
    try:
        return datetime.datetime.strptime(
            stem.replace("ALERT_", ""), "%Y%m%d_%H%M%S"
        ).strftime("%d %b %Y  %H:%M:%S")
    except Exception: return stem


# ══════════════════════════════════════════════════════════════════════════════
# PURE CSS — no images embedded here, just styles
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=Barlow:wght@300;400;500;600;700;800&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
    background:#060910!important;
    color:#e0e8ff!important;
    font-family:'Barlow',sans-serif!important;
    scroll-behavior:smooth;
}
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse at 15% 0%,rgba(15,30,90,0.7) 0%,transparent 50%),
        radial-gradient(ellipse at 85% 100%,rgba(8,18,60,0.6) 0%,transparent 50%),
        #060910!important;
}
[data-testid="stAppViewContainer"]::before{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background:repeating-linear-gradient(
        -45deg,transparent,transparent 60px,
        rgba(255,204,0,0.008) 60px,rgba(255,204,0,0.008) 61px
    );
}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stToolbar"],[data-testid="stDecoration"],.stDeployButton{display:none}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:#030608}
::-webkit-scrollbar-thumb{background:linear-gradient(#FFCC00,#FF9900);border-radius:2px}

/* ── NAV ───────────────────── */
.cs-nav{
    display:flex;align-items:center;justify-content:space-between;
    padding:0 36px;height:64px;
    background:rgba(4,8,22,0.97);
    border-bottom:1px solid rgba(255,204,0,0.15);
    backdrop-filter:blur(24px);
    box-shadow:0 4px 40px rgba(0,0,0,0.7);
    position:sticky;top:0;z-index:999;
    margin:-1rem -1rem 0 -1rem;
    animation:nav-drop .6s ease both;
}
@keyframes nav-drop{from{opacity:0;transform:translateY(-18px)}to{opacity:1;transform:translateY(0)}}

.cs-nav-logo img{
    height:46px;width:auto;object-fit:contain;
    filter:drop-shadow(0 0 10px rgba(255,204,0,0.4));
    transition:filter .35s,transform .35s;
    animation:logo-pulse 3.5s ease-in-out infinite;
}
@keyframes logo-pulse{
    0%,100%{filter:drop-shadow(0 0 8px rgba(255,204,0,.3))}
    50%{filter:drop-shadow(0 0 22px rgba(255,204,0,.7)) brightness(1.1)}
}
.cs-nav-logo img:hover{filter:drop-shadow(0 0 28px rgba(255,204,0,.9)) brightness(1.12);transform:scale(1.05)}

.cs-nav-center{
    font-family:'Orbitron',monospace;font-size:10px;font-weight:400;
    letter-spacing:.4em;text-transform:uppercase;color:rgba(255,204,0,.3);
}
.cs-nav-right{display:flex;align-items:center;gap:14px}
.cs-nav-clock{font-family:'Share Tech Mono',monospace;font-size:11px;color:rgba(180,200,240,.38);letter-spacing:.06em}

.cs-pill{display:flex;align-items:center;gap:7px;padding:7px 16px;border-radius:3px;font-family:'Share Tech Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.1em;border:1px solid;transition:all .3s}
.cs-pill.live{background:rgba(0,180,80,.1);border-color:rgba(0,220,100,.3);color:#00dc64}
.cs-pill.offline{background:rgba(255,204,0,.05);border-color:rgba(255,204,0,.18);color:rgba(255,204,0,.45)}
.cs-pill.alert{background:rgba(220,30,60,.12);border-color:rgba(255,60,80,.35);color:#ff5070;animation:pill-flash .7s infinite}
@keyframes pill-flash{0%,100%{background:rgba(220,30,60,.1)}50%{background:rgba(220,30,60,.28)}}

.cs-nav-badge{display:flex;align-items:center;gap:6px;padding:7px 16px;border-radius:3px;background:rgba(255,204,0,.07);border:1px solid rgba(255,204,0,.22);color:#FFCC00;font-family:'Share Tech Mono',monospace;font-size:11px;text-decoration:none;transition:all .25s;letter-spacing:.08em}
.cs-nav-badge:hover{background:rgba(255,204,0,.16);border-color:rgba(255,204,0,.5);transform:translateY(-2px);box-shadow:0 4px 16px rgba(255,204,0,.2)}

/* ── HERO ───────────────────── */
.cs-hero{
    position:relative;min-height:86vh;
    display:flex;flex-direction:column;justify-content:center;
    overflow:hidden;border-bottom:1px solid rgba(255,204,0,.12);
    margin:0 -1rem;
}
.cs-hero-bg{
    position:absolute;inset:0;z-index:0;
    background:linear-gradient(135deg,#050c1e 0%,#08142e 50%,#050c1e 100%);
}
.cs-hero-overlay{
    position:absolute;inset:0;z-index:1;
    background:
        radial-gradient(ellipse at 25% 50%,rgba(20,50,140,.5) 0%,transparent 55%),
        radial-gradient(ellipse at 75% 30%,rgba(8,20,80,.4) 0%,transparent 45%),
        linear-gradient(to bottom,transparent 60%,#060910 100%);
}
.cs-hero-grid{
    position:absolute;inset:0;z-index:1;pointer-events:none;
    background-image:
        linear-gradient(rgba(255,204,0,.03) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,204,0,.03) 1px,transparent 1px);
    background-size:65px 65px;
    animation:grid-drift 22s linear infinite;
}
@keyframes grid-drift{from{background-position:0 0}to{background-position:65px 65px}}
.cs-hero-scan{
    position:absolute;inset:0;z-index:2;pointer-events:none;
    background:linear-gradient(to bottom,transparent,rgba(255,204,0,.022) 50%,transparent);
    background-size:100% 200%;
    animation:scan-sweep 5s linear infinite;
}
@keyframes scan-sweep{from{background-position:0 -100%}to{background-position:0 200%}}

.cs-hero-content{position:relative;z-index:3;padding:70px 60px;max-width:720px}

.cs-eyebrow{
    display:inline-flex;align-items:center;gap:8px;
    font-family:'Share Tech Mono',monospace;font-size:11px;
    letter-spacing:.25em;text-transform:uppercase;
    color:rgba(255,204,0,.65);margin-bottom:20px;
    background:rgba(255,204,0,.04);border:1px solid rgba(255,204,0,.13);
    padding:6px 16px;border-radius:2px;
    animation:fade-right .9s ease both .2s;
}
@keyframes fade-right{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}
.cs-dot{width:7px;height:7px;background:#FFCC00;border-radius:50%;animation:blink-dot 1.2s ease-in-out infinite;box-shadow:0 0 8px rgba(255,204,0,.6)}
@keyframes blink-dot{0%,100%{opacity:1}50%{opacity:.2}}

.cs-hero-logo-wrap{margin-bottom:20px;animation:logo-rise 1s cubic-bezier(.34,1.56,.64,1) both .15s}
@keyframes logo-rise{from{opacity:0;transform:translateY(22px) scale(.88)}to{opacity:1;transform:translateY(0) scale(1)}}
.cs-hero-logo-wrap img{height:62px;width:auto;filter:drop-shadow(0 0 18px rgba(255,204,0,.45))}

.cs-tagline{
    font-family:'Orbitron',monospace;font-size:11px;font-weight:400;
    letter-spacing:.45em;text-transform:uppercase;
    color:rgba(255,204,0,.45);margin-bottom:20px;
    animation:fade-up .8s ease both .45s;
}
@keyframes fade-up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}

.cs-hero-title{
    font-family:'Barlow',sans-serif;font-size:clamp(38px,5.5vw,68px);
    font-weight:800;line-height:1.06;color:#fff;margin-bottom:18px;
    animation:fade-up .9s ease both .38s;
}
.cs-hero-title-gold{color:#FFCC00;animation:gold-glow 3.5s ease-in-out infinite}
@keyframes gold-glow{0%,100%{text-shadow:0 0 28px rgba(255,204,0,.25)}50%{text-shadow:0 0 55px rgba(255,204,0,.6)}}

.cs-hero-sub{
    font-size:16px;font-weight:300;color:rgba(185,210,255,.52);
    max-width:500px;line-height:1.75;margin-bottom:40px;
    animation:fade-up .9s ease both .52s;
}
.cs-hero-stats{display:flex;gap:0;animation:fade-up .9s ease both .65s}
.cs-stat-item{padding:0 28px 0 0;margin-right:28px;border-right:1px solid rgba(255,204,0,.12)}
.cs-stat-item:last-child{border-right:none}
.cs-stat-num{font-family:'Orbitron',monospace;font-size:28px;font-weight:700;color:#FFCC00;line-height:1;margin-bottom:4px}
.cs-stat-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:rgba(180,200,240,.3)}

/* floating cam card — image-free, uses icon */
.cs-cam-card{
    position:absolute;right:70px;top:50%;transform:translateY(-50%);
    z-index:3;width:290px;
    background:linear-gradient(145deg,rgba(10,18,52,.94),rgba(6,12,36,.97));
    border:1px solid rgba(255,204,0,.22);border-radius:16px;overflow:hidden;
    box-shadow:0 28px 80px rgba(0,0,0,.65),0 0 0 1px rgba(255,204,0,.07),
               inset 0 1px 0 rgba(255,204,0,.1);
    animation:card-float 6s ease-in-out infinite,card-in 1s ease both .65s;
    backdrop-filter:blur(20px);
}
@keyframes card-float{0%,100%{transform:translateY(-50%) translateY(0)}50%{transform:translateY(-50%) translateY(-14px)}}
@keyframes card-in{from{opacity:0;transform:translateY(-50%) translateX(40px)}to{opacity:1;transform:translateY(-50%) translateX(0)}}

.cs-cam-preview{
    width:100%;height:160px;display:flex;align-items:center;justify-content:center;
    background:linear-gradient(160deg,#060e28,#0a1640);
    border-bottom:1px solid rgba(255,204,0,.1);
    position:relative;overflow:hidden;
}
.cs-cam-preview img{
    width:100%;height:100%;object-fit:cover;
    filter:brightness(.75) saturate(.55);
}
.cs-cam-preview-icon{
    font-size:42px;opacity:.18;
    position:absolute;
}
.cs-cam-scan{
    position:absolute;inset:0;
    background:linear-gradient(to bottom,transparent,rgba(255,204,0,.025) 50%,transparent);
    background-size:100% 200%;
    animation:scan-sweep 3s linear infinite;
}
.cs-cam-corner{
    position:absolute;top:8px;left:8px;
    font-family:'Share Tech Mono',monospace;font-size:9px;
    color:rgba(255,204,0,.5);letter-spacing:.1em;
    background:rgba(0,0,0,.4);padding:3px 7px;border-radius:2px;
    border:1px solid rgba(255,204,0,.15);
}
.cs-cam-body{padding:14px 18px}
.cs-cam-title{font-family:'Orbitron',monospace;font-size:11px;font-weight:600;color:#FFCC00;letter-spacing:.1em;margin-bottom:5px}
.cs-cam-desc{font-size:11px;color:rgba(180,200,240,.42);line-height:1.5}
.cs-cam-status{display:flex;align-items:center;gap:8px;margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,204,0,.09)}
.cs-cam-dot{width:8px;height:8px;border-radius:50%;animation:blink-dot 1.1s ease-in-out infinite}
.cs-cam-txt{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.1em}

/* ── STAT CARDS ──────────────── */
.cs-stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}
.cs-card{
    background:linear-gradient(145deg,#0c1538,#0e1a46);
    border:1px solid rgba(255,204,0,.1);border-radius:14px;
    padding:26px 24px;position:relative;overflow:hidden;
    transition:all .4s cubic-bezier(.34,1.56,.64,1);
    animation:card-rise .65s ease both;
}
@keyframes card-rise{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
.cs-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,204,0,.65),transparent);opacity:0;transition:opacity .3s}
.cs-card:hover{border-color:rgba(255,204,0,.35);transform:translateY(-8px);box-shadow:0 22px 55px rgba(0,0,0,.6),0 0 0 1px rgba(255,204,0,.1)}
.cs-card:hover::before{opacity:1}
.cs-card.link{cursor:pointer}
.cs-card-corner{position:absolute;top:0;right:0;width:0;height:0;border-style:solid;border-width:0 48px 48px 0;border-color:transparent rgba(255,204,0,.06) transparent transparent}
.cs-card-icon{font-size:28px;margin-bottom:12px;display:block}
.cs-card-val{font-family:'Orbitron',monospace;font-size:40px;font-weight:700;line-height:1;margin-bottom:6px}
.cs-card-val.gold {color:#FFCC00;text-shadow:0 0 28px rgba(255,204,0,.35)}
.cs-card-val.red  {color:#ff5070;text-shadow:0 0 28px rgba(255,80,112,.35)}
.cs-card-val.white{color:#e8eeff}
.cs-card-val.cyan {color:#60c0ff;font-size:26px;padding-top:8px}
.cs-card-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:rgba(180,200,240,.3);margin-bottom:14px}
.cs-bar{height:2px;background:rgba(255,255,255,.05);border-radius:1px;overflow:hidden}
.cs-bar-fill{height:100%;border-radius:1px;animation:bar-in 1.6s ease both}
@keyframes bar-in{from{width:0!important}}
.bg{background:linear-gradient(90deg,rgba(255,204,0,.3),#FFCC00)}
.br{background:linear-gradient(90deg,rgba(255,80,112,.3),#ff5070)}
.bw{background:linear-gradient(90deg,rgba(180,210,255,.3),#c0d8ff)}
.bc{background:linear-gradient(90deg,rgba(60,180,255,.3),#60c0ff)}

/* ── FEATURES ────────────────── */
.cs-features{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:12px 0 32px}
.cs-feat{
    background:linear-gradient(145deg,rgba(10,18,50,.7),rgba(6,12,34,.8));
    border:1px solid rgba(255,204,0,.08);border-radius:14px;
    padding:26px 22px;position:relative;overflow:hidden;
    transition:all .4s cubic-bezier(.34,1.56,.64,1);
    animation:card-rise .7s ease both;
}
.cs-feat::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,204,0,.45),transparent);opacity:0;transition:opacity .3s}
.cs-feat:hover{border-color:rgba(255,204,0,.24);transform:translateY(-6px);box-shadow:0 18px 50px rgba(0,0,0,.5)}
.cs-feat:hover::before{opacity:1}
.cs-feat-icon{width:50px;height:50px;border-radius:12px;background:linear-gradient(135deg,rgba(255,204,0,.12),rgba(255,150,0,.06));border:1px solid rgba(255,204,0,.2);display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:16px;transition:all .3s}
.cs-feat:hover .cs-feat-icon{background:linear-gradient(135deg,rgba(255,204,0,.22),rgba(255,150,0,.12));box-shadow:0 0 22px rgba(255,204,0,.22);transform:scale(1.1) rotate(-6deg)}
.cs-feat-title{font-family:'Orbitron',monospace;font-size:12px;font-weight:600;color:#fff;margin-bottom:10px;letter-spacing:.05em}
.cs-feat-desc{font-size:13px;color:rgba(170,190,235,.4);line-height:1.65}

/* ── STATUS BANNER ───────────── */
.cs-banner{display:flex;align-items:center;gap:16px;padding:16px 24px;border-radius:12px;font-family:'Barlow',sans-serif;font-size:13px;margin-bottom:24px;position:relative;overflow:hidden;animation:fade-up .5s ease both}
.cs-banner-bar{position:absolute;left:0;top:0;bottom:0;width:3px;background:#ff5070;animation:bar-pulse .8s infinite}
@keyframes bar-pulse{0%,100%{opacity:1}50%{opacity:.2}}
.cs-banner.active {background:rgba(0,160,70,.07);border:1px solid rgba(0,200,80,.18);color:rgba(150,240,180,.9)}
.cs-banner.offline{background:rgba(255,204,0,.04);border:1px solid rgba(255,204,0,.1);color:rgba(255,204,0,.4)}
.cs-banner.danger {background:rgba(180,20,50,.1);border:1px solid rgba(255,60,80,.28);color:#ff8090;animation:danger-pulse .9s infinite}
@keyframes danger-pulse{0%,100%{background:rgba(180,20,50,.08)}50%{background:rgba(180,20,50,.2)}}
.cs-banner-icon{font-size:24px;flex-shrink:0}
.cs-banner-link{margin-left:auto;color:#FFCC00;text-decoration:none;font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.12em;padding:6px 14px;border:1px solid rgba(255,204,0,.3);border-radius:4px;transition:all .2s;white-space:nowrap}
.cs-banner-link:hover{background:rgba(255,204,0,.1);color:#FFD740}

/* ── PANEL HDR ───────────────── */
.cs-panel-hdr{display:flex;align-items:center;justify-content:space-between;padding:13px 20px;border-radius:10px;margin-bottom:14px;background:linear-gradient(90deg,rgba(255,204,0,.07),rgba(255,204,0,.02));border:1px solid rgba(255,204,0,.12);border-left:3px solid rgba(255,204,0,.5);font-family:'Share Tech Mono',monospace;font-size:11px;color:rgba(255,204,0,.55);text-transform:uppercase;letter-spacing:.14em}

/* ── BUTTONS ─────────────────── */
.stButton>button{font-family:'Barlow',sans-serif!important;font-weight:700!important;font-size:13px!important;text-transform:uppercase!important;letter-spacing:.12em!important;border-radius:8px!important;padding:12px 22px!important;width:100%!important;transition:all .3s cubic-bezier(.34,1.56,.64,1)!important}
.stButton>button:hover{transform:translateY(-3px)!important}
.stButton>button:active{transform:scale(.96)!important}
div[data-testid="column"]:nth-child(1) .stButton>button{background:linear-gradient(135deg,rgba(255,204,0,.1),rgba(200,150,0,.05))!important;border:1px solid #FFCC00!important;color:#FFCC00!important}
div[data-testid="column"]:nth-child(1) .stButton>button:hover{background:linear-gradient(135deg,#FFCC00,#FFB300)!important;color:#03071a!important;box-shadow:0 0 32px rgba(255,204,0,.55),0 8px 24px rgba(0,0,0,.3)!important}
div[data-testid="column"]:nth-child(2) .stButton>button{background:linear-gradient(135deg,rgba(255,60,80,.1),rgba(200,20,50,.05))!important;border:1px solid #ff5070!important;color:#ff5070!important}
div[data-testid="column"]:nth-child(2) .stButton>button:hover{background:linear-gradient(135deg,#ff3355,#cc1133)!important;color:#fff!important;box-shadow:0 0 32px rgba(255,51,85,.5)!important}
div[data-testid="column"]:nth-child(3) .stButton>button{background:transparent!important;border:1px solid rgba(100,140,220,.25)!important;color:rgba(140,180,255,.55)!important}
div[data-testid="column"]:nth-child(3) .stButton>button:hover{border-color:rgba(100,160,255,.6)!important;color:#a0c0ff!important;background:rgba(60,100,220,.07)!important}
div[data-testid="column"]:nth-child(4) .stButton>button{background:transparent!important;border:1px solid rgba(255,180,0,.18)!important;color:rgba(255,180,0,.42)!important}
div[data-testid="column"]:nth-child(4) .stButton>button:hover{border-color:rgba(255,200,0,.5)!important;color:#FFCC00!important;background:rgba(255,204,0,.05)!important}

/* ── ALERTS + LOG ────────────── */
.cs-alert{display:flex;align-items:flex-start;gap:12px;background:rgba(200,20,50,.06);border:1px solid rgba(255,60,80,.14);border-left:3px solid #ff5070;border-radius:10px;padding:14px 16px;margin-bottom:10px;animation:slide-right .35s ease both;transition:all .25s}
@keyframes slide-right{from{opacity:0;transform:translateX(16px)}to{opacity:1;transform:translateX(0)}}
.cs-alert:hover{background:rgba(200,20,50,.12);border-left-color:#FFCC00;transform:translateX(-3px)}
.cs-alert-icon{font-size:20px;flex-shrink:0;margin-top:2px}
.cs-alert-time{font-family:'Share Tech Mono',monospace;font-size:10px;color:rgba(255,140,100,.5);margin-bottom:3px}
.cs-alert-msg{font-size:13px;font-weight:600;color:#ff8090}
.cs-log{background:#03091e;border:1px solid rgba(255,204,0,.07);border-radius:10px;padding:16px;font-family:'Share Tech Mono',monospace;font-size:11px;color:rgba(160,190,240,.5);height:272px;overflow-y:auto;line-height:1.9}
.cs-log-dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:8px;vertical-align:middle}
.cs-log-s{color:#40d080} .cs-log-u{color:#ff6070}

/* ── SCREENSHOTS ─────────────── */
.cs-shot{position:relative;background:#0a1330;border:1px solid rgba(255,204,0,.1);border-radius:14px;overflow:hidden;margin-bottom:8px;transition:all .4s cubic-bezier(.34,1.56,.64,1);animation:card-rise .5s ease both;box-shadow:0 4px 24px rgba(0,0,0,.35)}
.cs-shot:hover{border-color:rgba(255,204,0,.4);transform:translateY(-8px) scale(1.015);box-shadow:0 20px 60px rgba(0,0,0,.6),0 0 0 1px rgba(255,204,0,.14)}
.cs-shot img{width:100%;display:block;cursor:pointer;transition:transform .45s ease}
.cs-shot:hover img{transform:scale(1.06)}
.cs-shot-overlay{position:absolute;inset:0;background:linear-gradient(to bottom,rgba(3,7,22,.25),rgba(3,7,22,.72));opacity:0;display:flex;align-items:center;justify-content:center;transition:opacity .3s;pointer-events:none}
.cs-shot:hover .cs-shot-overlay{opacity:1}
.cs-shot-btn{color:#FFCC00;font-family:'Orbitron',monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;background:rgba(255,204,0,.1);border:1px solid rgba(255,204,0,.35);padding:8px 18px;border-radius:4px}
.cs-shot-foot{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:10px;color:rgba(255,180,60,.6);background:linear-gradient(90deg,rgba(255,204,0,.05),rgba(255,150,0,.03));border-top:1px solid rgba(255,204,0,.09)}

/* ── MISC ────────────────────── */
.cs-confirm{background:rgba(180,20,50,.1);border:1px solid rgba(255,60,80,.28);border-radius:8px;padding:12px 16px;margin-bottom:8px;font-family:'Share Tech Mono',monospace;font-size:12px;color:#ff8090;animation:slide-right .2s ease}
.cs-scroll-btn{display:inline-flex;align-items:center;gap:7px;background:rgba(255,204,0,.07);border:1px solid rgba(255,204,0,.22);color:#FFCC00;padding:9px 22px;border-radius:6px;text-decoration:none;font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.1em;transition:all .25s}
.cs-scroll-btn:hover{background:rgba(255,204,0,.16);border-color:rgba(255,204,0,.5);transform:translateY(-2px);box-shadow:0 4px 16px rgba(255,204,0,.2)}
.cs-section-hdr{display:flex;align-items:center;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:rgba(255,204,0,.4);margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid rgba(255,204,0,.08)}
.bg-gold{background:rgba(255,204,0,.08);border:1px solid rgba(255,204,0,.22);color:#FFCC00;padding:3px 12px;border-radius:3px;font-size:10px}
.bg-red {background:rgba(255,60,80,.08);border:1px solid rgba(255,60,80,.22);color:#ff8090;padding:3px 12px;border-radius:3px;font-size:10px}
.cs-info{padding:14px 20px;border-radius:10px;font-size:13px;margin-bottom:18px;animation:fade-up .5s ease both}
.cs-info.navy{background:rgba(20,50,140,.12);border:1px solid rgba(80,120,220,.22);color:rgba(160,200,255,.8)}
.cs-info.gold{background:rgba(255,204,0,.04);border:1px solid rgba(255,204,0,.12);color:rgba(255,210,100,.7)}
.cs-empty{padding:50px 20px;text-align:center;background:rgba(6,12,34,.7);border:1px dashed rgba(255,204,0,.1);border-radius:14px;color:rgba(160,180,220,.22);font-family:'Share Tech Mono',monospace;font-size:13px;animation:fade-up .6s ease both}
.cs-divider{display:flex;align-items:center;gap:16px;margin:36px 0}
.cs-divider-line{flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(255,204,0,.2),transparent)}
.cs-divider-txt{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.22em;color:rgba(255,204,0,.3);text-transform:uppercase;white-space:nowrap}
hr{border:none!important;border-top:1px solid rgba(255,204,0,.06)!important;margin:30px 0!important}
.cs-footer{margin-top:60px;padding:40px 0 30px;border-top:1px solid rgba(255,204,0,.07);text-align:center}
.cs-footer img{height:32px;opacity:.18;margin-bottom:14px}
.cs-footer-tag{font-family:'Orbitron',monospace;font-size:11px;letter-spacing:.32em;color:rgba(255,204,0,.2);text-transform:uppercase;margin-bottom:10px}
.cs-footer-copy{font-family:'Share Tech Mono',monospace;font-size:10px;color:rgba(160,180,220,.16);letter-spacing:.14em}
</style>
""", unsafe_allow_html=True)


# ── Lightbox JS ────────────────────────────────────────────────────────────────
components.html("""
<script>
function openLightbox(src,caption){
    ['cs-lb'].forEach(function(id){
        try{var e=window.parent.document.getElementById(id);if(e)e.remove();}catch(ex){}
        var e2=document.getElementById(id);if(e2)e2.remove();
    });
    var ov=document.createElement('div');
    ov.id='cs-lb';
    ov.style.cssText='position:fixed;inset:0;z-index:99999;background:rgba(2,4,14,.97);display:flex;align-items:center;justify-content:center;backdrop-filter:blur(18px);';
    ov.innerHTML='<div style="position:relative;max-width:92vw;max-height:92vh;">'
        +'<button onclick="document.getElementById(\'cs-lb\').remove();try{window.parent.document.getElementById(\'cs-lb\').remove();}catch(e){}" '
        +'style="position:absolute;top:-15px;right:-15px;width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#FFCC00,#FF9900);border:none;color:#03071a;font-size:16px;font-weight:bold;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px rgba(255,204,0,.6);z-index:1;transition:transform .25s;" '
        +'onmouseover="this.style.transform=\'scale(1.2) rotate(90deg)\'" onmouseout="this.style.transform=\'scale(1)\'">&#x2715;</button>'
        +'<img src="'+src+'" style="max-width:90vw;max-height:80vh;border-radius:12px;border:1px solid rgba(255,204,0,.2);box-shadow:0 0 80px rgba(255,204,0,.1),0 40px 80px rgba(0,0,0,.8);display:block;"/>'
        +'<div style="text-align:center;margin-top:14px;font-family:Share Tech Mono,monospace;font-size:12px;color:rgba(255,200,80,.6);">&#9888; '+caption+'</div>'
        +'</div>';
    ov.onclick=function(e){if(e.target===ov){ov.remove();try{window.parent.document.getElementById('cs-lb').remove();}catch(ex){}}};
    try{window.parent.document.body.appendChild(ov);window.parent.openLightbox=openLightbox;}
    catch(e){document.body.appendChild(ov);}
}
window.openLightbox=openLightbox;
document.addEventListener('keydown',function(e){
    if(e.key==='Escape'){
        try{var el=window.parent.document.getElementById('cs-lb');if(el)el.remove();}catch(ex){}
        var el2=document.getElementById('cs-lb');if(el2)el2.remove();
    }
});
</script>
""", height=0)


# ── Data ───────────────────────────────────────────────────────────────────────
running        = is_running()
screenshots    = get_screenshots()
students_total, unknowns_total = count_log_events()
recent_alerts  = parse_recent_alerts()
uptime         = uptime_str()
now_str        = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

pill_cls = "alert" if (running and unknowns_total > 0) else ("live" if running else "offline")
pill_lbl = ("⚠ ALERT" if (running and unknowns_total > 0)
            else ("● LIVE" if running else "○ OFFLINE"))
cam_color = "#00ff80" if running else "#ff5070"
cam_label = "LIVE MONITORING" if running else "OFFLINE"


# ══════════════════════════════════════════════════════════════════════════════
# NAV  — logo only here, no other image
# ══════════════════════════════════════════════════════════════════════════════
nav_logo_html = (f'<img src="{LOGO_SRC}" style="height:46px;width:auto;"/>'
                 if LOGO_SRC else
                 '<span style="font-family:Orbitron,sans-serif;font-size:18px;'
                 'font-weight:700;color:#fff">CAM<span style="color:#FFCC00">SECURE</span></span>')

st.markdown(f"""
<div class="cs-nav">
  <div class="cs-nav-logo">{nav_logo_html}</div>
  <div class="cs-nav-center">✦ Your Security Matters ✦</div>
  <div class="cs-nav-right">
    <span class="cs-nav-clock">{now_str}</span>
    <div class="cs-pill {pill_cls}">{pill_lbl}</div>
    <a href="#screenshot-anchor" class="cs-nav-badge">📸 {len(screenshots)} ALERTS</a>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO  — no images embedded, cam card uses CSS-only preview
# ══════════════════════════════════════════════════════════════════════════════
hero_logo_html = (f'<div class="cs-hero-logo-wrap"><img src="{LOGO_SRC}" '
                  f'alt="CamSecure"/></div>'
                  if LOGO_SRC else "")

# cam card preview: if image available embed it; otherwise icon-only
if CAM_SRC:
    cam_preview = (f'<img src="{CAM_SRC}" style="width:100%;height:160px;'
                   f'object-fit:cover;filter:brightness(.75) saturate(.55);display:block;"/>')
else:
    cam_preview = '<div class="cs-cam-preview-icon">📹</div>'

st.markdown(f"""
<div class="cs-hero">
  <div class="cs-hero-bg"></div>
  <div class="cs-hero-overlay"></div>
  <div class="cs-hero-grid"></div>
  <div class="cs-hero-scan"></div>

  <div class="cs-hero-content">
    <div class="cs-eyebrow"><div class="cs-dot"></div>AI-Powered Gate Security System</div>
    {hero_logo_html}
    <div class="cs-tagline">✦ &nbsp; Your Security Matters &nbsp; ✦</div>
    <h1 class="cs-hero-title">
      Protecting Every<br>
      <span class="cs-hero-title-gold">Entry Point</span><br>
      In Real Time
    </h1>
    <p class="cs-hero-sub">
      Advanced face recognition powered by MediaPipe &amp; FaceNet.
      Instant alerts, auto-screenshot evidence, and live detection —
      all from one intelligent dashboard.
    </p>
    <div class="cs-hero-stats">
      <div class="cs-stat-item"><div class="cs-stat-num">12+</div><div class="cs-stat-lbl">Years of Excellence</div></div>
      <div class="cs-stat-item"><div class="cs-stat-num">5K+</div><div class="cs-stat-lbl">Institutions Secured</div></div>
      <div class="cs-stat-item"><div class="cs-stat-num">99.8%</div><div class="cs-stat-lbl">Detection Accuracy</div></div>
    </div>
  </div>

  <div class="cs-cam-card">
    <div class="cs-cam-preview">
      {cam_preview}
      <div class="cs-cam-scan"></div>
      <div class="cs-cam-corner">● REC</div>
    </div>
    <div class="cs-cam-body">
      <div class="cs-cam-title">GATE-01 · FRONT ENTRANCE</div>
      <div class="cs-cam-desc">FaceNet + MediaPipe · 1280×720 · 30fps</div>
      <div class="cs-cam-status">
        <div class="cs-cam-dot" style="background:{cam_color};box-shadow:0 0 8px {cam_color}"></div>
        <span class="cs-cam-txt" style="color:{cam_color}">{cam_label}</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAT CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="cs-stat-row">
  <div class="cs-card" style="animation-delay:0s">
    <div class="cs-card-corner"></div>
    <span class="cs-card-icon">✅</span>
    <div class="cs-card-val gold">{students_total}</div>
    <div class="cs-card-lbl">Verified Students</div>
    <div class="cs-bar"><div class="cs-bar-fill bg" style="width:{min(100,students_total)}%"></div></div>
  </div>
  <div class="cs-card" style="animation-delay:.1s">
    <div class="cs-card-corner"></div>
    <span class="cs-card-icon">🚨</span>
    <div class="cs-card-val red">{unknowns_total}</div>
    <div class="cs-card-lbl">Unknown Alerts</div>
    <div class="cs-bar"><div class="cs-bar-fill br" style="width:{min(100,unknowns_total*5)}%"></div></div>
  </div>
  <div class="cs-card link" style="animation-delay:.2s">
    <a href="#screenshot-anchor" style="text-decoration:none;color:inherit;display:block">
      <div class="cs-card-corner"></div>
      <span class="cs-card-icon">📸</span>
      <div class="cs-card-val white">{len(screenshots)}</div>
      <div class="cs-card-lbl">Screenshots · click to view ↓</div>
      <div class="cs-bar"><div class="cs-bar-fill bw" style="width:{min(100,len(screenshots)*10)}%"></div></div>
    </a>
  </div>
  <div class="cs-card" style="animation-delay:.3s">
    <div class="cs-card-corner"></div>
    <span class="cs-card-icon">⏱</span>
    <div class="cs-card-val cyan">{uptime}</div>
    <div class="cs-card-lbl">Session Uptime</div>
    <div class="cs-bar"><div class="cs-bar-fill bc" style="width:55%"></div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cs-features">
  <div class="cs-feat" style="animation-delay:.05s">
    <div class="cs-feat-icon">🧠</div>
    <div class="cs-feat-title">AI Face Recognition</div>
    <div class="cs-feat-desc">FaceNet InceptionResnetV1 with VGGFace2 weights. 512-dimensional embeddings for precise identity matching in any lighting.</div>
  </div>
  <div class="cs-feat" style="animation-delay:.15s">
    <div class="cs-feat-icon">⚡</div>
    <div class="cs-feat-title">Real-Time Detection</div>
    <div class="cs-feat-desc">MediaPipe tile-scanning detects multiple faces simultaneously at 30fps with sub-second alert response time.</div>
  </div>
  <div class="cs-feat" style="animation-delay:.25s">
    <div class="cs-feat-icon">🛡</div>
    <div class="cs-feat-title">Instant Alerts</div>
    <div class="cs-feat-desc">Auto-screenshot on unknown detection with visual + audio alerts. Full timestamped evidence log kept automatically.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATUS BANNER
# ══════════════════════════════════════════════════════════════════════════════
if running and unknowns_total > 0:
    st.markdown(f"""
    <div class="cs-banner danger">
      <div class="cs-banner-bar"></div>
      <div class="cs-banner-icon">⚠</div>
      <div><div style="font-weight:700;font-size:14px;letter-spacing:.06em">SECURITY ALERT ACTIVE</div>
           <div style="font-size:12px;opacity:.8;margin-top:2px">Unauthorized person detected — Security team notified</div></div>
      <a href="#screenshot-anchor" class="cs-banner-link">VIEW EVIDENCE ↓</a>
    </div>""", unsafe_allow_html=True)
elif running:
    st.markdown("""
    <div class="cs-banner active">
      <div class="cs-banner-icon">🛡</div>
      <div><div style="font-weight:700;font-size:14px;letter-spacing:.06em">SYSTEM ACTIVE — ALL CLEAR</div>
           <div style="font-size:12px;opacity:.7;margin-top:2px">Gate camera is live. All persons verified as authorized students.</div></div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="cs-banner offline">
      <div class="cs-banner-icon">📷</div>
      <div><div style="font-weight:700;font-size:14px;letter-spacing:.06em">SYSTEM OFFLINE</div>
           <div style="font-size:12px;opacity:.6;margin-top:2px">Click "Open Camera" below to begin gate monitoring.</div></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cs-divider">
  <div class="cs-divider-line"></div>
  <div class="cs-divider-txt">Camera Control Panel</div>
  <div class="cs-divider-line"></div>
</div>
<div class="cs-panel-hdr">
  <span>📷 Gate Camera — Control Panel</span>
  <span>GATE-01 · FRONT ENTRANCE · MEDIAPIPE + FACENET</span>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("▶  Open Camera", disabled=running, key="btn_open"):
        if not RUN_SCRIPT.exists():
            st.error(f"❌  run.py not found in: {Path.cwd()}")
        else:
            try:
                st.session_state.process    = subprocess.Popen(
                    [sys.executable, str(RUN_SCRIPT)], cwd=str(Path.cwd()))
                st.session_state.cam_running = True
                st.session_state.start_time  = datetime.datetime.now()
                time.sleep(0.8); st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
with c2:
    if st.button("■  Stop Camera", disabled=not running, key="btn_stop"):
        p = st.session_state.process
        if p:
            p.terminate()
            try: p.wait(timeout=4)
            except Exception: p.kill()
        st.session_state.process     = None
        st.session_state.cam_running = False
        st.session_state.start_time  = None
        st.rerun()
with c3:
    if st.button("↺  Refresh Stats", key="btn_refresh"): st.rerun()
with c4:
    if st.button("🗑  Clear Log", key="btn_clear_log"):
        if LOG_FILE.exists(): LOG_FILE.write_text("")
        st.rerun()

if running:
    st.markdown("""<div class="cs-info navy">
      📷 <strong>Camera window is open on your desktop.</strong>
      Press <strong>Q</strong> to quit, <strong>S</strong> for a manual screenshot.
      Use <strong>Refresh Stats</strong> to update the dashboard.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="cs-info gold">
      ℹ️  Ensure <code>run.py</code> and <code>face_db/embeddings.pkl</code>
      are in the same folder as this app.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALERTS + LOG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cs-divider">
  <div class="cs-divider-line"></div>
  <div class="cs-divider-txt">Live Detection Feed</div>
  <div class="cs-divider-line"></div>
</div>""", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown(f"""<div class="cs-panel-hdr">
      <span>🚨 Recent Security Alerts</span><span>{unknowns_total} TOTAL</span>
    </div>""", unsafe_allow_html=True)
    if recent_alerts:
        for i, line in enumerate(recent_alerts):
            try:    ts = line[1:20]; msg = "Unknown person detected at gate"
            except: ts = ""; msg = line
            st.markdown(f"""
            <div class="cs-alert" style="animation-delay:{i*.07}s">
              <div class="cs-alert-icon">⚠</div>
              <div><div class="cs-alert-time">🕐 {ts}</div>
                   <div class="cs-alert-msg">{msg}</div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="text-align:center;margin-top:16px">
          <a href="#screenshot-anchor" class="cs-scroll-btn">📸 VIEW ALERT EVIDENCE ↓</a>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="cs-empty">
          <div style="font-size:32px;margin-bottom:10px;opacity:.2">🛡</div>
          No incidents recorded<br>
          <span style="font-size:11px;opacity:.45">Gate is secure — monitoring active</span>
        </div>""", unsafe_allow_html=True)

with right:
    st.markdown(f"""<div class="cs-panel-hdr">
      <span>📋 Detection Log</span><span>{students_total+unknowns_total} ENTRIES</span>
    </div>""", unsafe_allow_html=True)
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text(errors="ignore").splitlines()[-50:]
        log_html = ""
        for line in reversed(lines):
            if not line.strip(): continue
            if   "STUDENT" in line: log_html += f'<div class="cs-log-s"><span class="cs-log-dot" style="background:#40d080"></span>{line}</div>'
            elif "UNKNOWN" in line: log_html += f'<div class="cs-log-u"><span class="cs-log-dot" style="background:#ff6070"></span>{line}</div>'
            else: log_html += f'<div style="color:rgba(140,160,200,.2)">{line}</div>'
        st.markdown(f'<div class="cs-log">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div class="cs-log" style="color:rgba(140,160,200,.15);text-align:center;padding:50px;line-height:2.5">
          Detection log will appear here<br>
          <span style="font-size:10px">once the camera begins monitoring</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT GALLERY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cs-divider">
  <div class="cs-divider-line"></div>
  <div class="cs-divider-txt">Alert Evidence Gallery</div>
  <div class="cs-divider-line"></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div id="screenshot-anchor"></div>', unsafe_allow_html=True)

del_col, title_col = st.columns([1, 5])
with title_col:
    st.markdown(f"""<div class="cs-section-hdr">
      <span>📸 Alert Evidence — Screenshots</span>
      <span><span class="bg-red">{len(screenshots)} SAVED</span>&nbsp;
            <span class="bg-gold">AUTO-CAPTURED</span></span>
    </div>""", unsafe_allow_html=True)
with del_col:
    if screenshots:
        if st.button("🗑 Delete All", key="del_all"):
            st.session_state.delete_confirm = "__ALL__"

if st.session_state.delete_confirm == "__ALL__":
    st.markdown("""<div class="cs-confirm">
      ⚠ Permanently delete <strong>ALL</strong> screenshots? This cannot be undone.
    </div>""", unsafe_allow_html=True)
    yc, nc, _ = st.columns([1, 1, 4])
    with yc:
        if st.button("✓ Confirm", key="confirm_del_all"):
            for f in screenshots:
                try: f.unlink()
                except Exception: pass
            st.session_state.delete_confirm = None; st.rerun()
    with nc:
        if st.button("✕ Cancel", key="cancel_del_all"):
            st.session_state.delete_confirm = None; st.rerun()

if screenshots:
    for ri, row in enumerate([screenshots[i:i+3] for i in range(0, len(screenshots), 3)]):
        cols = st.columns(3)
        for col, shot in zip(cols, row):
            with col:
                b64 = screenshot_to_b64(shot)
                ts  = fmt_ts(shot.stem)
                if b64:
                    src = f"data:image/jpeg;base64,{b64}"
                    st.markdown(f"""
                    <div class="cs-shot" style="animation-delay:{ri*.1}s">
                      <div class="cs-shot-overlay"><div class="cs-shot-btn">🔍 Enlarge</div></div>
                      <img src="{src}" alt="Alert"
                           onclick="(window.parent.openLightbox||window.openLightbox)('{src}','{ts}')"
                           title="Click to enlarge"/>
                      <div class="cs-shot-foot">
                        <span>⚠ {ts}</span>
                        <span style="font-size:9px;color:rgba(255,204,0,.26);letter-spacing:.1em">↗ ENLARGE</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("🗑 Delete", key=f"del_{shot.name}"):
                        st.session_state.delete_confirm = shot.name
                    if st.session_state.delete_confirm == shot.name:
                        st.markdown(f"""<div class="cs-confirm">
                          Delete <strong>{shot.name}</strong>?</div>""",
                          unsafe_allow_html=True)
                        y, n = st.columns(2)
                        with y:
                            if st.button("✓ Yes", key=f"yes_{shot.name}"):
                                try: shot.unlink()
                                except Exception: pass
                                st.session_state.delete_confirm = None; st.rerun()
                        with n:
                            if st.button("✕ No", key=f"no_{shot.name}"):
                                st.session_state.delete_confirm = None; st.rerun()
else:
    st.markdown("""<div class="cs-empty" style="padding:60px 20px">
      <div style="font-size:48px;margin-bottom:16px;opacity:.1">🛡</div>
      No alert screenshots on record<br>
      <span style="font-size:11px;opacity:.35">Screenshots are captured automatically when an unknown person is detected</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER  — logo only here
# ══════════════════════════════════════════════════════════════════════════════
footer_logo = (f'<img src="{LOGO_SRC}" alt="CamSecure"/><br/>' if LOGO_SRC else "")
st.markdown(f"""
<div class="cs-footer">
  {footer_logo}
  <div class="cs-footer-tag">✦ &nbsp; Your Security Matters &nbsp; ✦</div>
  <div class="cs-footer-copy">
    CAMSECURE &nbsp;·&nbsp; COLLEGE GATE SECURITY SYSTEM
    &nbsp;·&nbsp; MEDIAPIPE + FACENET (VGGFACE2)
    &nbsp;·&nbsp; {datetime.datetime.now().year}
  </div>
</div>""", unsafe_allow_html=True)