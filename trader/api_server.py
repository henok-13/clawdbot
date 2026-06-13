"""
iPhone-friendly REST API + mobile web dashboard for the XAU/USD trader bot.

Endpoints:
  GET  /           — mobile PWA dashboard (add to iPhone home screen)
  GET  /api/status — bot state, equity, open trades
  GET  /api/news   — latest gold headlines
  GET  /api/signal — most recent evaluated signal
  POST /api/control — {"action": "pause"|"resume"}

Start alongside the bot:
  python run_trader.py --with-api --api-port 8080
"""

import threading
import logging
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, jsonify, request, render_template_string

from .config import risk_cfg, strategy_cfg
from .news_feed import news_feed

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Shared mutable state — written by bot loop, read by API
_state: dict = {
    "running": True,
    "paused": False,
    "last_tick": None,
    "last_signal": None,
    "open_trades": [],
    "equity": 0.0,
    "daily_pnl": 0.0,
    "total_trades_today": 0,
    "wins_today": 0,
}
_lock = threading.Lock()


# ── State helpers called by bot.py ───────────────────────────────────────────

def update_state(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)
        _state["last_tick"] = datetime.now(timezone.utc).isoformat()


def is_paused() -> bool:
    with _lock:
        return _state["paused"]


# ── REST endpoints ───────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    with _lock:
        data = dict(_state)
    # Serialise Signal dataclass if present
    if data.get("last_signal"):
        sig = data["last_signal"]
        data["last_signal"] = {
            "direction": sig.direction,
            "entry_type": sig.entry_type,
            "strength": sig.strength,
            "reason": sig.reason,
        }
    return jsonify(data)


@app.route("/api/news")
def api_news():
    headlines = news_feed.latest_headlines(n=8)
    return jsonify([
        {
            "title": h.title,
            "source": h.source,
            "published_at": h.published_at.isoformat(),
            "url": h.url,
            "is_high_impact": h.is_high_impact,
        }
        for h in headlines
    ])


@app.route("/api/control", methods=["POST"])
def api_control():
    body = request.get_json(silent=True) or {}
    action = body.get("action", "")
    if action == "pause":
        update_state(paused=True)
        logger.info("Bot PAUSED via iPhone dashboard")
        return jsonify({"ok": True, "paused": True})
    if action == "resume":
        update_state(paused=False)
        logger.info("Bot RESUMED via iPhone dashboard")
        return jsonify({"ok": True, "paused": False})
    return jsonify({"ok": False, "error": "unknown action"}), 400


@app.route("/")
def dashboard():
    return render_template_string(_DASHBOARD_HTML,
                                  symbol=strategy_cfg.symbol,
                                  tp=risk_cfg.take_profit_points,
                                  sl=risk_cfg.stop_loss_points)


# ── Mobile PWA HTML (single file, no CDN deps) ───────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="XAU Trader">
<title>XAU/USD Trader</title>
<style>
  :root{--gold:#f5c518;--green:#26a69a;--red:#ef5350;--bg:#121212;--card:#1e1e1e;--text:#e0e0e0;--sub:#888}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
  header{background:var(--card);padding:16px 20px 12px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #2a2a2a;position:sticky;top:0;z-index:10}
  header h1{font-size:1.1rem;font-weight:700;color:var(--gold);letter-spacing:.5px}
  #dot{width:10px;height:10px;border-radius:50%;background:#555;display:inline-block;margin-right:6px;transition:background .3s}
  #dot.live{background:var(--green);box-shadow:0 0 6px var(--green)}
  #dot.paused{background:var(--gold)}
  main{padding:16px;display:grid;gap:12px}
  .card{background:var(--card);border-radius:14px;padding:16px}
  .card h2{font-size:.7rem;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}
  .metric-row{display:flex;gap:12px}
  .metric{flex:1;text-align:center}
  .metric .val{font-size:1.5rem;font-weight:700;color:var(--gold)}
  .metric .lbl{font-size:.65rem;color:var(--sub);margin-top:2px}
  .signal-box{text-align:center;padding:8px 0}
  .signal-box .dir{font-size:2rem;font-weight:800;letter-spacing:1px}
  .signal-box .dir.BUY{color:var(--green)}
  .signal-box .dir.SELL{color:var(--red)}
  .signal-box .dir.NONE{color:var(--sub);font-size:1.2rem}
  .signal-box .type{font-size:.8rem;color:var(--sub);margin-top:4px}
  .strength-bar{background:#2a2a2a;border-radius:6px;height:6px;margin-top:10px;overflow:hidden}
  .strength-fill{height:100%;border-radius:6px;background:var(--gold);transition:width .5s}
  .news-item{padding:10px 0;border-bottom:1px solid #2a2a2a}
  .news-item:last-child{border:none}
  .news-item .headline{font-size:.85rem;line-height:1.35}
  .news-item .meta{font-size:.7rem;color:var(--sub);margin-top:4px}
  .news-item.hi .headline{color:var(--gold)}
  .trade-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #2a2a2a;font-size:.85rem}
  .trade-row:last-child{border:none}
  .buy-tag{color:var(--green);font-weight:700}
  .sell-tag{color:var(--red);font-weight:700}
  .ctrl-row{display:flex;gap:10px;margin-top:4px}
  button{flex:1;padding:14px;border:none;border-radius:12px;font-size:.9rem;font-weight:600;cursor:pointer;transition:opacity .2s}
  button:active{opacity:.7}
  #btn-pause{background:var(--gold);color:#000}
  #btn-resume{background:var(--green);color:#fff}
  .ts{font-size:.65rem;color:var(--sub);text-align:right;margin-top:8px}
  .pnl-pos{color:var(--green)}
  .pnl-neg{color:var(--red)}
</style>
</head>
<body>
<header>
  <h1><span id="dot"></span>{{ symbol }} Bot</h1>
  <span style="font-size:.75rem;color:var(--sub)">TP {{ tp }}pt &nbsp;SL {{ sl }}pt</span>
</header>
<main>
  <!-- Equity / P&L -->
  <div class="card">
    <h2>Account</h2>
    <div class="metric-row">
      <div class="metric"><div class="val" id="equity">—</div><div class="lbl">Equity</div></div>
      <div class="metric"><div class="val" id="daily-pnl">—</div><div class="lbl">Day P&L</div></div>
      <div class="metric"><div class="val" id="win-rate">—</div><div class="lbl">Win rate</div></div>
    </div>
  </div>

  <!-- Signal -->
  <div class="card">
    <h2>Current Signal</h2>
    <div class="signal-box">
      <div class="dir NONE" id="sig-dir">NO SIGNAL</div>
      <div class="type" id="sig-type"></div>
    </div>
    <div class="strength-bar"><div class="strength-fill" id="sig-bar" style="width:0%"></div></div>
    <div class="ts" id="sig-reason"></div>
  </div>

  <!-- Open trades -->
  <div class="card">
    <h2>Open Trades</h2>
    <div id="trades-list"><div style="color:var(--sub);font-size:.85rem">No open trades</div></div>
  </div>

  <!-- Control -->
  <div class="card">
    <h2>Control</h2>
    <div class="ctrl-row">
      <button id="btn-pause" onclick="control('pause')">Pause Bot</button>
      <button id="btn-resume" onclick="control('resume')">Resume Bot</button>
    </div>
    <div class="ts" id="last-tick"></div>
  </div>

  <!-- News -->
  <div class="card">
    <h2>Gold News</h2>
    <div id="news-list"><div style="color:var(--sub);font-size:.85rem">Loading…</div></div>
  </div>
</main>

<script>
function fmt(n){return n==null?'—':'$'+n.toFixed(2)}
function timeSince(iso){
  if(!iso)return'';
  const s=Math.floor((Date.now()-new Date(iso))/1000);
  if(s<60)return s+'s ago';
  if(s<3600)return Math.floor(s/60)+'m ago';
  return Math.floor(s/3600)+'h ago';
}

async function refresh(){
  try{
    const [st,nw]=await Promise.all([
      fetch('/api/status').then(r=>r.json()),
      fetch('/api/news').then(r=>r.json()),
    ]);

    // dot
    const dot=document.getElementById('dot');
    dot.className=st.paused?'paused':'live';

    // account
    document.getElementById('equity').textContent=fmt(st.equity);
    const pnl=st.daily_pnl||0;
    const pnlEl=document.getElementById('daily-pnl');
    pnlEl.textContent=(pnl>=0?'+':'')+fmt(pnl);
    pnlEl.className='val '+(pnl>=0?'pnl-pos':'pnl-neg');
    const wr=st.total_trades_today>0?Math.round(st.wins_today/st.total_trades_today*100)+'%':'—';
    document.getElementById('win-rate').textContent=wr;

    // signal
    const sig=st.last_signal;
    const dirEl=document.getElementById('sig-dir');
    if(sig){
      dirEl.textContent=sig.direction;
      dirEl.className='dir '+sig.direction;
      document.getElementById('sig-type').textContent=sig.entry_type;
      document.getElementById('sig-bar').style.width=(sig.strength*100)+'%';
      document.getElementById('sig-reason').textContent=sig.reason||'';
    } else {
      dirEl.textContent='NO SIGNAL';
      dirEl.className='dir NONE';
      document.getElementById('sig-type').textContent='';
      document.getElementById('sig-bar').style.width='0%';
      document.getElementById('sig-reason').textContent='';
    }

    // trades
    const tl=document.getElementById('trades-list');
    if(st.open_trades&&st.open_trades.length){
      tl.innerHTML=st.open_trades.map(t=>`
        <div class="trade-row">
          <span class="${t.type}-tag">${t.type}</span>
          <span>${t.lot} lot</span>
          <span class="${t.profit>=0?'pnl-pos':'pnl-neg'}">${t.profit>=0?'+':''}${fmt(t.profit)}</span>
        </div>`).join('');
    } else {
      tl.innerHTML='<div style="color:var(--sub);font-size:.85rem">No open trades</div>';
    }

    // last tick
    document.getElementById('last-tick').textContent='Updated '+timeSince(st.last_tick);

    // news
    const nl=document.getElementById('news-list');
    nl.innerHTML=nw.map(h=>`
      <div class="news-item${h.is_high_impact?' hi':''}">
        <div class="headline">${h.is_high_impact?'⚠ ':''}${h.title}</div>
        <div class="meta">${h.source} · ${timeSince(h.published_at)}</div>
      </div>`).join('');

  }catch(e){console.error(e)}
}

async function control(action){
  try{
    const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
    const j=await r.json();
    document.getElementById('dot').className=j.paused?'paused':'live';
  }catch(e){alert('Error: '+e)}
}

refresh();
setInterval(refresh,15000);  // refresh every 15 s
</script>
</body>
</html>
"""


def start_api_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start Flask in a daemon thread so it doesn't block the bot loop."""
    def _run():
        import os
        os.environ.setdefault("WERKZEUG_RUN_MAIN", "true")  # suppress reloader noise
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, name="api-server", daemon=True)
    t.start()
    logger.info("iPhone dashboard running at http://<your-server-ip>:%d", port)
