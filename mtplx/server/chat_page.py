"""The browser chat page an MTPLX server serves at ``/``.

Pure string templating on the standard library, so the Splash bridge renders
the same page as the MLX server without importing the MLX stack. Each engine
fills the same slots; ``MLX_ENGINE_UI`` holds the MLX engine's own wording.
"""

from __future__ import annotations

import html
import json
from typing import Any

# The page's engine-specific slots, as the MLX engine fills them. Help texts
# are HTML. ``locked_controls`` maps a control to the value the engine fixes
# it at; the page shows it disabled and never lets a payload move it.
MLX_ENGINE_UI: dict[str, Any] = {
    "speculative_label": "MTP",
    "depth_label": "Draft depth",
    "depth_help": "MTP draft tokens per verify cycle.",
    "top_p_min": 0,
    "top_k_min": 0,
    "top_k_max": 100,
    "top_k_help": "0 disables top-k.",
    "presence_help": "0 is exact (best for coding); 0.5&ndash;1.5 discourages repetition.",
    "locked_controls": {},
}


def chat_ui_html(
    *,
    model_id: str,
    server_url: str,
    api_key_required: bool,
    default_settings: dict[str, Any],
    engine_ui: dict[str, Any] | None = None,
) -> str:
    """Render the browser chat page.

    ``engine_ui`` overrides entries of ``MLX_ENGINE_UI`` for an engine whose
    sampler or speculation differs; left out, the page is the MLX engine's.
    """
    ui = {**MLX_ENGINE_UI, **(engine_ui or {})}
    api_note = "API key required" if api_key_required else "local · no API key"
    depth_max = max(1, int(default_settings.get("depth_max", 3) or 3))
    default_depth = max(1, min(depth_max, int(default_settings.get("depth", 3))))
    default_settings = {
        "mtp_enabled": bool(default_settings.get("mtp_enabled", True)),
        "temperature": float(default_settings.get("temperature", 0.6)),
        "top_p": float(default_settings.get("top_p", 0.95)),
        "top_k": int(default_settings.get("top_k", 20)),
        "depth": default_depth,
        "max_tokens": int(default_settings.get("max_tokens", 16384)),
        "reasoning": str(default_settings.get("reasoning", "auto")),
        "system": str(default_settings.get("system", "")),
    }
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>MTPLX</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' x2='1' y1='0' y2='1'%3E%3Cstop offset='0' stop-color='%23a5b4fc'/%3E%3Cstop offset='1' stop-color='%235b8dee'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='32' height='32' rx='9' fill='url(%23g)'/%3E%3Ctext x='50%25' y='59%25' font-family='-apple-system,Segoe UI,sans-serif' font-size='15' font-weight='800' fill='white' text-anchor='middle'%3EM%3C/text%3E%3C/svg%3E">
  <style>
    :root {
      color-scheme: dark;
      --bg: #0a0b0d;
      --surface: #14161a;
      --surface-2: #1a1d22;
      --line: rgba(255, 255, 255, 0.07);
      --line-strong: rgba(255, 255, 255, 0.14);
      --text: #ececec;
      --muted: #9ca3af;
      --muted-2: #6b7280;
      --accent: #5b8dee;
      --accent-strong: #7aa3f3;
      --accent-soft: rgba(91, 141, 238, 0.10);
      --user-tint: rgba(91, 141, 238, 0.09);
      --reason-bg: #1a1726;
      --reason-text: #c5b8e0;
      --reason-border: rgba(165, 132, 245, 0.22);
      --code-bg: #0a0b0d;
      --code-border: rgba(255, 255, 255, 0.06);
      --ok: #5fd28b;
      --warn: #f0b85c;
      --error: #ef6868;
      --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
      --font-mono: "SF Mono", ui-monospace, Menlo, Monaco, "Cascadia Code", Consolas, monospace;
      --sidebar-w: 268px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 15px;
      line-height: 1.6;
      display: grid;
      grid-template-rows: 48px 1fr;
      grid-template-columns: var(--sidebar-w) 1fr;
      grid-template-areas:
        "topbar topbar"
        "sidebar chat";
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    @media (max-width: 900px) {
      body { grid-template-columns: 1fr; grid-template-areas: "topbar" "chat"; }
      aside#sidebar { display: none; }
      aside#sidebar.open { display: block; position: fixed; inset: 48px 0 0 0; z-index: 30; }
    }

    /* Topbar */
    header.topbar {
      grid-area: topbar;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(20, 22, 26, 0.85);
      backdrop-filter: blur(12px);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 14px;
      letter-spacing: 0.2px;
    }
    .brand .logo {
      width: 24px;
      height: 24px;
      border-radius: 7px;
      background: linear-gradient(135deg, #a5b4fc, var(--accent));
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: 800; font-size: 13px;
    }
    .topbar-meta { color: var(--muted); font-size: 13px; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .topbar-meta .sep { color: var(--muted-2); margin: 0 8px; }
    .topbar-meta .dim { color: var(--muted-2); }
    .topbar-actions { display: flex; align-items: center; gap: 4px; }
    .runtime-pill {
      display: inline-flex; align-items: center;
      min-height: 24px;
      padding: 0 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: var(--surface);
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    .runtime-pill[hidden] { display: none; }
    .icon-btn {
      width: 32px; height: 32px;
      border: 0; border-radius: 8px;
      background: transparent; color: var(--muted);
      cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      transition: background 0.12s, color 0.12s;
    }
    .icon-btn:hover { background: var(--surface-2); color: var(--text); }
    .icon-btn svg { width: 16px; height: 16px; }

    /* Sidebar */
    aside#sidebar {
      grid-area: sidebar;
      background: var(--surface);
      border-right: 1px solid var(--line);
      overflow-y: auto;
      padding: 18px 16px 24px;
    }
    .sb-section { margin-bottom: 20px; }
    .sb-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--muted-2);
      margin: 0 0 10px;
      padding: 0 2px;
    }
    .sb-row { margin-bottom: 14px; padding: 0 2px; }
    .sb-row label {
      display: flex; justify-content: space-between; align-items: baseline;
      font-size: 13px; color: var(--text); margin-bottom: 6px;
    }
    .sb-row label .v {
      color: var(--muted);
      font-weight: 600; font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .sb-row textarea, .sb-row select {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--text);
      padding: 9px 11px;
      font: inherit; font-size: 13px;
      outline: none;
      transition: border-color 0.12s, box-shadow 0.12s;
    }
    .sb-row textarea:focus, .sb-row select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .sb-row textarea { min-height: 64px; max-height: 200px; resize: vertical; line-height: 1.45; }
    .sb-row .help { color: var(--muted-2); font-size: 11px; margin-top: 5px; line-height: 1.4; }
    .switch-row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; margin-bottom: 12px;
    }
    .switch-row label { margin: 0; display: inline-flex; align-items: baseline; gap: 6px; }
    .switch {
      position: relative; display: inline-flex; align-items: center;
      width: 40px; height: 24px; flex: 0 0 auto;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .switch .track {
      position: absolute; inset: 0;
      border-radius: 999px;
      background: var(--surface-2);
      border: 1px solid var(--line);
      transition: background 0.12s, border-color 0.12s;
    }
    .switch .thumb {
      position: absolute; top: 4px; left: 4px;
      width: 16px; height: 16px; border-radius: 50%;
      background: var(--muted);
      transition: transform 0.12s, background 0.12s;
    }
    .switch input:checked + .track { background: var(--accent); border-color: var(--accent); }
    .switch input:checked + .track + .thumb { transform: translateX(16px); background: white; }
    .switch input:focus-visible + .track { box-shadow: 0 0 0 3px var(--accent-soft); }
    .switch input:disabled + .track, .switch input:disabled + .track + .thumb { opacity: 0.55; }
    .sb-row select {
      appearance: none; -webkit-appearance: none; padding-right: 28px;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath d='M2 4l3 3 3-3' stroke='%239ca3af' stroke-width='1.4' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 10px center; background-size: 10px;
    }

    /* Slider — thin track + dot thumb (Open WebUI style) */
    input[type="range"] {
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 18px;
      background: transparent; cursor: pointer;
      margin: 0;
    }
    input[type="range"]::-webkit-slider-runnable-track {
      height: 3px; border-radius: 999px;
      background: linear-gradient(to right, var(--accent) 0%, var(--accent) var(--filled, 0%), var(--surface-2) var(--filled, 0%), var(--surface-2) 100%);
    }
    input[type="range"]::-moz-range-track { height: 3px; border-radius: 999px; background: var(--surface-2); }
    input[type="range"]::-moz-range-progress { height: 3px; border-radius: 999px; background: var(--accent); }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px; height: 14px;
      border-radius: 50%;
      background: white; border: 0;
      margin-top: -5.5px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
      cursor: grab;
      transition: transform 0.1s;
    }
    input[type="range"]:active::-webkit-slider-thumb { transform: scale(1.18); cursor: grabbing; }
    input[type="range"]::-moz-range-thumb {
      width: 14px; height: 14px; border-radius: 50%;
      background: white; border: 0;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
      cursor: grab;
    }
    input[type="range"]:disabled { cursor: default; opacity: 0.45; }

    .sb-actions { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
    .sb-btn {
      width: 100%;
      background: transparent; border: 1px solid var(--line);
      color: var(--muted);
      padding: 9px 12px; border-radius: 8px;
      cursor: pointer; font: inherit; font-size: 12px;
      transition: border-color 0.12s, color 0.12s, background 0.12s;
    }
    .sb-btn:hover { border-color: var(--line-strong); color: var(--text); background: var(--surface-2); }

    /* Chat area */
    main.chat-area {
      grid-area: chat; min-height: 0;
      display: grid; grid-template-rows: 1fr auto;
      position: relative; overflow: hidden;
    }
    #messages { overflow-y: auto; scroll-behavior: auto; padding: 24px 24px 8px; }
    .messages-inner { max-width: 760px; margin: 0 auto; }
    #messages-bottom { height: 1px; }

    /* Turns — borderless, ChatGPT/Open-WebUI style */
    .turn { padding: 14px 0; }
    .turn-user { display: flex; justify-content: flex-end; }
    .turn-user .turn-body {
      max-width: 85%;
      background: var(--user-tint);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 16px;
      white-space: pre-wrap;
    }
    .turn-assistant {
      display: grid;
      grid-template-columns: 32px 1fr;
      gap: 14px; align-items: start;
    }
    .avatar {
      width: 32px; height: 32px;
      border-radius: 9px;
      background: linear-gradient(135deg, #a5b4fc, var(--accent));
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: 800; font-size: 13px;
      letter-spacing: 0.5px; flex-shrink: 0; margin-top: 2px;
    }
    .turn-assistant .turn-body { min-width: 0; }

    /* Reasoning is its own card ABOVE the answer, not nested inside */
    .reasoning-block {
      margin: 0 0 12px;
      background: var(--reason-bg);
      border: 1px solid var(--reason-border);
      border-radius: 10px;
      overflow: hidden;
    }
    .reasoning-block[hidden] { display: none; }
    .reasoning-summary {
      cursor: pointer;
      padding: 10px 14px;
      color: var(--reason-text);
      font-size: 12px; font-weight: 600;
      letter-spacing: 0.2px;
      display: flex; align-items: center; gap: 8px;
      user-select: none;
    }
    .reasoning-summary .chev {
      width: 10px; height: 10px;
      transition: transform 0.15s;
      opacity: 0.75;
    }
    .reasoning-block.open .reasoning-summary .chev { transform: rotate(90deg); }
    .reasoning-summary .label { flex: 1; }
    .reasoning-summary .meta { color: var(--muted-2); font-size: 11px; font-weight: 500; }
    .reasoning-body {
      padding: 0 14px 12px;
      color: var(--reason-text);
      font-size: 13px; line-height: 1.6;
      white-space: pre-wrap;
      max-height: 320px; overflow-y: auto;
    }
    .reasoning-block:not(.open) .reasoning-body { display: none; }

    /* Answer body — markdown rendered */
    .answer { color: var(--text); }
    .answer.streaming-plain { white-space: pre-wrap; }
    .answer p { margin: 0 0 0.75em; }
    .answer p:last-child { margin-bottom: 0; }
    .answer h1, .answer h2, .answer h3, .answer h4 { margin: 1em 0 0.5em; line-height: 1.3; font-weight: 700; }
    .answer h1 { font-size: 1.5em; }
    .answer h2 { font-size: 1.3em; }
    .answer h3 { font-size: 1.13em; }
    .answer ul, .answer ol { padding-left: 1.4em; margin: 0.4em 0 0.8em; }
    .answer li { margin: 0.2em 0; }
    .answer a { color: var(--accent-strong); text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
    .answer a:hover { color: var(--accent); }
    .answer blockquote {
      border-left: 3px solid var(--line-strong);
      margin: 0.6em 0; padding: 0.2em 0 0.2em 1em;
      color: var(--muted);
    }
    .answer code {
      font-family: var(--font-mono); font-size: 0.92em;
      background: var(--code-bg);
      border: 1px solid var(--code-border);
      border-radius: 4px;
      padding: 1.5px 6px;
    }
    .answer pre {
      position: relative;
      margin: 0.9em 0; padding: 0;
      border: 1px solid var(--code-border);
      border-radius: 10px;
      background: var(--code-bg);
      overflow: hidden;
    }
    .answer pre code {
      display: block;
      padding: 12px 14px;
      font-size: 13px; line-height: 1.6;
      overflow-x: auto;
      border: 0; background: transparent; border-radius: 0;
      white-space: pre;
    }
    .copy-btn {
      position: absolute; top: 7px; right: 7px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px; font-weight: 500;
      padding: 3px 9px; border-radius: 5px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.12s, color 0.12s, border-color 0.12s;
    }
    .answer pre:hover .copy-btn { opacity: 1; }
    .copy-btn:hover { color: var(--text); border-color: var(--line-strong); }
    .copy-btn.copied { color: var(--ok); border-color: rgba(95, 210, 139, 0.4); }

    /* Stats below the answer — minimal inline pills, no border boxes */
    .stats {
      margin-top: 10px;
      display: flex; flex-wrap: wrap;
      gap: 4px 12px;
      font-size: 12px;
      color: var(--muted-2);
      font-variant-numeric: tabular-nums;
    }
    .stats[hidden] { display: none; }
    .stats .stat-tps { color: var(--ok); font-weight: 600; }
    .stats .stat-ttft { color: var(--muted); }

    /* Composer */
    .composer-wrap {
      padding: 12px 24px 18px;
      background: linear-gradient(to top, var(--bg) 0%, var(--bg) 75%, transparent);
    }
    .composer { max-width: 760px; margin: 0 auto; position: relative; }
    #status-row {
      min-height: 18px;
      display: flex; align-items: center; gap: 8px;
      font-size: 12px; color: var(--muted-2);
      padding: 0 4px 6px;
    }
    #status-row.ready { color: var(--muted); }
    #status-row.streaming { color: var(--accent-strong); }
    #status-row .dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: currentColor; flex-shrink: 0;
    }
    #live-stats { margin-left: auto; color: var(--muted-2); font-variant-numeric: tabular-nums; }
    #live-stats .tps { color: var(--ok); font-weight: 600; }

    .composer-box {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px; align-items: end;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 8px 8px 8px 14px;
      transition: border-color 0.12s, box-shadow 0.12s;
    }
    .composer-box:focus-within { border-color: var(--line-strong); }
    #prompt {
      min-height: 38px; max-height: 220px;
      resize: none;
      border: 0;
      background: transparent;
      color: var(--text);
      padding: 8px 0;
      font: inherit; font-size: 15px; line-height: 1.5;
      outline: none;
    }
    #prompt::placeholder { color: var(--muted-2); }
    .send-btn {
      width: 36px; height: 36px;
      border: 0; border-radius: 9px;
      background: var(--accent); color: white;
      cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      transition: background 0.12s, transform 0.05s;
    }
    .send-btn:hover { background: var(--accent-strong); }
    .send-btn:active { transform: scale(0.95); }
    .send-btn:disabled { opacity: 0.45; cursor: default; }
    .send-btn svg { width: 16px; height: 16px; }
    .send-btn.stop { background: var(--error); }
    .send-btn.stop:hover { background: #f47e7e; }
    .composer-actions { display: flex; align-items: center; gap: 4px; }
    .think-pill {
      position: relative;
      display: inline-flex; align-items: center; gap: 6px;
      height: 36px; padding: 0 24px 0 10px;
      border-radius: 9px;
      color: var(--muted); font-size: 13px; white-space: nowrap;
      cursor: pointer;
      transition: background 0.12s, color 0.12s;
    }
    .think-pill[hidden] { display: none; }
    .think-pill:hover, .think-pill:focus-within { background: var(--surface-2); color: var(--text); }
    .think-pill.off { color: var(--muted-2); }
    .think-pill select {
      appearance: none; -webkit-appearance: none;
      border: 0; background: transparent; color: inherit;
      font: inherit; font-weight: 600;
      padding: 0; outline: none; cursor: pointer;
    }
    .think-pill select option { background: var(--surface); color: var(--text); }
    .think-pill::after {
      content: ""; position: absolute; right: 10px; top: 50%;
      width: 5px; height: 5px;
      border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor;
      transform: translateY(-70%) rotate(45deg);
      pointer-events: none;
    }
    @media (max-width: 480px) { .think-label { display: none; } }

    @media (max-width: 900px) {
      header.topbar { padding: 0 12px; }
      .topbar-meta { display: none; }
      .runtime-pill { max-width: 45vw; overflow: hidden; text-overflow: ellipsis; }
      #messages { padding: 18px 14px 6px; }
      .composer-wrap { padding: 10px 14px 14px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <span class="brand"><span class="logo">M</span>MTPLX</span>
    <span class="topbar-meta"><span>__MODEL__</span><span class="sep">·</span><span class="dim">__API_NOTE__</span><span class="sep">·</span><span class="dim">__SERVER_URL__/v1</span></span>
    <div class="topbar-actions">
      <span id="runtime-pill" class="runtime-pill" hidden>Runtime</span>
      <button id="sidebar-toggle" class="icon-btn" title="Toggle settings" aria-label="Toggle settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <button id="new-chat-btn" class="icon-btn" title="Start a new conversation" aria-label="New conversation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>
      </button>
    </div>
  </header>
  <aside id="sidebar">
    <div class="sb-section">
      <p class="sb-title">Sampling</p>
      <div class="sb-row">
        <label for="ctl-temp">Temperature <span class="v" id="val-temp">0.60</span></label>
        <input id="ctl-temp" type="range" min="0" max="2" step="0.05" value="0.6">
      </div>
      <div class="sb-row">
        <label for="ctl-top-p">Top P <span class="v" id="val-top-p">0.95</span></label>
        <input id="ctl-top-p" type="range" min="__TOP_P_MIN__" max="1" step="0.01" value="0.95">
      </div>
      <div class="sb-row">
        <label for="ctl-top-k">Top K <span class="v" id="val-top-k">20</span></label>
        <input id="ctl-top-k" type="range" min="__TOP_K_MIN__" max="__TOP_K_MAX__" step="1" value="20">
        <p class="help">__TOP_K_HELP__</p>
      </div>
      <div class="sb-row">
        <label for="ctl-presence">Presence Penalty <span class="v" id="val-presence">0.00</span></label>
        <input id="ctl-presence" type="range" min="0" max="2" step="0.05" value="0">
        <p class="help">__PRESENCE_HELP__</p>
      </div>
    </div>
    <div class="sb-section">
      <p class="sb-title">Speculative</p>
      <div class="sb-row switch-row">
        <label for="ctl-mtp">__SPECULATIVE_LABEL__ <span class="v" id="val-mtp">on</span></label>
        <span class="switch">
          <input id="ctl-mtp" type="checkbox" checked>
          <span class="track"></span>
          <span class="thumb"></span>
        </span>
      </div>
      <div class="sb-row">
        <label for="ctl-depth">__DEPTH_LABEL__ <span class="v" id="val-depth">__DEPTH_VALUE__</span></label>
        <input id="ctl-depth" type="range" min="1" max="__DEPTH_MAX__" step="1" value="__DEPTH_VALUE__">
        <p class="help">__DEPTH_HELP__</p>
      </div>
    </div>
    <div class="sb-section">
      <p class="sb-title">Output</p>
      <div class="sb-row">
        <label for="ctl-max-tokens">Max tokens <span class="v" id="val-max-tokens">8k</span></label>
        <input id="ctl-max-tokens" type="range" min="256" max="32768" step="256" value="8192">
        <p class="help" id="max-tokens-help">Detecting context length…</p>
      </div>
      <div class="sb-row">
        <label for="ctl-think">Reasoning</label>
        <select id="ctl-think">
          <option value="auto" selected>Auto</option>
          <option value="on">Always show thinking</option>
          <option value="off">Hide thinking</option>
        </select>
      </div>
    </div>
    <div class="sb-section">
      <p class="sb-title">System prompt</p>
      <div class="sb-row">
        <textarea id="ctl-system" placeholder="Optional. Overrides the model default."></textarea>
      </div>
    </div>
    <div class="sb-actions">
      <button id="reset-defaults" class="sb-btn" type="button">Reset to defaults</button>
    </div>
  </aside>
  <main class="chat-area">
    <section id="messages" aria-live="polite">
      <div class="messages-inner">
        <div class="turn turn-assistant turn-greeting">
          <div class="avatar">M</div>
          <div class="turn-body">
            <div class="answer"><p>Ready when you are. Settings mirror the running MTPLX app.</p></div>
          </div>
        </div>
      </div>
      <div id="messages-bottom" aria-hidden="true"></div>
    </section>
    <div class="composer-wrap">
      <div class="composer">
        <div id="status-row" class="ready" role="status">
          <span class="dot"></span>
          <span id="status-text">Ready</span>
          <span id="live-stats"></span>
        </div>
        <form id="chat-form" autocomplete="off">
          <div class="composer-box">
            <textarea id="prompt" placeholder="Message MTPLX (Enter to send · Shift+Enter for newline)" rows="1" autofocus></textarea>
            <div class="composer-actions">
              <label id="think-pill" class="think-pill" title="How much the model thinks before it answers" hidden>
                <span class="think-label">Thinking</span>
                <select id="composer-think" aria-label="Thinking"></select>
              </label>
              <button id="send" class="send-btn" type="submit" aria-label="Send">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js" defer></script>
  <script>
    "use strict";
    const MODEL_ID = __MODEL_JSON__;
    const messagesEl = document.getElementById("messages");
    const messagesInner = messagesEl.querySelector(".messages-inner");
    const messagesBottom = document.getElementById("messages-bottom");
    const form = document.getElementById("chat-form");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send");
    const statusRow = document.getElementById("status-row");
    const statusText = document.getElementById("status-text");
    const liveStatsEl = document.getElementById("live-stats");
    const newChatBtn = document.getElementById("new-chat-btn");
    const sidebarToggleBtn = document.getElementById("sidebar-toggle");
    const sidebarEl = document.getElementById("sidebar");
    const runtimePillEl = document.getElementById("runtime-pill");
    const history = [];
    let activeAbort = null;
    let pinnedToBottom = true;
    let forceAutoScroll = false;
    let scrollFrame = null;
    let postLayoutScrollTimer = null;

    const SVG_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>';
    const SVG_STOP = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

    // ---------- settings ------------------------------------------------------
    const SETTINGS_KEY = "mtplx.chat.settings.v5:" + MODEL_ID;
    const LEGACY_SETTINGS_KEY = "mtplx.chat.settings.v4";
    const DEFAULTS = __DEFAULT_SETTINGS_JSON__;
    // Settings the running engine fixes, with the value it uses. Shown for
    // parity, never editable, and never taken from a server payload.
    const LOCKED_CONTROLS = __LOCKED_CONTROLS_JSON__;
    // RANGES is mutable so we can rewrite max_tokens.max after we discover
    // the model's real context window via /health. Hardcoding a 32768 cap
    // (our previous default) lied about a 256k-context model and stopped
    // users from raising the answer budget for long replies.
    const RANGES = {
      temperature: {min: 0, max: 2},
      top_p: {min: __TOP_P_MIN__, max: 1},
      top_k: {min: __TOP_K_MIN__, max: __TOP_K_MAX__},
      presence_penalty: {min: 0, max: 2},
      depth: {min: 1, max: __DEPTH_MAX__},
      max_tokens: {min: 256, max: 32768}
    };
    const maxTokensHelpEl = document.getElementById("max-tokens-help");
    function formatTokens(n) {
      if (n >= 1000) return (n / 1000).toFixed(1).replace(/\\.0$/, "") + "k";
      return String(n);
    }
    function runtimeLabelFromHealth(health) {
      if (health && health.runtime_mode) return String(health.runtime_mode);
      const profileName = health && health.profile && health.profile.name ? String(health.profile.name) : "";
      const mode = String((health && health.generation_mode) || "").toLowerCase() === "ar" ? "AR" : "MTP";
      const fanBoost = Boolean(health && (health.fan_boost_active || health.fan_mode === "max"));
      if (profileName === "sustained" && fanBoost) return "Sustained Max " + mode;
      if (profileName === "sustained") return "Sustained " + mode;
      if (profileName === "performance-cold" && fanBoost) return "Burst " + mode;
      if (profileName === "performance-cold") return "Performance-cold " + mode;
      if (profileName === "stable") return "Stable " + mode;
      return profileName ? profileName + " " + mode : mode;
    }
    function applyRuntimeHealth(health) {
      if (!runtimePillEl || !health) return;
      runtimePillEl.textContent = runtimeLabelFromHealth(health);
      runtimePillEl.hidden = false;
      runtimePillEl.title = runtimePillEl.textContent;
    }
    async function discoverServerLimits() {
      try {
        const res = await fetch("/health", {cache: "no-store"});
        if (!res.ok) throw new Error("health " + res.status);
        const health = await res.json();
        applyRuntimeHealth(health);
        const ctx = parseInt(health.context_window, 10);
        const serverCap = parseInt(health.max_response_tokens, 10);
        if (Number.isFinite(ctx) && ctx > 0) {
          // Allow up to (context - some headroom) tokens of output. We leave
          // 4k for the prompt so users typing a long question don't get a
          // 400 from the server even at the slider's max.
          let cap = Math.max(1024, ctx - 4096);
          if (Number.isFinite(serverCap) && serverCap > 0) cap = Math.min(cap, serverCap);
          // Round down to a clean step.
          cap = Math.floor(cap / 256) * 256;
          RANGES.max_tokens.max = cap;
          if (ctlEls.max_tokens) {
            ctlEls.max_tokens.max = String(cap);
            // Re-clamp the current value so a saved value above the new cap
            // doesn't render off the right edge of the slider.
            ctlEls.max_tokens.value = String(Math.min(parseInt(ctlEls.max_tokens.value, 10) || DEFAULTS.max_tokens, cap));
          }
          if (maxTokensHelpEl) {
            // ctx is /health's context_window: what THIS launch is serving,
            // which memory sizing may cap well below the model's native
            // context. Labeling it "the model's" misled low-RAM users into
            // reading the machine cap as a model property.
            maxTokensHelpEl.textContent =
              "Cap is this server's " + formatTokens(ctx) + " context window (slider tops out at " +
              formatTokens(cap) + ").";
          }
          refreshLabels();
          refreshSliderFills();
        }
      } catch (err) {
        if (maxTokensHelpEl) maxTokensHelpEl.textContent =
          "Could not detect context length; using a 32k default.";
      }
    }
    const ctlEls = {
      temperature: document.getElementById("ctl-temp"),
      top_p: document.getElementById("ctl-top-p"),
      top_k: document.getElementById("ctl-top-k"),
      presence_penalty: document.getElementById("ctl-presence"),
      mtp_enabled: document.getElementById("ctl-mtp"),
      depth: document.getElementById("ctl-depth"),
      max_tokens: document.getElementById("ctl-max-tokens"),
      reasoning: document.getElementById("ctl-think"),
      system: document.getElementById("ctl-system")
    };
    const valEls = {
      temperature: document.getElementById("val-temp"),
      top_p: document.getElementById("val-top-p"),
      top_k: document.getElementById("val-top-k"),
      presence_penalty: document.getElementById("val-presence"),
      mtp_enabled: document.getElementById("val-mtp"),
      depth: document.getElementById("val-depth"),
      max_tokens: document.getElementById("val-max-tokens")
    };
    for (const key of Object.keys(LOCKED_CONTROLS)) {
      if (ctlEls[key]) ctlEls[key].disabled = true;
    }

    // ---------- thinking (chat bar) ------------------------------------------
    // The sidebar's reasoning setting, next to Send. A model whose template
    // takes effort levels lists them; the loaded model's policy arrives with
    // /v1/mtplx/settings, and a model without thinking hides the selector.
    const thinkPillEl = document.getElementById("think-pill");
    const composerThinkEl = document.getElementById("composer-think");
    const EFFORT_LABELS = {xhigh: "XHigh", high: "High", medium: "Medium", low: "Low"};
    let reasoningPolicy = null;
    let currentEffort = "auto";
    function effortLevels() {
      const levels = reasoningPolicy && Array.isArray(reasoningPolicy.effort_levels)
        ? reasoningPolicy.effort_levels
        : [];
      return levels.map(String);
    }
    function normalizeEffort(value) {
      const effort = String(value || "auto");
      return effortLevels().includes(effort) ? effort : "auto";
    }
    function renderThinkOptions() {
      const supported = Boolean(reasoningPolicy) && reasoningPolicy.supported !== false;
      thinkPillEl.hidden = !supported;
      if (!supported) return;
      const options = [["auto", "Auto"]];
      const levels = effortLevels();
      if (levels.length) {
        for (const level of levels) {
          options.push([level, EFFORT_LABELS[level] || level.charAt(0).toUpperCase() + level.slice(1)]);
        }
      } else {
        options.push(["on", "On"]);
      }
      options.push(["off", "Off"]);
      const signature = JSON.stringify(options);
      if (composerThinkEl.dataset.signature === signature) return;
      composerThinkEl.dataset.signature = signature;
      composerThinkEl.innerHTML = options
        .map(([value, label]) => '<option value="' + value + '">' + escapeHtml(label) + "</option>")
        .join("");
    }
    function showThinkChoice() {
      const reasoning = ctlEls.reasoning.value;
      let choice = reasoning === "on" ? "on" : "auto";
      if (reasoning === "off") choice = "off";
      else if (effortLevels().length) choice = normalizeEffort(currentEffort);
      if ([...composerThinkEl.options].some((option) => option.value === choice)) {
        composerThinkEl.value = choice;
      }
      thinkPillEl.classList.toggle("off", choice === "off");
    }
    composerThinkEl.addEventListener("change", () => {
      const choice = composerThinkEl.value;
      if (choice === "off" || choice === "on") {
        ctlEls.reasoning.value = choice;
      } else if (choice === "auto") {
        currentEffort = "auto";
        if (ctlEls.reasoning.value === "off" || !effortLevels().length) ctlEls.reasoning.value = "auto";
      } else {
        // Picking how hard to think is asking for thinking.
        currentEffort = choice;
        if (ctlEls.reasoning.value === "off") ctlEls.reasoning.value = "on";
      }
      lastLocalSettingsEditAt = performance.now();
      settings = readSettings();
      saveSettings(settings);
      scheduleDaemonSettingsSync(settings, {immediate: true});
    });
    function loadStoredSystemPrompt() {
      try {
        const raw = window.localStorage.getItem(SETTINGS_KEY);
        const fallback = window.localStorage.getItem(LEGACY_SETTINGS_KEY);
        const parsed = JSON.parse(raw || fallback || "{}");
        return parsed && typeof parsed.system === "string" ? parsed.system : "";
      } catch (err) {
        console.warn("settings prompt load failed", err);
        return "";
      }
    }
    function loadSettings() {
      return Object.assign({}, DEFAULTS, {system: loadStoredSystemPrompt()});
    }
    function settingsFromDaemonPayload(payload) {
      payload = payload || {};
      if (payload.reasoning_policy && typeof payload.reasoning_policy === "object") {
        reasoningPolicy = payload.reasoning_policy;
        renderThinkOptions();
      }
      const rawMode = payload.generation_mode == null ? "" : String(payload.generation_mode);
      const mode = rawMode.toLowerCase();
      if (payload.depth_max) {
        RANGES.depth.max = Math.max(1, parseInt(payload.depth_max, 10) || RANGES.depth.max);
        if (ctlEls.depth) ctlEls.depth.max = String(RANGES.depth.max);
      }
      return Object.assign({}, DEFAULTS, {
        temperature: payload.temperature,
        top_p: payload.top_p,
        top_k: payload.top_k,
        presence_penalty: payload.presence_penalty == null ? DEFAULTS.presence_penalty : payload.presence_penalty,
        mtp_enabled: mode ? mode === "mtp" : DEFAULTS.mtp_enabled,
        depth: payload.depth,
        max_tokens: payload.max_response_tokens == null ? DEFAULTS.max_tokens : payload.max_response_tokens,
        reasoning: payload.reasoning || DEFAULTS.reasoning,
        reasoning_effort: payload.reasoning_effort || "auto",
        system: loadStoredSystemPrompt()
      });
    }
    function daemonSettingsPayload(s) {
      const normalized = normalizeSettings(s || {});
      const payload = {
        temperature: normalized.temperature,
        top_p: normalized.top_p,
        top_k: normalized.top_k,
        presence_penalty: normalized.presence_penalty,
        generation_mode: normalized.mtp_enabled ? "mtp" : "ar",
        depth: normalized.depth,
        max_response_tokens: normalized.max_tokens,
        reasoning: normalized.reasoning
      };
      if (effortLevels().length) payload.reasoning_effort = normalized.reasoning_effort;
      return payload;
    }
    function daemonSettingsSignature(s) {
      return JSON.stringify(daemonSettingsPayload(s));
    }
    async function fetchDaemonSettings() {
      const res = await fetch("/v1/mtplx/settings", {cache: "no-store"});
      if (!res.ok) throw new Error("settings " + res.status);
      const payload = await res.json();
      return settingsFromDaemonPayload(payload);
    }
    function saveSettings(s) {
      try {
        window.localStorage.setItem(
          SETTINGS_KEY,
          JSON.stringify({system: String((s && s.system) || "")})
        );
      } catch (_e) { /* ignore quota */ }
    }
    function clamp(value, min, max, fallback, isInt) {
      const n = isInt ? parseInt(value, 10) : parseFloat(value);
      if (!Number.isFinite(n)) return fallback;
      return Math.min(max, Math.max(min, n));
    }
    function normalizeSettings(s) {
      s = s || {};
      const normalized = {
        temperature: clamp(s.temperature, RANGES.temperature.min, RANGES.temperature.max, DEFAULTS.temperature, false),
        top_p: clamp(s.top_p, RANGES.top_p.min, RANGES.top_p.max, DEFAULTS.top_p, false),
        top_k: clamp(s.top_k, RANGES.top_k.min, RANGES.top_k.max, DEFAULTS.top_k, true),
        presence_penalty: clamp(s.presence_penalty, RANGES.presence_penalty.min, RANGES.presence_penalty.max, DEFAULTS.presence_penalty, false),
        mtp_enabled: s.mtp_enabled == null ? DEFAULTS.mtp_enabled !== false : s.mtp_enabled !== false,
        depth: clamp(s.depth, RANGES.depth.min, RANGES.depth.max, DEFAULTS.depth, true),
        max_tokens: clamp(s.max_tokens, RANGES.max_tokens.min, RANGES.max_tokens.max, DEFAULTS.max_tokens, true),
        reasoning: ["auto", "on", "off"].includes(String(s.reasoning || "")) ? String(s.reasoning) : DEFAULTS.reasoning,
        reasoning_effort: normalizeEffort(s.reasoning_effort),
        system: String(s.system || "")
      };
      return Object.assign(normalized, LOCKED_CONTROLS);
    }
    function applySettingsToUI(s) {
      const normalized = normalizeSettings(s);
      ctlEls.temperature.value = normalized.temperature;
      ctlEls.top_p.value = normalized.top_p;
      ctlEls.top_k.value = normalized.top_k;
      ctlEls.presence_penalty.value = normalized.presence_penalty;
      ctlEls.mtp_enabled.checked = normalized.mtp_enabled;
      ctlEls.depth.value = normalized.depth;
      ctlEls.max_tokens.value = normalized.max_tokens;
      ctlEls.reasoning.value = normalized.reasoning;
      currentEffort = normalized.reasoning_effort;
      ctlEls.system.value = normalized.system;
      refreshLabels();
      refreshSliderFills();
    }
    let lastSyncedSettingsSignature = "";
    let settingsSyncTimer = null;
    let settingsSyncSeq = 0;
    let lastLocalSettingsEditAt = 0;
    async function refreshDaemonSettings(options) {
      const opts = options || {};
      if (!opts.force) {
        const recentlyEdited = performance.now() - lastLocalSettingsEditAt < 700;
        if (activeAbort || settingsSyncTimer || recentlyEdited) return;
      }
      try {
        const serverSettings = normalizeSettings(await fetchDaemonSettings());
        settings = Object.assign({}, serverSettings, {system: loadStoredSystemPrompt()});
        lastSyncedSettingsSignature = daemonSettingsSignature(settings);
        applySettingsToUI(settings);
        saveSettings(settings);
      } catch (err) {
        console.warn("daemon settings refresh failed", err);
      }
    }
    async function syncDaemonSettings(nextSettings, options) {
      const opts = options || {};
      const signature = daemonSettingsSignature(nextSettings);
      if (!opts.force && signature === lastSyncedSettingsSignature) return normalizeSettings(nextSettings);
      const seq = ++settingsSyncSeq;
      const response = await fetch("/v1/mtplx/settings", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(daemonSettingsPayload(nextSettings))
      });
      if (!response.ok) {
        let detail = "settings update failed: " + response.status;
        try {
          const errBody = await response.json();
          if (errBody?.error?.message) detail = errBody.error.message;
        } catch (_e) { /* ignore */ }
        throw new Error(detail);
      }
      const payload = await response.json();
      if (seq !== settingsSyncSeq) return normalizeSettings(nextSettings);
      const serverSettings = normalizeSettings(settingsFromDaemonPayload(payload));
      lastSyncedSettingsSignature = daemonSettingsSignature(serverSettings);
      return serverSettings;
    }
    function scheduleDaemonSettingsSync(nextSettings, options) {
      const opts = options || {};
      if (settingsSyncTimer) {
        clearTimeout(settingsSyncTimer);
        settingsSyncTimer = null;
      }
      settingsSyncTimer = setTimeout(() => {
        settingsSyncTimer = null;
        syncDaemonSettings(nextSettings).then((serverSettings) => {
          settings = Object.assign({}, serverSettings, {system: nextSettings.system});
          applySettingsToUI(settings);
          saveSettings(settings);
        }).catch((err) => {
          console.warn("daemon settings sync failed", err);
          setStatus("Settings update failed", "error");
        });
      }, opts.immediate ? 0 : 180);
    }
    function refreshLabels() {
      valEls.temperature.textContent = Number(ctlEls.temperature.value).toFixed(2);
      valEls.top_p.textContent = Number(ctlEls.top_p.value).toFixed(2);
      const tk = parseInt(ctlEls.top_k.value, 10) || 0;
      valEls.top_k.textContent = tk === 0 ? "off" : String(tk);
      const pp = Number(ctlEls.presence_penalty.value) || 0;
      valEls.presence_penalty.textContent = pp === 0 ? "off" : pp.toFixed(2);
      const mtpOn = Boolean(ctlEls.mtp_enabled.checked);
      valEls.mtp_enabled.textContent = mtpOn ? "on" : "off";
      ctlEls.depth.disabled = !mtpOn || "depth" in LOCKED_CONTROLS;
      valEls.depth.textContent = mtpOn ? String(parseInt(ctlEls.depth.value, 10) || 0) : "off";
      const mt = parseInt(ctlEls.max_tokens.value, 10) || 0;
      valEls.max_tokens.textContent = mt >= 1000 ? (mt / 1000).toFixed(1).replace(/\\.0$/, "") + "k" : String(mt);
      showThinkChoice();
    }
    function refreshSliderFills() {
      for (const key of ["temperature", "top_p", "top_k", "presence_penalty", "depth", "max_tokens"]) {
        const el = ctlEls[key];
        const range = RANGES[key];
        if (!el || !range) continue;
        if (key === "depth" && !ctlEls.mtp_enabled.checked) {
          el.style.setProperty("--filled", "0%");
          continue;
        }
        const value = Number(el.value);
        const span = range.max - range.min;
        const pct = span > 0 ? ((value - range.min) / span) * 100 : 100;
        el.style.setProperty("--filled", pct.toFixed(2) + "%");
      }
    }
    function readSettings() {
      const s = {
        temperature: clamp(ctlEls.temperature.value, RANGES.temperature.min, RANGES.temperature.max, DEFAULTS.temperature, false),
        top_p: clamp(ctlEls.top_p.value, RANGES.top_p.min, RANGES.top_p.max, DEFAULTS.top_p, false),
        top_k: clamp(ctlEls.top_k.value, RANGES.top_k.min, RANGES.top_k.max, DEFAULTS.top_k, true),
        presence_penalty: clamp(ctlEls.presence_penalty.value, RANGES.presence_penalty.min, RANGES.presence_penalty.max, DEFAULTS.presence_penalty, false),
        mtp_enabled: Boolean(ctlEls.mtp_enabled.checked),
        depth: clamp(ctlEls.depth.value, RANGES.depth.min, RANGES.depth.max, DEFAULTS.depth, true),
        max_tokens: clamp(ctlEls.max_tokens.value, RANGES.max_tokens.min, RANGES.max_tokens.max, DEFAULTS.max_tokens, true),
        reasoning: ctlEls.reasoning.value || "auto",
        reasoning_effort: normalizeEffort(currentEffort),
        system: (ctlEls.system.value || "").trim()
      };
      refreshLabels();
      refreshSliderFills();
      return s;
    }
    let settings = loadSettings();
    applySettingsToUI(settings);
    fetchDaemonSettings().catch((err) => {
      console.warn("daemon settings hydrate failed", err);
      return loadSettings();
    }).then((serverSettings) => {
      settings = normalizeSettings(serverSettings);
      lastSyncedSettingsSignature = daemonSettingsSignature(settings);
      applySettingsToUI(settings);
      saveSettings(settings);
      return discoverServerLimits();
    }).then(() => {
      // Re-clamp + redraw after the real context window arrives so users
      // who reload the page don't see "8k" sitting under a fresh 256k cap.
      settings = readSettings();
      saveSettings(settings);
    });
    window.setInterval(() => refreshDaemonSettings(), 1500);
    window.addEventListener("focus", () => refreshDaemonSettings({force: true}));
    function handleSettingsControlEdit(key) {
      return () => {
        lastLocalSettingsEditAt = performance.now();
        settings = readSettings();
        saveSettings(settings);
        if (key !== "system") {
          scheduleDaemonSettingsSync(settings, {immediate: key === "mtp_enabled" || key === "reasoning"});
        }
      };
    }
    for (const key of Object.keys(ctlEls)) {
      const handler = handleSettingsControlEdit(key);
      ctlEls[key].addEventListener("input", handler);
      ctlEls[key].addEventListener("change", handler);
    }
    document.getElementById("reset-defaults").addEventListener("click", () => {
      settings = Object.assign({}, DEFAULTS);
      lastLocalSettingsEditAt = performance.now();
      applySettingsToUI(settings);
      saveSettings(settings);
      scheduleDaemonSettingsSync(settings, {immediate: true});
    });
    sidebarToggleBtn.addEventListener("click", () => sidebarEl.classList.toggle("open"));

    // ---------- markdown ------------------------------------------------------
    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }
    function renderMarkdown(text) {
      try {
        if (typeof window.marked !== "undefined") {
          window.marked.setOptions({
            gfm: true,
            breaks: true,
            mangle: false,
            headerIds: false
          });
          return window.marked.parse(String(text || ""));
        }
      } catch (err) {
        console.warn("markdown parse failed; falling back to text", err);
      }
      // Fallback: escape and convert simple newlines to <br>
      return escapeHtml(text).replace(/\\n/g, "<br>");
    }
    function attachCopyButtons(scope) {
      for (const pre of scope.querySelectorAll("pre")) {
        if (pre.querySelector(".copy-btn")) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "copy-btn";
        btn.textContent = "Copy";
        btn.addEventListener("click", async () => {
          const code = pre.querySelector("code");
          const text = code ? code.textContent || "" : pre.textContent || "";
          try {
            await navigator.clipboard.writeText(text);
            btn.textContent = "Copied";
            btn.classList.add("copied");
            setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1400);
          } catch (err) { btn.textContent = "Error"; }
        });
        pre.appendChild(btn);
      }
    }

    // ---------- scroll handling ----------------------------------------------
    const SCROLL_PIN_THRESHOLD = 160;
    function isPinned() {
      const remaining = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      return remaining <= SCROLL_PIN_THRESHOLD;
    }
    messagesEl.addEventListener("scroll", () => {
      if (forceAutoScroll) return;
      pinnedToBottom = isPinned();
    });
    function scrollToBottom(opts) {
      const options = opts || {};
      if (!options.force && !forceAutoScroll && !pinnedToBottom) return;
      messagesEl.scrollTop = messagesEl.scrollHeight;
      if (messagesBottom && messagesBottom.scrollIntoView) {
        messagesBottom.scrollIntoView({block: "end", inline: "nearest", behavior: "auto"});
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
      pinnedToBottom = true;
    }
    function scheduleScrollToBottom(opts) {
      const options = opts || {};
      if (!options.force && !forceAutoScroll && !pinnedToBottom) return;
      if (options.force) pinnedToBottom = true;
      if (postLayoutScrollTimer) {
        clearTimeout(postLayoutScrollTimer);
        postLayoutScrollTimer = null;
      }
      if (scrollFrame !== null) return;
      scrollFrame = requestAnimationFrame(() => {
        scrollFrame = null;
        scrollToBottom({force: options.force});
        // Streamed code blocks and final markdown can grow after the text node
        // update that triggered this scroll. Recheck on the next frame and
        // once more after layout settles so coding output keeps following.
        requestAnimationFrame(() => scrollToBottom({force: options.force}));
        postLayoutScrollTimer = setTimeout(() => {
          postLayoutScrollTimer = null;
          scrollToBottom({force: options.force});
        }, 40);
      });
    }
    if (window.ResizeObserver) {
      const scrollObserver = new ResizeObserver(() => scheduleScrollToBottom());
      scrollObserver.observe(messagesInner);
    }

    // ---------- DOM helpers ---------------------------------------------------
    function setStatus(text, kind) {
      statusText.textContent = text;
      statusRow.className = kind || "";
    }
    function appendUser(text) {
      const node = document.createElement("div");
      node.className = "turn turn-user";
      const body = document.createElement("div");
      body.className = "turn-body";
      body.textContent = text;
      node.appendChild(body);
      messagesInner.appendChild(node);
      scheduleScrollToBottom();
    }
    function appendAssistantTurn() {
      const node = document.createElement("div");
      node.className = "turn turn-assistant";
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = "M";
      const body = document.createElement("div");
      body.className = "turn-body";

      const reasoningBlock = document.createElement("div");
      reasoningBlock.className = "reasoning-block";
      reasoningBlock.hidden = true;
      const reasoningSummary = document.createElement("div");
      reasoningSummary.className = "reasoning-summary";
      reasoningSummary.innerHTML =
        '<svg class="chev" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2l4 3-4 3"/></svg>' +
        '<span class="label">Thinking</span>' +
        '<span class="meta"></span>';
      const reasoningBody = document.createElement("div");
      reasoningBody.className = "reasoning-body";
      reasoningBlock.appendChild(reasoningSummary);
      reasoningBlock.appendChild(reasoningBody);
      reasoningSummary.addEventListener("click", () => reasoningBlock.classList.toggle("open"));

      const answerBody = document.createElement("div");
      answerBody.className = "answer streaming-plain";

      const stats = document.createElement("div");
      stats.className = "stats";
      stats.hidden = true;

      body.appendChild(reasoningBlock);
      body.appendChild(answerBody);
      body.appendChild(stats);
      node.appendChild(avatar);
      node.appendChild(body);
      messagesInner.appendChild(node);
      scheduleScrollToBottom();
      return {node, reasoningBlock, reasoningSummary, reasoningBody, answerBody, stats};
    }
    function setReasoningMeta(turn, text) {
      const meta = turn.reasoningSummary.querySelector(".meta");
      if (meta) meta.textContent = text;
    }
    function renderStats(statsEl, stats) {
      if (!stats) return;
      const verifyMs = stats.verify_calls ? (1000 * Number(stats.verify_time_s || 0) / Number(stats.verify_calls)) : null;
      const tps = Number(stats.decode_tok_s ?? stats.request_tok_s);
      const generationMode = String(stats.generation_mode || "").toLowerCase();
      const mtpDepth = Number(stats.mtp_depth ?? stats.speculative_depth);
      const parts = [];
      if (generationMode === "ar") parts.push("AR");
      else if (generationMode === "mtp") parts.push("MTP depth " + (Number.isFinite(mtpDepth) ? mtpDepth : "?"));
      else if (generationMode === "splash") {
        const accepted = Number(stats.draft_acceptance_rate);
        parts.push("DFlash 2" + (Number.isFinite(accepted) ? " " + Math.round(accepted * 100) + "% accepted" : ""));
      }
      if (Number.isFinite(tps)) parts.push('<span class="stat-tps">' + tps.toFixed(1) + ' tok/s</span>');
      if (Number.isFinite(Number(stats.completion_tokens))) parts.push(Number(stats.completion_tokens) + ' tokens');
      if (Number.isFinite(Number(stats.reasoning_tokens)) && Number(stats.reasoning_tokens) > 0) parts.push(Number(stats.reasoning_tokens) + ' thinking');
      if (Number.isFinite(Number(stats.ttft_s))) parts.push('<span class="stat-ttft">ttft ' + Number(stats.ttft_s).toFixed(2) + 's</span>');
      if (Number.isFinite(verifyMs)) parts.push(verifyMs.toFixed(1) + ' ms/verify');
      if (Number.isFinite(Number(stats.verify_calls))) parts.push(Number(stats.verify_calls) + ' verifies');
      statsEl.innerHTML = parts.join(' <span style="color:var(--muted-2)">·</span> ');
      statsEl.hidden = parts.length === 0;
    }
    function renderLiveStats(state) {
      const explicitTps = Number(state.tps);
      const tps = Number.isFinite(explicitTps)
        ? explicitTps
        : (state.tokens > 0 && state.elapsed > 0 ? (state.tokens / state.elapsed) : null);
      const parts = [];
      if (tps !== null) parts.push('<span class="tps">' + tps.toFixed(1) + ' tok/s</span>');
      if (state.tokens > 0) parts.push(state.tokens + ' tokens');
      liveStatsEl.innerHTML = parts.length ? '· ' + parts.join(' · ') : '';
    }
    function applyProgressToLiveState(liveState, progress) {
      if (!progress) return;
      const tokens = Number(progress.completion_tokens ?? progress.generated_tokens);
      const elapsed = Number(progress.decode_elapsed_s ?? progress.elapsed_s);
      const tps = Number(progress.decode_tok_s ?? progress.tok_s);
      if (Number.isFinite(tokens) && tokens >= 0) liveState.tokens = tokens;
      if (Number.isFinite(elapsed) && elapsed >= 0) liveState.elapsed = elapsed;
      if (Number.isFinite(tps) && tps >= 0) liveState.tps = tps;
      liveState.hasServerProgress = true;
      renderLiveStats(liveState);
    }
    function applyFinalStatsToLiveState(liveState, stats) {
      if (!stats) return;
      const tokens = Number(stats.completion_tokens ?? stats.generated_tokens);
      const elapsed = Number(stats.decode_elapsed_s ?? stats.request_elapsed_s ?? stats.elapsed_s);
      const tps = Number(stats.decode_tok_s ?? stats.request_tok_s ?? stats.tok_s);
      if (Number.isFinite(tokens) && tokens >= 0) liveState.tokens = tokens;
      if (Number.isFinite(elapsed) && elapsed >= 0) liveState.elapsed = elapsed;
      if (Number.isFinite(tps) && tps >= 0) liveState.tps = tps;
      liveState.hasServerProgress = true;
      renderLiveStats(liveState);
    }

    // ---------- streaming -----------------------------------------------------
    function splitFallbackReasoning(reasoningText, answerText) {
      if (answerText.trim() || !reasoningText.trim()) return {reasoningText, answerText};
      const blocks = reasoningText.trim().split(/\\n\\s*\\n/).filter(Boolean);
      if (blocks.length < 2) return {reasoningText, answerText};
      const answer = blocks[blocks.length - 1].trim();
      const reasoning = blocks.slice(0, -1).join("\\n\\n").trim();
      return {reasoningText: reasoning, answerText: answer};
    }
    function drainSse(buffer, onPayload) {
      let offset = 0;
      while (true) {
        const index = buffer.indexOf("\\n\\n", offset);
        if (index < 0) break;
        const eventText = buffer.slice(offset, index);
        offset = index + 2;
        for (const line of eventText.split("\\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;
          try { onPayload(JSON.parse(data)); } catch (err) { console.warn(err, data); }
        }
      }
      return buffer.slice(offset);
    }
    function setSendButton(mode) {
      if (mode === "stop") {
        sendBtn.classList.add("stop");
        sendBtn.dataset.action = "stop";
        sendBtn.innerHTML = SVG_STOP;
        sendBtn.setAttribute("aria-label", "Stop generation");
      } else {
        sendBtn.classList.remove("stop");
        sendBtn.dataset.action = "send";
        sendBtn.innerHTML = SVG_SEND;
        sendBtn.setAttribute("aria-label", "Send");
      }
    }
    async function sendMessage(text) {
      // Remove the greeting bubble on first send.
      const greeting = messagesInner.querySelector(".turn-greeting");
      if (greeting) greeting.remove();

      pinnedToBottom = true;
      forceAutoScroll = true;
      appendUser(text);
      history.push({role: "user", content: text});
      const turn = appendAssistantTurn();
      scheduleScrollToBottom({force: true});
      setSendButton("stop");
      promptEl.disabled = false;
      setStatus("Thinking", "streaming");

      activeAbort = new AbortController();
      let assistantText = "";
      let reasoningText = "";
      let finalStats = null;
      let firstTokenAt = null;
      const startedAt = performance.now();
      const liveState = {tokens: 0, elapsed: 0, tps: null, hasServerProgress: false};
      // Transport watchdog: normal long generations should receive server
      // heartbeat/progress frames. Abort only when the stream itself goes
      // quiet, not when the model is still actively working.
      let lastChunkAt = performance.now();
      const STALL_WARN_MS = 30000;
      const STALL_ABORT_MS = 90000;
      let stallTimer = null;
      function armStallWatchdog() {
        if (stallTimer) clearInterval(stallTimer);
        stallTimer = setInterval(() => {
          const idle = performance.now() - lastChunkAt;
          if (idle > STALL_ABORT_MS) {
            clearInterval(stallTimer);
            stallTimer = null;
            try { activeAbort && activeAbort.abort("stalled"); } catch (_e) {}
          } else if (idle > STALL_WARN_MS) {
            setStatus("Waiting for stream data (" + Math.round(idle / 1000) + "s)", "streaming");
          }
        }, 1000);
      }
      function disarmStallWatchdog() {
        if (stallTimer) { clearInterval(stallTimer); stallTimer = null; }
      }
      try {
        let settingsNow = readSettings();
        if (settingsSyncTimer) {
          clearTimeout(settingsSyncTimer);
          settingsSyncTimer = null;
        }
        const serverSettings = await syncDaemonSettings(settingsNow);
        settingsNow = Object.assign({}, serverSettings, {system: settingsNow.system});
        settings = settingsNow;
        applySettingsToUI(settings);
        saveSettings(settings);
        const messages = settingsNow.system
          ? [{role: "system", content: settingsNow.system}, ...history]
          : history.slice();
        const requestBody = {
          model: MODEL_ID,
          messages,
          stream: true,
          temperature: settingsNow.temperature,
          top_p: settingsNow.top_p,
          generation_mode: settingsNow.mtp_enabled ? "mtp" : "ar",
          depth: settingsNow.depth,
          max_tokens: settingsNow.max_tokens
        };
        if (settingsNow.top_k > 0) requestBody.top_k = settingsNow.top_k;
        if (settingsNow.reasoning === "on") requestBody.enable_thinking = true;
        else if (settingsNow.reasoning === "off") requestBody.enable_thinking = false;
        if (settingsNow.reasoning !== "off" && effortLevels().includes(settingsNow.reasoning_effort)) {
          requestBody.reasoning_effort = settingsNow.reasoning_effort;
        }

        armStallWatchdog();
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-MTPLX-Client": "mtplx_browser",
            "X-MTPLX-Allow-Client-Controls": "1"
          },
          body: JSON.stringify(requestBody),
          signal: activeAbort.signal
        });
        if (!response.ok || !response.body) {
          let detail = "Request failed: " + response.status;
          try {
            const errBody = await response.json();
            if (errBody?.error?.message) detail = errBody.error.message;
          } catch (_e) { /* ignore */ }
          throw new Error(detail);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          lastChunkAt = performance.now();
          buffer += decoder.decode(value, {stream: true});
          buffer = drainSse(buffer, (payload) => {
            if (payload.error) throw new Error(payload.error.message || "generation failed");
            if (payload.mtplx_progress) {
              applyProgressToLiveState(liveState, payload.mtplx_progress);
              if (payload.mtplx_progress.heartbeat) {
                setStatus("Still working", "streaming");
              } else {
                setStatus("Thinking", "streaming");
              }
            }
            if (payload.mtplx_stats) {
              finalStats = payload.mtplx_stats;
              renderStats(turn.stats, finalStats);
              applyFinalStatsToLiveState(liveState, finalStats);
            }
            const delta = payload.choices?.[0]?.delta || {};
            const reasoningPiece = delta.reasoning_content || "";
            const answerPiece = delta.content || "";
            if (reasoningPiece) {
              reasoningText += reasoningPiece;
              turn.reasoningBody.textContent = reasoningText;
              if (turn.reasoningBlock.hidden) {
                turn.reasoningBlock.hidden = false;
                turn.reasoningBlock.classList.add("open");
              }
              setReasoningMeta(turn, "streaming…");
              setStatus("Thinking", "streaming");
            }
            if (answerPiece) {
              if (firstTokenAt === null) firstTokenAt = performance.now();
              assistantText += answerPiece;
              turn.answerBody.textContent = assistantText;
              if (!liveState.hasServerProgress) {
                liveState.tokens += 1;
                liveState.elapsed = (performance.now() - startedAt) / 1000;
                liveState.tps = null;
                renderLiveStats(liveState);
              }
              setStatus("Streaming", "streaming");
              // Auto-collapse the reasoning block once the answer starts (still expandable).
              if (!turn.reasoningBlock.hidden) {
                turn.reasoningBlock.classList.remove("open");
                setReasoningMeta(turn, "click to expand");
              }
            }
            scheduleScrollToBottom({force: true});
          });
        }
        const separated = splitFallbackReasoning(reasoningText, assistantText);
        reasoningText = separated.reasoningText;
        assistantText = separated.answerText;
        turn.reasoningBody.textContent = reasoningText;
        turn.reasoningBlock.hidden = !reasoningText.trim();
        if (!turn.reasoningBlock.hidden) setReasoningMeta(turn, "click to expand");
        const finalText = assistantText.trim() ? assistantText : "(No text returned.)";
        turn.answerBody.classList.remove("streaming-plain");
        turn.answerBody.innerHTML = renderMarkdown(finalText);
        attachCopyButtons(turn.answerBody);
        if (finalStats) renderStats(turn.stats, finalStats);
        history.push({role: "assistant", content: assistantText});
        setStatus("Ready", "ready");
        renderLiveStats(liveState);
      } catch (err) {
        const aborted = err && (err.name === "AbortError" || /aborted/i.test(String(err.message || "")));
        const stalled =
          aborted &&
          (performance.now() - lastChunkAt) > STALL_ABORT_MS;
        if (stalled) {
          turn.answerBody.classList.remove("streaming-plain");
          turn.answerBody.innerHTML = renderMarkdown(
            "*[stream connection went quiet for " + Math.round(STALL_ABORT_MS / 1000) + "s - request stopped]*\\n\\n" +
              "The browser did not receive stream data from MTPLX. Check the server terminal, " +
              "then retry or restart only if the MTPLX process has exited."
          );
          attachCopyButtons(turn.answerBody);
          setStatus("Stream disconnected", "");
        } else if (aborted) {
          turn.answerBody.classList.remove("streaming-plain");
          turn.answerBody.innerHTML = renderMarkdown(assistantText || "*[stopped]*");
          attachCopyButtons(turn.answerBody);
          history.push({role: "assistant", content: assistantText});
          setStatus("Stopped", "ready");
        } else {
          turn.answerBody.textContent = "Error: " + (err?.message || String(err));
          setStatus("Error", "");
        }
      } finally {
        disarmStallWatchdog();
        activeAbort = null;
        setSendButton("send");
        promptEl.disabled = false;
        promptEl.focus();
        autoResizePrompt();
        scheduleScrollToBottom({force: true});
        forceAutoScroll = false;
      }
    }

    // ---------- form wiring ---------------------------------------------------
    function autoResizePrompt() {
      promptEl.style.height = "auto";
      promptEl.style.height = Math.min(promptEl.scrollHeight, 220) + "px";
    }
    promptEl.addEventListener("input", autoResizePrompt);

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (sendBtn.dataset.action === "stop") {
        if (activeAbort) activeAbort.abort();
        return;
      }
      const text = promptEl.value.trim();
      if (!text) return;
      promptEl.value = "";
      autoResizePrompt();
      sendMessage(text);
    });
    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    newChatBtn.addEventListener("click", () => {
      if (activeAbort) activeAbort.abort();
      history.length = 0;
      messagesInner.innerHTML = '<div class="turn turn-assistant turn-greeting"><div class="avatar">M</div><div class="turn-body"><div class="answer"><p>New conversation. Settings on the left are unchanged.</p></div></div></div>';
      pinnedToBottom = true;
      forceAutoScroll = false;
      setStatus("Ready", "ready");
      renderLiveStats({tokens: 0, elapsed: 0});
      promptEl.focus();
      autoResizePrompt();
      scheduleScrollToBottom({force: true});
    });

    setSendButton("send");
    autoResizePrompt();
  </script>
</body>
</html>"""
    return (
        template.replace("__MODEL__", html.escape(model_id))
        .replace("__API_NOTE__", html.escape(api_note))
        .replace("__SERVER_URL__", html.escape(server_url))
        .replace("__MODEL_JSON__", json.dumps(model_id))
        .replace(
            "__DEFAULT_SETTINGS_JSON__", json.dumps(default_settings, sort_keys=True)
        )
        .replace("__DEPTH_VALUE__", str(default_depth))
        .replace("__DEPTH_MAX__", str(depth_max))
        .replace("__TOP_P_MIN__", json.dumps(ui["top_p_min"]))
        .replace("__TOP_K_MIN__", json.dumps(ui["top_k_min"]))
        .replace("__TOP_K_MAX__", json.dumps(ui["top_k_max"]))
        .replace("__TOP_K_HELP__", ui["top_k_help"])
        .replace("__PRESENCE_HELP__", ui["presence_help"])
        .replace("__SPECULATIVE_LABEL__", ui["speculative_label"])
        .replace("__DEPTH_LABEL__", ui["depth_label"])
        .replace("__DEPTH_HELP__", ui["depth_help"])
        .replace(
            "__LOCKED_CONTROLS_JSON__", json.dumps(ui["locked_controls"], sort_keys=True)
        )
    )
