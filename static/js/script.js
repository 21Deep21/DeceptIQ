"use strict";

/* =============================================================
   Phishing URL Analyzer — SOC investigation console (frontend).
   Vanilla JS + fetch() against the Flask backend. No frameworks.

   SECURITY: URL strings are UNTRUSTED input. Every dynamic value is
   HTML-escaped (esc()) or set via textContent — never injected as
   raw markup. All numbers displayed come from the backend response.
   ============================================================= */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

/* ---------------- helpers ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function fmtPct(p, digits = 1) { return (Number(p) * 100).toFixed(digits) + "%"; }
function fmtVal(v, digits = 4) {
  const n = Number(v);
  return (v === null || v === undefined || v === "" || Number.isNaN(n)) ? "—" : n.toFixed(digits);
}
function hhmmssUTC(iso) { return iso ? String(iso).slice(11, 19) : "—"; }
function dateUTC(iso) { return iso ? String(iso).slice(0, 10) : "—"; }
function truncMid(s, n = 72) {
  s = String(s ?? "");
  if (s.length <= n) return s;
  return s.slice(0, n - 14) + "…" + s.slice(-12);
}
const SEV_RANK = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
const IOC_WARN_STATUSES = new Set(["Suspicious name", "Non-standard", "Detected",
  "Raw IP host", "Present - review", "URL shortener"]);

const state = {
  result: null,
  modelInfo: null,
  history: { rows: [], total: 0, sortKey: "time", sortDir: "desc" },
};

/* ---------------- API ---------------- */
async function api(path, opts) {
  let resp;
  try { resp = await fetch(path, opts); }
  catch (e) { throw new Error("network error — analyzer unreachable"); }
  let body = null;
  try { body = await resp.json(); } catch (e) { /* empty body */ }
  if (!resp.ok) {
    const msg = (body && body.error && (body.error.message || body.error.code)) || ("HTTP " + resp.status);
    const err = new Error(msg);
    err.status = resp.status;
    throw err;
  }
  return body;
}
function postJSON(path, data) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/* ---------------- clipboard ---------------- */
async function copyText(text, btn) {
  let ok = false;
  try { await navigator.clipboard.writeText(text); ok = true; }
  catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
    ta.remove();
  }
  if (btn) {
    const old = btn.textContent;
    btn.textContent = ok ? "COPIED" : "FAILED";
    btn.classList.add(ok ? "ok" : "err");
    setTimeout(() => { btn.textContent = old; btn.classList.remove("ok", "err"); }, 900);
  }
}

/* ---------------- messages ---------------- */
function showMsg(sel, text, kind) { const el = $(sel); el.textContent = text; el.className = "msg " + (kind || "info"); }
function hideMsg(sel) { const el = $(sel); el.className = "msg hidden"; el.textContent = ""; }

/* ---------------- views ---------------- */
function switchView(name) {
  $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (name === "history") loadHistory();
  if (name === "model" || name === "dataset") loadModelInfo();
}

/* ---------------- health / clock ---------------- */
function shortBackend(b) {
  b = String(b || "—");
  if (b.startsWith("shap.")) return "shap.TreeExplainer";
  if (b.startsWith("LightGBM")) return "LightGBM native";
  return b;
}
async function pollHealth() {
  try {
    const h = await api("/health");
    $("#sys-dot").classList.remove("off");
    $("#sys-label").textContent = "SYSTEM ONLINE";
    $("#side-model").textContent = h.model || "—";
    $("#side-threshold").textContent = Number(h.threshold).toFixed(4);
    $("#side-backend").textContent = shortBackend(h.shap_backend);
    const cch = h.cache || {};
    $("#side-cache").textContent = `${cch.entries ?? "—"}/${cch.max_entries ?? "—"}`;
    const vt = (h.virustotal || {}).status || "—";
    const vtChip = $("#chip-vt");
    vtChip.textContent = "VT: " + String(vt).toUpperCase();
    vtChip.className = "chip " + (vt === "enabled" ? "ok" : "dim");
  } catch (e) {
    $("#sys-dot").classList.add("off");
    $("#sys-label").textContent = "SYSTEM OFFLINE";
  }
}
function tickClock() {
  const d = new Date();
  const two = (n) => String(n).padStart(2, "0");
  $("#clock").textContent = `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
}

/* ---------------- investigate ---------------- */
async function runScan(url, opts) {
  const btn = $("#analyze-btn");
  hideMsg("#scan-msg");
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "ANALYZING…";
  try {
    const payload = { url };
    if (opts && opts.enrich) payload.include_enrichment = true;
    if (opts && opts.vt) payload.include_external = true;
    const res = await postJSON("/predict", payload);
    state.result = res;
    renderResult(res);
    $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showMsg("#scan-msg", "SCAN FAILED — " + err.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

function verdictBadge(v) {
  return `<span class="badge ${v === "phishing" ? "v-phishing" : "v-safe"}">${esc(String(v || "—").toUpperCase())}</span>`;
}
function sevBadge(s) { return `<span class="badge sev-${String(s || "").toLowerCase()}">${esc(s || "—")}</span>`; }

function renderResult(res) {
  $("#result").classList.remove("hidden");
  renderVerdict(res);
  renderDecon(res.ioc || {});
  renderShap(res.shap || {});
  renderIoc(res.ioc || {});
  renderEnrich(res.enrichment);
  renderVt(res.virustotal);
}

function renderVerdict(res) {
  const isPhish = res.verdict === "phishing";
  $("#verdict-body").innerHTML = `
    <div class="verdict-line">
      ${verdictBadge(res.verdict)}
      <span class="score-num mono ${isPhish ? "red" : "green"}">${fmtPct(res.probability, res.probability < 0.01 ? 2 : 1)}</span>
      <span class="score-label dim">CALIBRATED PHISHING PROBABILITY</span>
      ${sevBadge(res.severity)}
    </div>
    <div class="score-bar ${isPhish ? "fill-red" : "fill-green"}">
      <div class="score-fill" style="width:${Math.min(100, Number(res.probability) * 100).toFixed(2)}%"></div>
      <div class="thr-mark" style="left:${(Number(res.threshold) * 100).toFixed(2)}%" title="classification threshold"></div>
    </div>
    <div class="score-bar-legend mono dim"><span>0%</span><span>threshold ${Number(res.threshold).toFixed(4)}</span><span>100%</span></div>
    <div class="meta-line mono dim">
      scan #${esc(res.scan_id)} · ${esc(hhmmssUTC(res.scanned_at))} UTC · ${esc(dateUTC(res.scanned_at))}
      ${res.cache_hit ? " · CACHE HIT" : ""} · p=${Number(res.probability).toFixed(6)}
      · model ${esc((res.model || {}).name || "—")} · ${esc((res.shap || {}).backend_label || "")}
    </div>`;
}

function renderDecon(ioc) {
  const c = ioc.components || {};
  const flags = ioc.flags || [];
  const flagSet = new Set(flags.map((f) => f.flag));
  const rows = [
    ["SCHEME", c.scheme, false],
    ["USERINFO", c.userinfo, flagSet.has("userinfo_present")],
    ["HOST", c.host, flagSet.has("ip_host")],
    ["SUBDOMAIN", c.subdomain, flagSet.has("many_subdomains")],
    ["REGISTRABLE DOMAIN", c.registrable_domain, false],
    ["TLD", c.tld, false],
    ["IP", c.is_ip ? c.ip : null, flagSet.has("ip_host")],
    ["PORT", c.port, flagSet.has("unusual_port")],
    ["PATH", c.path || null,
      flagSet.has("long_path") || flagSet.has("many_path_segments") ||
      flagSet.has("script_extension") || flagSet.has("executable_extension")],
    ["QUERY", c.query || null,
      flagSet.has("many_query_params") || flagSet.has("percent_encoding") ||
      flagSet.has("double_encoding") || flagSet.has("suspicious_parameter_names")],
    ["FRAGMENT", c.fragment || null, false],
  ];
  const urlEl = $("#decon-url");
  urlEl.textContent = truncMid(c.url || "", 84);
  urlEl.title = c.url || "";
  const body = rows.map(([k, v, hl]) => `
    <tr class="${hl ? "hl" : ""}">
      <td class="decon-k dim">${esc(k)}</td>
      <td class="mono decon-v">${esc(v ?? "—")}</td>
    </tr>`).join("");
  const flagsHtml = flags.length
    ? `<div class="flags-head dim">OBSERVED INDICATORS — NOT PROOF OF PHISHING</div>
       <ul class="flags-list">${flags.map((f) =>
         `<li><span class="flag-name mono">${esc(f.flag)}</span><span class="dim flag-detail">${esc(f.detail)}</span></li>`).join("")}</ul>`
    : `<div class="dim" style="padding-top:8px">No notable indicators observed.</div>`;
  $("#decon-body").innerHTML = `<table class="decon-table"><tbody>${body}</tbody></table>${flagsHtml}`;
}

function renderShap(sh) {
  $("#shap-meta").textContent =
    `base ${fmtVal(sh.base_value, 3)} · margin ${fmtVal(sh.raw_margin, 3)} · lexical ${fmtVal(sh.lexical_total, 2)} · n-grams ${fmtVal(sh.ngram_total, 2)}`;
  const all = [...(sh.increasing || []), ...(sh.decreasing || [])];
  const maxC = Math.max(1e-9, ...all.map((x) => Math.abs(Number(x.contribution))));
  const rows = (list) => {
    if (!list || !list.length) return `<div class="dim shap-empty">(none)</div>`;
    return list.map((r) => `
      <div class="shap-row">
        <span class="shap-name mono" title="${esc(r.feature)}">${esc(r.feature)}</span>
        <span class="shap-val mono dim" title="${esc(r.display_value ?? "")}">${esc(r.display_value ?? "")}</span>
        <span class="shap-bar"><span class="shap-fill ${Number(r.contribution) > 0 ? "pos" : "neg"}"
          style="width:${(Math.abs(Number(r.contribution)) / maxC * 50).toFixed(1)}%"></span></span>
        <span class="shap-contrib mono ${Number(r.contribution) > 0 ? "red" : "green"}">${Number(r.contribution) >= 0 ? "+" : ""}${Number(r.contribution).toFixed(4)}</span>
      </div>`).join("");
  };
  $("#shap-body").innerHTML = `
    <div class="shap-sub red-text">FEATURES INCREASING PHISHING RISK</div>${rows(sh.increasing)}
    <div class="shap-sub green-text">FEATURES REDUCING PHISHING RISK</div>${rows(sh.decreasing)}
    <div class="footnote dim">${esc(sh.note || "")}</div>`;
}

function renderIoc(ioc) {
  const rows = ioc.iocs || [];
  $("#ioc-count").textContent = `${rows.length} ROWS`;
  const body = $("#ioc-body");
  body.innerHTML = "";
  const table = document.createElement("table");
  table.className = "ioc-table";
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>TYPE</th><th>VALUE</th><th>STATUS</th><th></th></tr>";
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    const tdType = document.createElement("td");
    tdType.className = "mono dim";
    tdType.textContent = r.type || "";
    const tdVal = document.createElement("td");
    const warn = IOC_WARN_STATUSES.has(r.status);
    tdVal.className = "mono ioc-val" + (warn ? " warn" : "");
    tdVal.textContent = r.value ?? "—";
    tdVal.title = String(r.value ?? "");
    const tdStat = document.createElement("td");
    tdStat.className = "ioc-stat" + (warn ? " warn" : "");
    tdStat.textContent = r.status || "—";
    const tdCopy = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "COPY";
    btn.dataset.copy = String(r.value ?? "");
    tdCopy.appendChild(btn);
    tr.append(tdType, tdVal, tdStat, tdCopy);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  body.appendChild(table);
}

function kvRow(k, v) {
  const empty = v === null || v === undefined || v === "";
  return `<div class="kv-row"><span class="k dim">${esc(k)}</span><span class="v mono">${empty ? "Unavailable" : esc(v)}</span></div>`;
}

function renderEnrich(en) {
  const panel = $("#enrich-panel"), body = $("#enrich-body");
  if (!en) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const w = en.whois || {};
  const d = en.dns || {};
  const mx = (d.has_mx === null || d.has_mx === undefined) ? "Unavailable" : (d.has_mx ? "yes" : "no");
  body.innerHTML = `
    <div class="enrich-sub dim">WHOIS — ${esc(w.domain || "—")}</div>
    <div class="kv-grid">
      ${kvRow("REGISTRAR", w.registrar)}
      ${kvRow("CREATED", w.creation_date)}
      ${kvRow("EXPIRES", w.expiration_date)}
      ${kvRow("DOMAIN AGE (DAYS)", w.domain_age_days)}
      ${kvRow("DAYS TO EXPIRY", w.days_to_expiry)}
    </div>
    ${w.error ? `<div class="footnote dim">WHOIS: ${esc(w.error)}</div>` : ""}
    <div class="enrich-sub dim">DNS — ${esc(d.host || "—")}</div>
    <div class="kv-grid">
      ${kvRow("A RECORDS", d.a_record_count)}
      ${kvRow("A TTL (S)", d.a_record_ttl)}
      ${kvRow("MX PRESENT", mx)}
      ${kvRow("NS COUNT", d.ns_count)}
      ${kvRow("NS DIVERSITY", d.ns_diversity)}
    </div>
    ${d.error ? `<div class="footnote dim">DNS: ${esc(d.error)}</div>` : ""}
    <div class="footnote dim">${esc(w.note || "")} ${esc(d.note || "")}</div>`;
}

function statBox(label, value, tone) {
  return `<div class="stat-box"><span class="stat-val mono ${tone || ""}">${esc(value)}</span><span class="stat-lab dim">${esc(label)}</span></div>`;
}

function renderVt(vt) {
  const panel = $("#vt-panel"), chip = $("#vt-status-chip"), body = $("#vt-body");
  if (!vt) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  chip.textContent = String(vt.status || "UNKNOWN").toUpperCase();
  chip.className = "chip vt-" + String(vt.status || "unknown");
  if (vt.status === "ok") {
    const s = vt.last_analysis_stats || {};
    const det = Number(vt.engines_detected || 0);
    body.innerHTML = `
      <div class="kv-row"><span class="k dim">LOOKUP URL</span><span class="v mono">${esc(vt.lookup_url || "—")}</span></div>
      <div class="stat-grid">
        ${statBox("ENGINES DETECTED", (vt.engines_detected ?? 0) + " / " + (vt.engines_total ?? 0), det > 0 ? "red" : "green")}
        ${statBox("MALICIOUS", s.malicious ?? "—", "red")}
        ${statBox("SUSPICIOUS", s.suspicious ?? "—", "amber")}
        ${statBox("HARMLESS", s.harmless ?? "—", "green")}
        ${statBox("UNDETECTED", s.undetected ?? "—", "")}
      </div>
      <div class="kv-grid">
        ${kvRow("REPUTATION", vt.reputation)}
        ${kvRow("LAST ANALYSIS", vt.last_analysis_date)}
      </div>
      <div class="footnote dim">${esc(vt.note || "")}</div>`;
  } else {
    body.innerHTML = `
      <div class="kv-row"><span class="k dim">STATUS</span><span class="v mono">${esc(vt.status)}</span></div>
      ${vt.reason ? `<div class="kv-row"><span class="k dim">DETAIL</span><span class="v mono">${esc(vt.reason)}</span></div>` : ""}
      ${vt.lookup_url ? `<div class="kv-row"><span class="k dim">LOOKUP URL</span><span class="v mono">${esc(vt.lookup_url)}</span></div>` : ""}
      <div class="footnote dim">${esc(vt.note || "")}</div>`;
  }
}

/* ---------------- history ---------------- */
async function loadHistory() {
  const params = new URLSearchParams();
  params.set("limit", "100");
  const q = $("#hist-search").value.trim();
  if (q) params.set("q", q);
  const v = $("#hist-verdict").value; if (v) params.set("verdict", v);
  const s = $("#hist-severity").value; if (s) params.set("severity", s);
  params.set("order", $("#hist-order").value);
  try {
    const data = await api("/history?" + params.toString());
    state.history.rows = data.scans || [];
    state.history.total = data.total_matching ?? 0;
    renderHistory();
  } catch (err) {
    $("#hist-footer").textContent = "HISTORY LOAD FAILED — " + err.message;
    $("#hist-tbody").innerHTML = "";
  }
}

function sortRows(rows, key, dir) {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    let va, vb;
    if (key === "time") { va = a.scanned_at || ""; vb = b.scanned_at || ""; }
    else if (key === "score") { va = Number(a.probability); vb = Number(b.probability); }
    else if (key === "verdict") { va = a.verdict || ""; vb = b.verdict || ""; }
    else { va = SEV_RANK[a.severity] ?? -1; vb = SEV_RANK[b.severity] ?? -1; }
    if (va < vb) return -1 * sign;
    if (va > vb) return 1 * sign;
    return 0;
  });
}

function renderHistory() {
  const rows = sortRows(state.history.rows, state.history.sortKey, state.history.sortDir);
  $("#hist-tbody").innerHTML = rows.length ? rows.map((r) => `
    <tr>
      <td class="mono dim" title="${esc(dateUTC(r.scanned_at))}">${esc(hhmmssUTC(r.scanned_at))}</td>
      <td class="mono url-cell" title="${esc(r.url)}">${esc(truncMid(r.url, 58))}</td>
      <td>${verdictBadge(r.verdict)}</td>
      <td class="mono">${fmtPct(r.probability)}</td>
      <td>${sevBadge(r.severity)}</td>
      <td><button class="btn-mini rescan-btn" type="button" data-url="${esc(r.url)}">RESCAN</button></td>
    </tr>`).join("") : `<tr><td colspan="6" class="dim" style="text-align:center;padding:14px">No scans recorded.</td></tr>`;
  $$("#hist-table th.sortable").forEach((th) => {
    const active = th.dataset.sort === state.history.sortKey;
    th.classList.toggle("sorted", active);
    th.textContent = th.textContent.replace(/ [▲▼]$/, "");
    if (active) th.textContent += state.history.sortDir === "asc" ? " ▲" : " ▼";
  });
  $("#hist-footer").textContent =
    `SHOWING ${rows.length} OF ${state.history.total} MATCHING · SERVER LIMIT 100 · LOCAL SORT: ${state.history.sortKey.toUpperCase()} ${state.history.sortDir.toUpperCase()}`;
}

/* ---------------- batch ---------------- */
function renderBatch(data) {
  const s = data.summary || {};
  $("#batch-result").classList.remove("hidden");
  $("#batch-summary").textContent =
    `TOTAL ${s.total} · VALID ${s.valid} · PHISHING ${s.phishing} · SAFE ${s.safe} · ERRORS ${s.errors}`;
  const rows = (data.results || []).map((r) => {
    if (r.status !== "ok") {
      return `<tr>
        <td class="mono url-cell" title="${esc(r.url)}">${esc(truncMid(r.url, 54))}</td>
        <td colspan="3" class="err-text">ERROR — ${esc((r.error || {}).message || "invalid")}</td>
        <td></td></tr>`;
    }
    return `<tr>
      <td class="mono url-cell" title="${esc(r.url)}">${esc(truncMid(r.url, 54))}</td>
      <td>${verdictBadge(r.verdict)}</td>
      <td class="mono">${fmtPct(r.probability)}</td>
      <td>${sevBadge(r.severity)}</td>
      <td><button class="btn-mini rescan-btn" type="button" data-url="${esc(r.url)}">RESCAN</button></td>
    </tr>`;
  }).join("");
  $("#batch-body").innerHTML =
    `<table class="data-table"><thead><tr><th>URL</th><th>VERDICT</th><th>SCORE</th><th>SEVERITY</th><th></th></tr></thead>
     <tbody>${rows || `<tr><td colspan="5" class="dim" style="text-align:center;padding:14px">No results.</td></tr>`}</tbody></table>`;
}

/* ---------------- model & dataset views ---------------- */
async function loadModelInfo() {
  if (state.modelInfo) { renderModelInfo(); return; }
  try {
    state.modelInfo = await api("/api/model-info");
    renderModelInfo();
  } catch (e) { /* panels remain empty */ }
}
function renderModelInfo() {
  const mi = state.modelInfo;
  if (!mi) return;
  const m = mi.model || {};
  const t = mi.threshold || {};
  const tr = (mi.test_results || {}).calibrated_at_threshold || {};
  const vop = t.validation_operating_point || {};
  const bands = t.severity_bands || {};

  $("#model-body").innerHTML = `
    <div class="kv-grid">
      ${kvRow("MODEL", m.name)}
      ${kvRow("VERSION", m.version)}
      ${kvRow("EXPLAINER", m.shap_backend)}
    </div>
    <div class="footnote dim">${esc((m.calibration && ((m.calibration.approach || "") + " — " + (m.calibration.note || ""))) || "")}</div>
    <div class="kv-grid">
      ${Object.entries(m.library_versions || {}).map(([k, v]) => kvRow(k.toUpperCase(), v)).join("")}
    </div>`;

  $("#threshold-body").innerHTML = `
    <div class="big-num mono">${fmtVal(t.deployed, 4)}</div>
    <div class="dim" style="padding:2px 0 8px">DEPLOYED CLASSIFICATION THRESHOLD — VERDICT = PHISHING IF p ≥ THRESHOLD</div>
    <div class="kv-grid">
      ${kvRow("TARGET RECALL", t.target_recall)}
      ${kvRow("MIN PRECISION", t.min_precision)}
      ${kvRow("VAL ACCURACY", vop.accuracy)}
      ${kvRow("VAL PRECISION", vop.precision)}
      ${kvRow("VAL RECALL", vop.recall)}
      ${kvRow("VAL F1", vop.f1)}
    </div>
    <div class="kv-grid">
      ${kvRow("SEVERITY BANDS", `LOW<${bands.low_max} · MED<${bands.medium_max} · HIGH<${bands.high_max} · CRIT≥`)}
    </div>
    <div class="footnote dim">${esc(t.policy || "")}</div>`;

  $("#test-body").innerHTML = `
    <div class="kv-grid">
      ${kvRow("ACCURACY", tr.accuracy)} ${kvRow("PRECISION", tr.precision)}
      ${kvRow("RECALL", tr.recall)} ${kvRow("F1", tr.f1)}
      ${kvRow("ROC-AUC", tr.roc_auc)} ${kvRow("PR-AUC", tr.pr_auc)}
      ${kvRow("BRIER", tr.brier)} ${kvRow("ECE (10-BIN)", tr.ece_10bin)}
      ${kvRow("TEST ROWS", (mi.test_results || {}).n_test_rows)}
      ${kvRow("MISCLASSIFIED", (mi.test_results || {}).n_misclassified)}
      ${kvRow("TN", tr.tn)} ${kvRow("FP", tr.fp)}
      ${kvRow("FN", tr.fn)} ${kvRow("TP", tr.tp)}
    </div>
    <div class="footnote dim">One-time evaluation on the frozen held-out test split (domains disjoint from
    training); calibrated model at the deployed threshold. Never used for model selection,
    hyperparameters or threshold optimization.</div>`;

  $("#rationale-body").innerHTML = `<p class="rationale">${esc(m.selection_rationale || "—")}</p>`;

  const fin = (mi.dataset || {}).final || {};
  const src = Object.entries((mi.dataset || {}).sources || {});
  $("#dataset-body").innerHTML = `
    <div class="kv-grid">
      ${kvRow("TOTAL ROWS", fin.total)}
      ${kvRow("MALICIOUS (LABEL 1)", (fin.by_label || {})["1"])}
      ${kvRow("BENIGN (LABEL 0)", (fin.by_label || {})["0"])}
      ${kvRow("UNIQUE REGISTRABLE DOMAINS", fin.unique_registrable_domains)}
    </div>
    <div class="enrich-sub dim">BY SOURCE</div>
    <div class="kv-grid">${Object.entries(fin.by_source || {}).map(([k, v]) => kvRow(k.toUpperCase(), v)).join("")}</div>
    <div class="enrich-sub dim">BY THREAT TYPE</div>
    <div class="kv-grid">${Object.entries(fin.by_threat_type || {}).map(([k, v]) => kvRow(k.toUpperCase(), v)).join("")}</div>`;
  $("#sources-body").innerHTML = src.length
    ? `<table class="data-table"><thead><tr><th>SOURCE</th><th>STATUS</th><th>RAW RECORDS</th></tr></thead>
       <tbody>${src.map(([name, info]) => `<tr><td class="mono">${esc(name)}</td><td>${esc((info || {}).status || "—")}</td><td class="mono">${esc(String((info || {}).raw_records ?? "—"))}</td></tr>`).join("")}</tbody></table>`
    : `<div class="dim" style="padding:10px">Source report unavailable.</div>`;
  $("#notes-body").innerHTML = ((mi.dataset || {}).notes || []).map((n) => `<div class="note-row">${esc(n)}</div>`).join("") || `<div class="dim">No notes.</div>`;
}

/* ---------------- export ---------------- */
function download(filename, mime, text) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
}
function csvField(v) { return '"' + String(v ?? "").replace(/"/g, '""') + '"'; }
function buildCSV(r) {
  const sh = r.shap || {}, ioc = r.ioc || {}, c = ioc.components || {};
  const inc = (sh.increasing || []).map((x) => `${x.feature}:${Number(x.contribution).toFixed(4)}`).join("; ");
  const dec = (sh.decreasing || []).map((x) => `${x.feature}:${Number(x.contribution).toFixed(4)}`).join("; ");
  const flags = (ioc.flags || []).map((f) => f.flag).join("|");
  const header = ["scan_id", "scanned_at", "url", "verdict", "probability", "severity",
    "threshold", "model", "shap_backend", "base_value", "raw_margin", "lexical_total",
    "ngram_total", "top_increasing_features", "top_decreasing_features", "host",
    "registrable_domain", "ip", "port", "path", "query", "flags"].join(",");
  const row = [r.scan_id, r.scanned_at, r.url, r.verdict, r.probability, r.severity,
    r.threshold, (r.model || {}).name, sh.backend, sh.base_value, sh.raw_margin,
    sh.lexical_total, sh.ngram_total, inc, dec, c.host, c.registrable_domain, c.ip,
    c.port, c.path, c.query, flags].map(csvField).join(",");
  return header + "\n" + row + "\n";
}

/* ---------------- init ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  $$(".nav-btn").forEach((b) => b.addEventListener("click", () => switchView(b.dataset.view)));

  $("#scan-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const url = $("#url-input").value.trim();
    if (!url) { showMsg("#scan-msg", "Enter a URL to analyze.", "err"); return; }
    runScan(url, { enrich: $("#opt-enrich").checked, vt: $("#opt-vt").checked });
  });

  $("#btn-export-json").addEventListener("click", () => {
    if (!state.result) return;
    download(`scan_${state.result.scan_id || "result"}.json`, "application/json",
      JSON.stringify(state.result, null, 2));
  });
  $("#btn-export-csv").addEventListener("click", () => {
    if (!state.result) return;
    download(`scan_${state.result.scan_id || "result"}.csv`, "text/csv", buildCSV(state.result));
  });

  $("#btn-batch").addEventListener("click", async () => {
    hideMsg("#batch-msg");
    const urls = $("#batch-input").value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!urls.length) { showMsg("#batch-msg", "Enter at least one URL (one per line).", "err"); return; }
    if (urls.length > 50) { showMsg("#batch-msg", "Batch limit is 50 URLs.", "err"); return; }
    const btn = $("#btn-batch");
    btn.disabled = true;
    const old = btn.textContent;
    btn.textContent = "SCANNING…";
    try { renderBatch(await postJSON("/predict_batch", { urls })); }
    catch (err) { showMsg("#batch-msg", "BATCH FAILED — " + err.message, "err"); }
    finally { btn.disabled = false; btn.textContent = old; }
  });
  $("#btn-batch-clear").addEventListener("click", () => {
    $("#batch-input").value = "";
    hideMsg("#batch-msg");
    $("#batch-result").classList.add("hidden");
  });

  let histTimer = null;
  $("#hist-search").addEventListener("input", () => {
    clearTimeout(histTimer);
    histTimer = setTimeout(loadHistory, 350);
  });
  ["#hist-verdict", "#hist-severity", "#hist-order"].forEach((sel) =>
    $(sel).addEventListener("change", loadHistory));
  $("#btn-hist-refresh").addEventListener("click", loadHistory);
  $$("#hist-table th.sortable").forEach((th) => th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (state.history.sortKey === key) {
      state.history.sortDir = state.history.sortDir === "asc" ? "desc" : "asc";
    } else {
      state.history.sortKey = key;
      state.history.sortDir = "desc";
    }
    renderHistory();
  }));

  document.addEventListener("click", async (e) => {
    const copyBtn = e.target.closest(".copy-btn");
    if (copyBtn) { await copyText(copyBtn.dataset.copy || "", copyBtn); return; }
    const rescan = e.target.closest(".rescan-btn");
    if (rescan) {
      const url = rescan.dataset.url || "";
      $("#url-input").value = url;
      switchView("investigate");
      runScan(url, {});
    }
  });

  pollHealth();
  setInterval(pollHealth, 30000);
  tickClock();
  setInterval(tickClock, 1000);
  loadModelInfo();
});
