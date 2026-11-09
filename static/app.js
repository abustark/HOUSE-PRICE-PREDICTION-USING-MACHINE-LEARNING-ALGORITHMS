/* Bengaluru House Price Prediction — Minimalist Editorial Client
   Inspired by CoverCraft (coverltr.vercel.app) & Abufolio
   Features:
   - Automatic Lakh / Crore Financial Scaling (>= 100 Lakh -> Crore format, e.g. ₹1.86 Cr)
   - Zero-Reload In-App API Console & Live Request Dispatcher
   - Synchronized Total Sqft Number Input & Smooth Range Slider
   - Domain Extrapolation Warning Diagnostics & Unseen Categories Handling
   - Calibrated 80% Error Intervals & Side-by-Side Model Cross-Check
   - Smooth Mobile Scroll and Responsive View Switching
*/

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Theme Management (Light / Dark Mode)
// ---------------------------------------------------------------------------
function initTheme() {
  const stored = localStorage.getItem("hpp_theme");
  const systemPrefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = stored || (systemPrefersDark ? "dark" : "light");
  setTheme(initialTheme);

  const toggleBtn = $("theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      setTheme(next);
    });
  }
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("hpp_theme", theme);
  } catch (e) {
    // Ignore storage quota or disabled errors
  }
  const toggleBtn = $("theme-toggle");
  if (toggleBtn) {
    const nextTheme = theme === "dark" ? "light" : "dark";
    toggleBtn.setAttribute("title", `Switch to ${nextTheme} theme`);
    toggleBtn.setAttribute("aria-label", `Switch to ${nextTheme} theme`);
  }
}

// Initialize theme immediately to prevent flashing
initTheme();

let appData = {
  models: [],
  locations: [],
  meta: {},
  bestModel: "random_forest",
  selectedModel: "random_forest",
};

// ---------------------------------------------------------------------------
// Price Formatting Helpers
// ---------------------------------------------------------------------------
function fmtLakh(v) {
  const n = Number(v);
  if (isNaN(n)) return "0.00";
  return n.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function fmtInr(v) {
  const n = Number(v);
  if (isNaN(n)) return "0";
  return Math.round(n).toLocaleString("en-IN");
}

/**
 * Format price according to Indian financial nomenclature:
 * If value in Lakhs >= 100 (i.e. >= 1 Crore), display as ₹X.XX Cr.
 * If value in Lakhs < 100, display as ₹XX.XX Lakh.
 */
function formatPriceBreakdown(lakhVal) {
  const v = Number(lakhVal);
  if (isNaN(v) || v <= 0) {
    return { numStr: "–", unit: "Lakh", fullStr: "–", altStr: "" };
  }

  if (v >= 100) {
    const cr = v / 100.0;
    const numStr = cr.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    const lakhAlt = v.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    return {
      numStr,
      unit: "Cr",
      fullStr: `₹${numStr} Cr`,
      altStr: `₹${lakhAlt} Lakh`,
    };
  } else {
    const numStr = v.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    return {
      numStr,
      unit: "Lakh",
      fullStr: `₹${numStr} Lakh`,
      altStr: "",
    };
  }
}

function formatPrice(lakhVal) {
  return formatPriceBreakdown(lakhVal).fullStr;
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

function fillSelect(el, items, selectedVal) {
  if (!el) return;
  el.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item;
    opt.textContent = item;
    if (item === selectedVal) opt.selected = true;
    el.appendChild(opt);
  }
}

// ---------------------------------------------------------------------------
// Tab Navigation (Segmented View Switcher)
// ---------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".nav-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      const targetView = tab.getAttribute("data-view");
      document.querySelectorAll(".view-panel").forEach((panel) => {
        const isMatch = panel.id === `view-${targetView}`;
        panel.hidden = !isMatch;
        if (isMatch) {
          panel.classList.add("active");
        } else {
          panel.classList.remove("active");
        }
      });
    });
  });

  $("btn-back-to-top")?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

// ---------------------------------------------------------------------------
// Model Selection Chips
// ---------------------------------------------------------------------------
function renderModelChips() {
  const container = $("model-chips");
  if (!container) return;
  container.innerHTML = "";

  for (const m of appData.models) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "model-chip" + (m.key === appData.selectedModel ? " active" : "");
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(m.key === appData.selectedModel));

    const name = document.createElement("span");
    name.className = "chip-name";
    name.textContent = m.name;
    btn.appendChild(name);

    const r2 = document.createElement("small");
    r2.className = "chip-stat";
    r2.textContent = `R² ${m.r2_log.toFixed(3)}`;
    btn.appendChild(r2);

    btn.addEventListener("click", () => {
      appData.selectedModel = m.key;
      renderModelChips();
    });

    container.appendChild(btn);
  }
}

// ---------------------------------------------------------------------------
// Synchronized Range Slider & Number Input
// ---------------------------------------------------------------------------
function initSqftSync() {
  const numInput = $("total_sqft");
  const rangeInput = $("total_sqft_range");
  const displayLabel = $("sqft-val");

  function update(val) {
    const clamped = Math.max(100, Math.min(10000, Number(val) || 1200));
    numInput.value = clamped;
    rangeInput.value = clamped;
    displayLabel.textContent = `${Number(clamped).toLocaleString("en-IN")} sqft`;
  }

  numInput.addEventListener("input", (e) => update(e.target.value));
  rangeInput.addEventListener("input", (e) => update(e.target.value));
  update(numInput.value);
}

// ---------------------------------------------------------------------------
// Historical Presets
// ---------------------------------------------------------------------------
function setPreset(area, loc, bhk, sqft, bath, balc) {
  $("area_type").value = area;
  $("location").value = loc;
  $("bhk").value = String(bhk);
  $("total_sqft").value = String(sqft);
  $("total_sqft_range").value = String(sqft);
  $("sqft-val").textContent = `${Number(sqft).toLocaleString("en-IN")} sqft`;
  $("bath").value = String(bath);
  $("balcony").value = String(balc);
  document.querySelector('input[name="availability"][value="ready"]').checked = true;
  $("year").disabled = true;
}

function initPresets() {
  $("preset-whitefield")?.addEventListener("click", () =>
    setPreset("Super built-up Area", "Whitefield", 2, 1200, 2, 1)
  );
  $("preset-ecity")?.addEventListener("click", () =>
    setPreset("Super built-up Area", "Electronic City", 3, 1500, 3, 2)
  );
  $("preset-hsr")?.addEventListener("click", () =>
    setPreset("Built-up Area", "HSR Layout", 1, 650, 1, 1)
  );
  $("preset-indira")?.addEventListener("click", () =>
    setPreset("Plot Area", "Indiranagar", 4, 3200, 4, 2)
  );
}

// ---------------------------------------------------------------------------
// Prediction Handler & Valuation Rendering
// ---------------------------------------------------------------------------
function readFormData() {
  const under = document.querySelector('input[name="availability"]:checked').value === "under";
  return {
    area_type: $("area_type").value,
    location: $("location").value,
    society: $("society").value.trim(),
    bhk: Number($("bhk").value),
    total_sqft: Number($("total_sqft").value),
    bath: Number($("bath").value),
    balcony: Number($("balcony").value),
    completion_year: under ? Number($("year").value) : 0,
    model: appData.selectedModel,
  };
}

function showFormError(msg) {
  const el = $("form-error");
  el.textContent = msg;
  el.hidden = !msg;
}

function renderPredictionResult(data) {
  $("result-empty").hidden = true;
  $("result-body").hidden = false;

  // Format price: >= 100 Lakh becomes Crore (Cr)
  const breakdown = formatPriceBreakdown(data.predicted_price_lakh);
  $("price-num").textContent = breakdown.numStr;
  $("price-unit").textContent = breakdown.unit;

  // Subtext: show approx INR and Lakh equivalent if formatted in Crore
  let subtext = `≈ ₹${fmtInr(data.predicted_price_inr)}`;
  if (breakdown.unit === "Cr") {
    subtext += ` · (${breakdown.altStr})`;
  }
  $("price-subtext").textContent = subtext;
  $("active-model-badge").textContent = data.model_name || data.model;

  // Calibrated error range in Lakh / Crore
  if (data.historical_error_range) {
    const r = data.historical_error_range;
    $("error-range-box").hidden = false;
    const lowerFormatted = formatPrice(r.lower_lakh);
    const upperFormatted = formatPrice(r.upper_lakh);
    $("error-range-val").textContent = `${lowerFormatted} – ${upperFormatted}`;
  } else {
    $("error-range-box").hidden = true;
  }

  // Warnings list
  const warnBox = $("warnings-box");
  const warnList = $("warnings-list");
  warnList.innerHTML = "";
  if (data.warnings && data.warnings.length > 0) {
    warnBox.hidden = false;
    data.warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      warnList.appendChild(li);
    });
  } else {
    warnBox.hidden = true;
  }

  // Unseen society note
  $("unseen-note").hidden = !data.unseen_society;

  // Side-by-side comparison table
  const tbody = $("compare-rows");
  tbody.innerHTML = "";
  for (const m of appData.models) {
    const entry = data.all_models[m.key];
    if (!entry) continue;
    const tr = document.createElement("tr");
    if (m.key === data.model_key) tr.className = "highlight";
    
    // Format each model's estimate with Crore/Lakh scaling
    const modelPrice = formatPrice(entry.price_lakh);
    tr.innerHTML = `
      <td>${m.name}</td>
      <td class="text-right">${modelPrice}</td>
      <td class="text-right">${m.r2_log.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  }

  // Mobile convenience: scroll smoothly to result if on small viewport
  if (window.innerWidth <= 920) {
    $("result-container")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function handlePredict() {
  showFormError("");
  const btn = $("predict-btn");
  btn.disabled = true;
  btn.innerHTML = `<span>Calculating Valuation…</span>`;

  try {
    const payload = readFormData();
    const res = await fetchJSON("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      if (res.data?.detail) {
        const d = res.data.detail;
        if (typeof d === "string") msg = d;
        else if (d.errors) msg = d.errors.join("; ");
        else if (Array.isArray(d)) msg = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
      }
      throw new Error(msg);
    }

    renderPredictionResult(res.data);
  } catch (err) {
    showFormError(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>Calculate Estimated Valuation</span><span class="btn-arrow">→</span>`;
  }
}

// ---------------------------------------------------------------------------
// VIEW 2: INSTANT API CONSOLE
// ---------------------------------------------------------------------------
const API_ENDPOINTS = {
  predict: {
    method: "POST",
    path: "/predict",
    desc: "Calculate regression price estimates, calibrated 80% error intervals, and domain extrapolation warnings.",
    hasPayload: true,
    defaultPayload: JSON.stringify(
      {
        area_type: "Super built-up Area",
        location: "Whitefield",
        society: "Goted U",
        bhk: 2,
        total_sqft: 1200,
        bath: 2,
        balcony: 1,
        completion_year: 0,
        model: "random_forest",
      },
      null,
      2
    ),
  },
  models: {
    method: "GET",
    path: "/models",
    desc: "Returns test R², MAE (log & Lakh), MedAE, RMSE, and 5-fold cross-validation scores for all trained models.",
    hasPayload: false,
  },
  locations: {
    method: "GET",
    path: "/locations",
    desc: "Returns the alphabetical list of all locations present in the historical training data.",
    hasPayload: false,
  },
  meta: {
    method: "GET",
    path: "/meta",
    desc: "Returns the schema version, supported area types, and physical training domain constraints.",
    hasPayload: false,
  },
  manifest: {
    method: "GET",
    path: "/manifest",
    desc: "Returns the reproducibility manifest: dataset SHA-256 hash, split seeds, row counts, and feature ordering.",
    hasPayload: false,
  },
  health: {
    method: "GET",
    path: "/health",
    desc: "Liveness and schema health check returning status and loaded models.",
    hasPayload: false,
  },
};

let currentEndpointKey = "predict";

function initApiConsole() {
  const epBtns = document.querySelectorAll(".ep-nav-pill");
  epBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      epBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const epKey = btn.getAttribute("data-ep");
      currentEndpointKey = epKey;
      selectEndpoint(epKey);
    });
  });

  $("btn-send-api")?.addEventListener("click", sendApiRequest);
  
  $("btn-reset-payload")?.addEventListener("click", () => {
    const ep = API_ENDPOINTS[currentEndpointKey];
    if (ep && ep.defaultPayload) {
      $("ep-payload").value = ep.defaultPayload;
    }
  });

  $("btn-copy-response")?.addEventListener("click", () => {
    const text = $("ep-response").textContent;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        const btn = $("btn-copy-response");
        const orig = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = orig), 1800);
      });
    }
  });

  selectEndpoint("predict");
}

function selectEndpoint(key) {
  const ep = API_ENDPOINTS[key];
  if (!ep) return;

  const methodTag = $("ep-method");
  methodTag.textContent = ep.method;
  methodTag.className = `pill-method ${ep.method.toLowerCase()}`;

  $("ep-path").textContent = ep.path;
  $("ep-desc").textContent = ep.desc;

  const reqSection = $("ep-request-section");
  if (ep.hasPayload) {
    reqSection.hidden = false;
    $("ep-payload").value = ep.defaultPayload;
  } else {
    reqSection.hidden = true;
  }

  $("api-status-pill").hidden = true;
  $("ep-response").textContent = `Click "Send Live Request" to invoke ${ep.method} ${ep.path} live...`;
}

async function sendApiRequest() {
  const ep = API_ENDPOINTS[currentEndpointKey];
  const sendBtn = $("btn-send-api");
  const statusPill = $("api-status-pill");
  const responseBox = $("ep-response");

  sendBtn.disabled = true;
  sendBtn.innerHTML = `<span>Invoking…</span>`;
  statusPill.hidden = true;
  responseBox.textContent = "Dispatching request to FastAPI backend…";

  try {
    const options = { method: ep.method };
    if (ep.hasPayload) {
      options.headers = { "Content-Type": "application/json" };
      options.body = $("ep-payload").value;
    }

    const t0 = performance.now();
    const res = await fetch(ep.path, options);
    const ms = Math.round(performance.now() - t0);
    const data = await res.json().catch(() => ({}));

    statusPill.hidden = false;
    statusPill.textContent = `Status: ${res.status} ${res.statusText || (res.ok ? "OK" : "Error")} (${ms}ms)`;
    statusPill.style.color = res.ok ? "var(--status-emerald)" : "var(--status-rose)";
    statusPill.style.background = res.ok ? "var(--status-emerald-bg)" : "var(--status-rose-bg)";
    statusPill.style.border = res.ok ? "1px solid rgba(16, 185, 129, 0.3)" : "1px solid rgba(244, 63, 94, 0.3)";

    responseBox.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    statusPill.hidden = false;
    statusPill.textContent = "Network Error";
    statusPill.style.color = "var(--status-rose)";
    statusPill.style.background = "var(--status-rose-bg)";
    responseBox.textContent = `Error: ${err.message || String(err)}`;
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML = `<span>Send Live Request</span><span>→</span>`;
  }
}

// ---------------------------------------------------------------------------
// VIEW 3: MODEL PERFORMANCE
// ---------------------------------------------------------------------------
function populateMetricsTable() {
  const tbody = $("metrics-table-body");
  if (!tbody || !appData.models.length) return;

  tbody.innerHTML = "";
  for (const m of appData.models) {
    const tr = document.createElement("tr");
    if (m.key === appData.bestModel) tr.className = "highlight";
    
    tr.innerHTML = `
      <td><strong>${m.name}</strong> ${m.key === appData.bestModel ? '<span class="meta-tag active-model-tag">Best</span>' : ""}</td>
      <td class="text-right">${m.r2_log.toFixed(3)}</td>
      <td class="text-right">${m.mae_log.toFixed(3)}</td>
      <td class="text-right">${formatPrice(m.mae_lakh)}</td>
      <td class="text-right">${formatPrice(m.medae_lakh)}</td>
      <td class="text-right">${formatPrice(m.rmse_lakh)}</td>
      <td class="text-right">${m.cv_r2_log_mean.toFixed(3)} (±${m.cv_r2_log_std.toFixed(3)})</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---------------------------------------------------------------------------
// App Initialization
// ---------------------------------------------------------------------------
async function initApp() {
  initTabs();
  initSqftSync();
  initPresets();
  initApiConsole();

  // Load models, locations, metadata in parallel
  const [modelsRes, locsRes, metaRes] = await Promise.all([
    fetchJSON("/models"),
    fetchJSON("/locations"),
    fetchJSON("/meta"),
  ]);

  if (modelsRes.ok) {
    appData.models = modelsRes.data.models || [];
    appData.bestModel = modelsRes.data.best_model || "random_forest";
    appData.selectedModel = appData.bestModel;
    renderModelChips();
    populateMetricsTable();
  }

  if (locsRes.ok) {
    appData.locations = locsRes.data.locations || [];
    fillSelect($("location"), appData.locations, "Whitefield");
  }

  if (metaRes.ok) {
    appData.meta = metaRes.data;
    const areas = metaRes.data.supported_area_types || [
      "Built-up Area",
      "Carpet Area",
      "Plot Area",
      "Super built-up Area",
    ];
    fillSelect($("area_type"), areas, "Super built-up Area");
  }

  // Fill BHK select (Studio / RK to 10 BHK)
  const bhkSelect = $("bhk");
  if (bhkSelect) {
    bhkSelect.innerHTML = "";
    for (let i = 0; i <= 10; i++) {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = i === 0 ? "Studio / RK" : `${i} BHK`;
      if (i === 2) opt.selected = true;
      bhkSelect.appendChild(opt);
    }
  }

  // Possession change handler
  document.querySelectorAll('input[name="availability"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const under = document.querySelector('input[name="availability"]:checked').value === "under";
      $("year").disabled = !under;
    });
  });

  $("predict-btn")?.addEventListener("click", handlePredict);

  // Initialize live update notification listener
  initUpdateChecker();
}

// ---------------------------------------------------------------------------
// Real-Time Update Checker & Refresh Notification Toast
// ---------------------------------------------------------------------------
const APP_CLIENT_VERSION = "2.3.3";
let updateToastDismissed = false;

function initUpdateChecker() {
  const toast = $("update-toast");
  const refreshBtn = $("toast-refresh-btn");
  const dismissBtn = $("toast-dismiss-btn");
  const verBadge = $("toast-new-ver");

  if (!toast) return;

  // Ensure initially hidden
  toast.hidden = true;
  toast.classList.remove("visible");
  toast.style.display = "none";

  // Check if already dismissed this session
  try {
    if (sessionStorage.getItem("hpp_update_dismissed")) {
      updateToastDismissed = true;
    }
  } catch (e) {}

  function hideToast(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    updateToastDismissed = true;
    toast.hidden = true;
    toast.classList.remove("visible");
    toast.style.display = "none";
    try {
      sessionStorage.setItem("hpp_update_dismissed", "true");
    } catch (err) {}
  }

  function triggerRefresh(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    // Bust browser and intermediate cache by reloading with unique timestamp query
    const targetUrl = new URL(window.location.href);
    targetUrl.searchParams.set("reload", Date.now().toString());
    window.location.replace(targetUrl.toString());
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", triggerRefresh);
  }

  if (dismissBtn) {
    dismissBtn.addEventListener("click", hideToast);
  }

  async function checkForUpdate() {
    if (updateToastDismissed) return;
    try {
      // Fetch version.json with cache: "no-store" and cache-busting query
      const res = await fetch(`/static/version.json?t=${Date.now()}`, {
        cache: "no-store",
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.version && data.version !== APP_CLIENT_VERSION) {
        if (verBadge) {
          verBadge.textContent = `v${data.version}`;
        }
        toast.hidden = false;
        toast.style.display = "";
        toast.classList.add("visible");
      }
    } catch (e) {
      // Ignore background network errors
    }
  }

  // Check 5s after initial load
  setTimeout(checkForUpdate, 5000);

  // Check periodically every 60 seconds
  setInterval(checkForUpdate, 60000);

  // Check whenever user switches back to this browser tab
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      checkForUpdate();
    }
  });
}

initApp().catch((err) => {
  console.error("Initialization error:", err);
  showFormError("Failed to connect to backend: " + err.message);
});
