/* Bengaluru House Price Prediction — Modern SPA Client
   Features:
   - Instant In-App API Console & Explorer (zero full page reload)
   - Synchronized Range Slider + Number Inputs
   - Domain Extrapolation Warning Diagnostics
   - Historical Calibrated Error Ranges
   - Multi-Model Performance Comparison View
*/

const $ = (id) => document.getElementById(id);

let appData = {
  models: [],
  locations: [],
  meta: {},
  bestModel: "random_forest",
  selectedModel: "random_forest",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fmtLakh(v) {
  return Number(v).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function fmtInr(v) {
  return "₹" + Math.round(Number(v)).toLocaleString("en-IN");
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

function fillSelect(el, items, selectedVal) {
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
// Tab Navigation
// ---------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
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
        panel.hidden = panel.id !== `view-${targetView}`;
        if (panel.id === `view-${targetView}`) {
          panel.classList.add("active");
        } else {
          panel.classList.remove("active");
        }
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Model Chips
// ---------------------------------------------------------------------------
function renderModelChips() {
  const container = $("model-chips");
  container.innerHTML = "";

  for (const m of appData.models) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "model-chip" + (m.key === appData.selectedModel ? " active" : "");
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(m.key === appData.selectedModel));

    const name = document.createElement("span");
    name.textContent = m.name;
    btn.appendChild(name);

    if (m.key === appData.bestModel) {
      const tag = document.createElement("span");
      tag.className = "best-tag";
      tag.textContent = "Best";
      btn.appendChild(tag);
    }

    const r2 = document.createElement("small");
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
// Two-Way Sync for Sqft Slider & Input
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
// Presets
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
// Prediction Handler
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

  $("price-lakh").textContent = fmtLakh(data.predicted_price_lakh);
  $("price-inr").textContent = `≈ ${fmtInr(data.predicted_price_inr)}`;
  $("active-model-badge").textContent = data.model_name || data.model;

  // Calibrated error range
  if (data.historical_error_range) {
    const r = data.historical_error_range;
    $("error-range-box").hidden = false;
    $("error-range-val").textContent = `₹${fmtLakh(r.lower_lakh)} Lakh – ₹${fmtLakh(r.upper_lakh)} Lakh`;
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

  // Comparison table
  const tbody = $("compare-rows");
  tbody.innerHTML = "";
  for (const m of appData.models) {
    const entry = data.all_models[m.key];
    if (!entry) continue;
    const tr = document.createElement("tr");
    if (m.key === data.model_key) tr.className = "highlight";
    tr.innerHTML = `
      <td>${m.name}</td>
      <td class="align-right">₹${fmtLakh(entry.price_lakh)} Lakh</td>
      <td class="align-right">${m.r2_log.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function handlePredict() {
  showFormError("");
  const btn = $("predict-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> Computing Prediction…`;

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
    btn.innerHTML = `<span class="btn-icon">⚡</span> Calculate Estimated Price`;
  }
}

// ---------------------------------------------------------------------------
// VIEW 2: INSTANT API CONSOLE
// ---------------------------------------------------------------------------
const API_ENDPOINTS = {
  predict: {
    method: "POST",
    path: "/predict",
    desc: "Send house attributes to calculate regression estimates, calibrated 80% error intervals, and domain extrapolation warnings.",
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
  const epBtns = document.querySelectorAll(".ep-btn");
  epBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      epBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const epKey = btn.getAttribute("data-ep");
      currentEndpointKey = epKey;
      selectEndpoint(epKey);
    });
  });

  $("btn-send-api").addEventListener("click", sendApiRequest);
  selectEndpoint("predict");
}

function selectEndpoint(key) {
  const ep = API_ENDPOINTS[key];
  if (!ep) return;

  const methodTag = $("ep-method");
  methodTag.textContent = ep.method;
  methodTag.className = `method-tag method-${ep.method.toLowerCase()}`;

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
  $("ep-response").textContent = `Click "Send Request" to invoke ${ep.method} ${ep.path} live...`;
}

async function sendApiRequest() {
  const ep = API_ENDPOINTS[currentEndpointKey];
  const sendBtn = $("btn-send-api");
  const statusPill = $("api-status-pill");
  const responseBox = $("ep-response");

  sendBtn.disabled = true;
  sendBtn.textContent = "Invoking…";
  statusPill.hidden = true;
  responseBox.textContent = "Sending request to FastAPI backend…";

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
    statusPill.style.color = res.ok ? "var(--emerald)" : "var(--rose)";
    statusPill.style.background = res.ok ? "rgba(16, 185, 129, 0.15)" : "rgba(244, 63, 94, 0.15)";
    statusPill.style.borderColor = res.ok ? "rgba(16, 185, 129, 0.3)" : "rgba(244, 63, 94, 0.3)";

    responseBox.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    statusPill.hidden = false;
    statusPill.textContent = "Network Error";
    statusPill.style.color = "var(--rose)";
    responseBox.textContent = `Error: ${err.message || String(err)}`;
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "▶ Send Request";
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
      <td><strong>${m.name}</strong> ${m.key === appData.bestModel ? '<span class="best-tag">Selected</span>' : ""}</td>
      <td>${m.r2_log.toFixed(3)}</td>
      <td>${m.mae_log.toFixed(3)}</td>
      <td>₹${fmtLakh(m.mae_lakh)} Lakh</td>
      <td>₹${fmtLakh(m.medae_lakh)} Lakh</td>
      <td>₹${fmtLakh(m.rmse_lakh)} Lakh</td>
      <td>${m.cv_r2_log_mean.toFixed(3)} (±${m.cv_r2_log_std.toFixed(3)})</td>
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

  // Load models, locations, metadata
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

  // Fill BHK select (0 to 10)
  const bhkSelect = $("bhk");
  bhkSelect.innerHTML = "";
  for (let i = 0; i <= 10; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = i === 0 ? "Studio / RK" : `${i} BHK`;
    if (i === 2) opt.selected = true;
    bhkSelect.appendChild(opt);
  }

  // Possession change
  document.querySelectorAll('input[name="availability"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const under = document.querySelector('input[name="availability"]:checked').value === "under";
      $("year").disabled = !under;
    });
  });

  $("predict-btn").addEventListener("click", handlePredict);
}

initApp().catch((err) => {
  console.error("Initialization error:", err);
  showFormError("Failed to connect to backend: " + err.message);
});
