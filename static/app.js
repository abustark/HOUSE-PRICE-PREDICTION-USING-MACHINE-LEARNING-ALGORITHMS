/* Bengaluru House Price Prediction - single-page frontend.
   Talks to the FastAPI backend in api/server.py. No frameworks, no build step. */

const $ = (id) => document.getElementById(id);

let modelsInfo = [];
let bestModel = "random_forest";
let selectedModel = "random_forest";

function fmtLakh(v) {
  return Number(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtInr(v) {
  return "₹" + Math.round(v).toLocaleString("en-IN");
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function fillSelect(el, options, placeholder) {
  el.innerHTML = "";
  if (placeholder) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = placeholder;
    opt.disabled = true;
    el.appendChild(opt);
  }
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    el.appendChild(opt);
  }
}

function renderChips() {
  const wrap = $("model-chips");
  wrap.innerHTML = "";
  for (const m of modelsInfo) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (m.key === selectedModel ? " active" : "");
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(m.key === selectedModel));
    const label = document.createElement("span");
    label.textContent = m.name;
    const score = document.createElement("small");
    score.textContent = `R² ${m.r2_log.toFixed(3)}`;
    btn.append(label, score);
    if (m.key === bestModel) {
      const dot = document.createElement("span");
      dot.className = "best-dot";
      dot.title = "best model";
      btn.prepend(dot);
    }
    btn.addEventListener("click", () => {
      selectedModel = m.key;
      renderChips();
    });
    wrap.appendChild(btn);
  }
}

function readForm() {
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
    model: selectedModel,
  };
}

function showError(msg) {
  const el = $("form-error");
  el.textContent = msg;
  el.hidden = !msg;
}

function renderResult(data) {
  $("result-empty").hidden = true;
  $("result-body").hidden = false;

  $("price-lakh").textContent = fmtLakh(data.predicted_price_lakh);
  $("price-inr").textContent = `≈ ${fmtInr(data.predicted_price_inr)}`;
  const chosen = modelsInfo.find((m) => m.key === data.model_key);
  $("result-model").textContent = `Model: ${data.model} · R² (log) ${chosen ? chosen.r2_log.toFixed(3) : ""}`;

  const tbody = $("compare-rows");
  tbody.innerHTML = "";
  for (const m of modelsInfo) {
    const entry = data.all_models[m.key];
    if (!entry) continue;
    const tr = document.createElement("tr");
    if (m.key === data.model_key) tr.className = "highlight";
    tr.innerHTML =
      `<td>${m.name}</td>` +
      `<td class="num">${fmtLakh(entry.price_lakh)}</td>` +
      `<td class="num">${m.r2_log.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  }
  $("unseen-note").hidden = !data.unseen_society;
}

async function predict() {
  showError("");
  const btn = $("predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const payload = readForm();
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `request failed (${res.status})`);
    renderResult(data);
  } catch (err) {
    showError(String(err.message || err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict price";
  }
}

async function init() {
  const [models, locations, meta] = await Promise.all([
    fetchJSON("/models"),
    fetchJSON("/locations"),
    fetchJSON("/meta"),
  ]);
  modelsInfo = models.models;
  bestModel = models.best_model;
  selectedModel = bestModel;

  fillSelect($("area_type"), meta.area_types);
  fillSelect($("location"), locations.locations);
  fillSelect(
    $("bhk"),
    Array.from({ length: 11 }, (_, i) => i),
  );
  $("bhk").selectedIndex = 2;
  for (const opt of $("bhk").options) {
    opt.textContent = opt.value === "0" ? "Studio / RK" : `${opt.value} BHK`;
  }
  renderChips();

  document.querySelectorAll('input[name="availability"]').forEach((r) =>
    r.addEventListener("change", () => {
      $("year").disabled = r.value === "ready" && !r.checked;
      const under = document.querySelector('input[name="availability"]:checked').value === "under";
      $("year").disabled = !under;
    })
  );
  $("predict-btn").addEventListener("click", predict);
}

init().catch((err) => showError("Failed to load app data: " + err.message));
