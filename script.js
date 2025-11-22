// script.js — espera encontrar en el mismo directorio:
//  - model.json  (TFJS model)
//  - penaltis_meta.json  (metadatos: feature_columns, scaler.mean & scale)
// The model was trained on features:
//  [Velocidad_kmh, Angulo_grados, Distancia_Portero_m, Presion_Partido=..., Pie_Dominante=...]
// penaltis_meta.json must include:
//  { "feature_columns": [...], "scaler": {"numeric_cols":[...], "mean":[...],"scale":[...]} }

let model = null;
let meta = null;

const RANGES = {
  "Velocidad_kmh": [70, 115],
  "Angulo_grados": [5, 45],
  "Distancia_Portero_m": [0.4, 1.2]
};

async function loadMetaAndModel() {
  try {
    const resp = await fetch("penaltis_meta.json");
    meta = await resp.json();
  } catch (e) {
    console.warn("No se encontró penaltis_meta.json en el repo. Asegúrate de subirlo.");
    document.getElementById("message").textContent = "Falta penaltis_meta.json (subelo al repo).";
  }

  try {
    model = await tf.loadLayersModel("model.json");
    console.log("Modelo TFJS cargado correctamente.");
    document.getElementById("message").textContent = "Modelo cargado ✓";
  } catch (e) {
    console.error("Error al cargar model.json:", e);
    if (!document.getElementById("message").textContent) document.getElementById("message").textContent = "Falta model.json o .bin en el repo.";
  }
}

function showError(msg){
  const m = document.getElementById("message");
  m.style.color = "#ffd966";
  m.textContent = msg;
  document.getElementById("result").textContent = "";
}

function showResult(text, ok=true){
  const r = document.getElementById("result");
  r.style.color = ok ? "#b8ffcc" : "#ffb4b4";
  r.textContent = text;
  document.getElementById("message").textContent = "";
}

// Construye el vector de entrada (alineado a feature_columns del meta)
function buildInputVector(values){
  // values: {Velocidad_kmh, Angulo_grados, Distancia_Portero_m, Presion_Partido, Pie_Dominante}
  const cols = meta.feature_columns;
  const row = {};
  // numeric
  row["Velocidad_kmh"] = parseFloat(values.Velocidad_kmh);
  row["Angulo_grados"] = parseFloat(values.Angulo_grados);
  row["Distancia_Portero_m"] = parseFloat(values.Distancia_Portero_m);
  // categorical -> one-hot: meta.feature_columns contains e.g. Presion_Partido_Alta etc.
  // Build temp map for categories
  if (values.Presion_Partido) row["Presion_Partido"] = values.Presion_Partido;
  if (values.Pie_Dominante) row["Pie_Dominante"] = values.Pie_Dominante;

  // Create vector following cols order
  const vec = [];
  for (let c of cols){
    if (c === "Velocidad_kmh" || c === "Angulo_grados" || c === "Distancia_Portero_m"){
      vec.push(row[c] !== undefined ? row[c] : 0.0);
    } else {
      // categorical one-hot names like "Presion_Partido_Alta" or "Pie_Dominante_Derecho"
      // parse column name
      const parts = c.split("_");
      const catName = parts[0] + "_" + parts[1]; // e.g. "Presion_Partido_Alta" -> ["Presion","Partido","Alta"] => need original format
      // We'll fallback: check if c includes '=' pattern (if meta used c="Presion_Partido=Alta") or underscores for value
      // Try variants:
      let matched = false;
      if (c.includes("=")){
        const [k, v] = c.split("=");
        const key = k.trim();
        const val = v.trim();
        if (row[key] === val) { vec.push(1.0); matched = true; } else { vec.push(0.0); matched = true; }
      } else {
        // Try splitting last underscore as value
        const lastUnd = c.lastIndexOf("_");
        if (lastUnd > 0){
          const key = c.substring(0, lastUnd);
          const val = c.substring(lastUnd+1);
          if (row[key] !== undefined && (""+row[key]).toLowerCase() === val.toLowerCase()){
            vec.push(1.0);
          } else {
            vec.push(0.0);
          }
          matched = true;
        }
      }
      if (!matched) vec.push(0.0);
    }
  }
  return vec;
}

function scaleNumeric(vec){
  // meta.scaler: { numeric_cols: [...], mean: [...], scale: [...] }
  if(!meta || !meta.scaler) return vec;
  const means = meta.scaler.mean;
  const scales = meta.scaler.scale;
  const numericCols = meta.scaler.numeric_cols;
  // numeric positions in feature_columns
  const cols = meta.feature_columns;
  const out = vec.slice(); // clone
  for (let i=0;i<numericCols.length;i++){
    const colName = numericCols[i];
    const idx = cols.indexOf(colName);
    if (idx >= 0){
      const mean = means[i];
      const scale = scales[i] || 1.0;
      out[idx] = (out[idx] - mean) / scale;
    }
  }
  return out;
}

async function predictFromInputs(){
  // read DOM
  const vel = parseFloat(document.getElementById("vel").value);
  const ang = parseFloat(document.getElementById("ang").value);
  const dist = parseFloat(document.getElementById("dist").value);
  const presion = document.getElementById("presion").value;
  const pie = document.getElementById("pie").value;

  // validate ranges
  if (isNaN(vel) || vel < RANGES.Velocidad_kmh[0] || vel > RANGES.Velocidad_kmh[1]){
    showError(`Velocidad fuera de rango (${RANGES.Velocidad_kmh[0]}–${RANGES.Velocidad_kmh[1]})`);
    return;
  }
  if (isNaN(ang) || ang < RANGES.Angulo_grados[0] || ang > RANGES.Angulo_grados[1]){
    showError(`Ángulo fuera de rango (${RANGES.Angulo_grados[0]}–${RANGES.Angulo_grados[1]})`);
    return;
  }
  if (isNaN(dist) || dist < RANGES.Distancia_Portero_m[0] || dist > RANGES.Distancia_Portero_m[1]){
    showError(`Distancia fuera de rango (${RANGES.Distancia_Portero_m[0]}–${RANGES.Distancia_Portero_m[1]})`);
    return;
  }

  if (!meta || !model){
    showError("Modelo o metadatos no cargados.");
    return;
  }

  // build vector
  const rawVec = buildInputVector({Velocidad_kmh:vel, Angulo_grados:ang, Distancia_Portero_m:dist, Presion_Partido:presion, Pie_Dominante:pie});
  const scaled = scaleNumeric(rawVec);
  // to tensor
  const t = tf.tensor2d([scaled]);
  const pred = model.predict(t);
  const arr = await pred.array();
  const prob = arr[0][0];
  const label = prob >= 0.5 ? "GOL" : "FALLADO";
  showResult(`${label} — Probabilidad de gol: ${(prob*100).toFixed(1)}%`);
}

document.getElementById("predictBtn").addEventListener("click", predictFromInputs);

// Cargar en inicio
loadMetaAndModel();
