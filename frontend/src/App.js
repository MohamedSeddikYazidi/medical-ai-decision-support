import React, { useState, useCallback, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const r = await fetch(`${API}${path}`);
  return r.ok ? r.json() : null;
}

async function apiPredict(body, modelName) {
  const url = modelName ? `${API}/predict?model=${modelName}` : `${API}/predict`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Prediction failed");
  return data;
}

// ── Constants ─────────────────────────────────────────────────────────────────
const AGE_OPTS = [
  "[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
  "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)",
];

const ADMISSION_TYPES = {
  1: "Emergency", 2: "Urgent", 3: "Elective",
  4: "Newborn", 5: "Not Available", 6: "NULL", 7: "Trauma", 8: "Not Mapped",
};

const DISCHARGE_OPTS = {
  1:  "Home / Self Care",
  2:  "Short-term Hospital",
  3:  "SNF (Skilled Nursing)",
  4:  "Intermediate Care",
  5:  "Inpatient Rehab",
  6:  "Home Health Care",
  11: "Expired",
  13: "Hospice / Home",
  14: "Hospice / Medical",
  22: "Rehab Facility",
  25: "Not Mapped",
};

const RISK = {
  "Low Risk":    { color: "#00d4aa", bg: "rgba(0,212,170,0.10)", icon: "●" },
  "Medium Risk": { color: "#f0a500", bg: "rgba(240,165,0,0.10)",  icon: "◆" },
  "High Risk":   { color: "#ff4757", bg: "rgba(255,71,87,0.10)",  icon: "▲" },
};

const MODEL_COLOR = {
  logistic_regression: "#3d7eff",
  random_forest:       "#00d4aa",
  xgboost:             "#f0a500",
  lightgbm:            "#a855f7",
};
const MODEL_LABEL = {
  logistic_regression: "Logistic Regression",
  random_forest:       "Random Forest",
  xgboost:             "XGBoost",
  lightgbm:            "LightGBM ★",
};

const BAR_FEATURES = [
  { key: "time_in_hospital",   label: "Days in Hospital" },
  { key: "num_lab_procedures", label: "Lab Procedures"   },
  { key: "num_medications",    label: "Medications"      },
  { key: "number_outpatient",  label: "Outpatient Visits"},
  { key: "number_emergency",   label: "Emergency Visits" },
  { key: "number_inpatient",   label: "Prior Inpatient"  },
  { key: "num_diagnoses",      label: "# Diagnoses"      },
];

const BAR_COLORS = ["#3d7eff","#6366f1","#8b5cf6","#00d4aa","#f0a500","#ff4757","#ec4899"];

const DEFAULTS = {
  age: "[50-60)", race: "Caucasian", gender: "Male",
  time_in_hospital: 3, num_lab_procedures: 35, num_procedures: 1,
  num_medications: 8, num_diagnoses: 3,
  number_outpatient: 0, number_emergency: 0, number_inpatient: 0,
  diag_1: "250", diag_2: "401", diag_3: "428",
  A1Cresult: "None", max_glu_serum: "None",
  admission_type_id: 3, discharge_disposition_id: 1,
  insulin: "Steady", change: "No", diabetesMed: "Yes",
};

// ── CSS ───────────────────────────────────────────────────────────────────────
const CSS = `
  *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
  :root {
    --void: #060b18; --deep: #0a1020; --panel: #0d1526;
    --card: #111928;  --input: #18233a;
    --bdr: rgba(255,255,255,0.07); --bdr-hi: rgba(100,160,255,0.22);
    --text: #e2e8f8; --mid: #8898b8; --muted: #445570;
    --accent: #3d7eff; --glow: #3d7eff33;
    --teal: #00d4aa; --gold: #f0a500; --red: #ff4757; --purple: #a855f7;
    --serif: 'DM Serif Display', Georgia, serif;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Inter', system-ui, sans-serif;
    --r: 10px;
  }
  html,body,#root { height:100%; background:var(--void); color:var(--text);
    font-family:var(--sans); font-size:14px; -webkit-font-smoothing:antialiased; }
  ::-webkit-scrollbar{width:4px}
  ::-webkit-scrollbar-thumb{background:var(--bdr-hi);border-radius:2px}

  .app { min-height:100vh; display:grid; grid-template-rows:auto 1fr; }

  /* Header */
  .hdr { height:58px; background:var(--deep); border-bottom:1px solid var(--bdr);
    padding:0 1.5rem; display:flex; align-items:center; justify-content:space-between;
    position:sticky; top:0; z-index:200; }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand-ico { width:32px; height:32px; border-radius:9px;
    background:linear-gradient(135deg,var(--accent),#7c3aed);
    display:flex; align-items:center; justify-content:center; font-size:16px; }
  .brand-name { font-family:var(--serif); font-size:1.15rem; }
  .brand-name b { color:var(--accent); font-weight:400; }
  .hdr-mid { font-family:var(--mono); font-size:11px; color:var(--muted);
    letter-spacing:.1em; display:flex; gap:1.5rem; }
  .hdr-mid span { color:var(--mid); }
  .status { display:flex; align-items:center; gap:7px;
    font-family:var(--mono); font-size:11px; color:var(--muted); }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--teal);
    box-shadow:0 0 7px var(--teal); animation:pulse 2s infinite; }
  .dot.off { background:var(--red); box-shadow:0 0 7px var(--red); }
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

  /* Layout */
  .body { display:grid; grid-template-columns:390px 1fr; min-height:0; }
  .sidebar { background:var(--panel); border-right:1px solid var(--bdr);
    overflow-y:auto; padding:1.1rem; display:flex; flex-direction:column; gap:.85rem; }
  .main { overflow-y:auto; padding:1.1rem; display:flex; flex-direction:column; gap:.85rem; }

  /* Section title */
  .sec { font-family:var(--mono); font-size:10px; font-weight:600; color:var(--muted);
    text-transform:uppercase; letter-spacing:.15em; margin-bottom:.6rem;
    display:flex; align-items:center; gap:8px; }
  .sec::after { content:''; flex:1; height:1px; background:var(--bdr); }

  /* Card */
  .card { background:var(--card); border:1px solid var(--bdr); border-radius:var(--r);
    padding:1rem; transition:border-color .18s; }
  .card:hover { border-color:var(--bdr-hi); }

  /* Form fields */
  .fld { margin-bottom:.65rem; }
  .fld label { display:block; font-size:10px; font-weight:600; color:var(--mid);
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
  .fld input, .fld select {
    width:100%; background:var(--input); border:1px solid var(--bdr);
    border-radius:7px; color:var(--text); font-family:var(--mono); font-size:12.5px;
    padding:7px 9px; outline:none; transition:border-color .18s, box-shadow .18s;
    appearance:none; }
  .fld input:focus, .fld select:focus {
    border-color:var(--accent); box-shadow:0 0 0 3px var(--glow); }
  .fld select option { background:var(--card); }
  .g2 { display:grid; grid-template-columns:1fr 1fr; gap:0 .6rem; }
  .g3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0 .6rem; }

  /* Model selector */
  .model-list { display:flex; flex-direction:column; gap:5px; }
  .mbtn { display:flex; align-items:center; justify-content:space-between;
    padding:9px 11px; background:var(--input); border:1px solid var(--bdr);
    border-radius:8px; cursor:pointer; transition:all .15s;
    color:var(--mid); font-family:var(--mono); font-size:11px; }
  .mbtn:hover { border-color:var(--bdr-hi); color:var(--text); }
  .mbtn.active { border-color:var(--accent); background:rgba(61,126,255,.07);
    color:var(--text); }
  .mbtn-left { display:flex; align-items:center; gap:8px; }
  .mdot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .mbtn-right { font-size:10px; color:var(--muted); text-align:right; line-height:1.5; }
  .badge { font-family:var(--mono); font-size:9px; padding:1px 7px; border-radius:10px;
    background:rgba(61,126,255,.13); color:var(--accent);
    border:1px solid rgba(61,126,255,.28); margin-left:6px; }
  .badge.teal { background:rgba(0,212,170,.1); color:var(--teal);
    border-color:rgba(0,212,170,.3); }
  .badge.purple { background:rgba(168,85,247,.12); color:var(--purple);
    border-color:rgba(168,85,247,.3); }

  /* Submit button */
  .btn { width:100%; padding:11px; border:none; border-radius:9px;
    background:linear-gradient(135deg,var(--accent),#6d28d9); color:#fff;
    font-family:var(--mono); font-size:12px; font-weight:600; letter-spacing:.1em;
    cursor:pointer; transition:opacity .2s, transform .15s;
    box-shadow:0 4px 20px rgba(61,126,255,.28); }
  .btn:hover:not(:disabled) { opacity:.88; transform:translateY(-1px); }
  .btn:disabled { opacity:.45; cursor:not-allowed; }
  .spin { display:inline-block; width:12px; height:12px; border:2px solid rgba(255,255,255,.3);
    border-top-color:#fff; border-radius:50%; animation:rot .65s linear infinite;
    vertical-align:middle; margin-right:6px; }
  @keyframes rot{to{transform:rotate(360deg)}}

  /* Gauge — pure CSS arc, no SVG overlap issues */
  .gauge-wrap { display:flex; flex-direction:column; align-items:center;
    gap:.6rem; padding:1.4rem .5rem .8rem; }
  .gauge-ring { position:relative; width:160px; height:85px; overflow:hidden; }
  .gauge-track { width:160px; height:160px; border-radius:50%;
    border:14px solid rgba(255,255,255,.06); position:absolute; top:0; left:0; }
  .gauge-fill { width:160px; height:160px; border-radius:50%;
    border:14px solid transparent; position:absolute; top:0; left:0;
    transform-origin:center center; }
  .gauge-pct { font-family:var(--serif); font-size:2.8rem; line-height:1;
    letter-spacing:-.02em; text-align:center; }
  .gauge-sub { font-family:var(--mono); font-size:9.5px; color:var(--muted);
    text-transform:uppercase; letter-spacing:.15em; }
  .risk-badge { display:inline-flex; align-items:center; gap:5px;
    padding:4px 13px; border-radius:20px; font-family:var(--mono);
    font-size:10.5px; font-weight:600; letter-spacing:.1em; border:1px solid currentColor; }

  /* Stats */
  .stat-grid { display:grid; grid-template-columns:repeat(3,1fr);
    gap:1px; background:var(--bdr); border-radius:8px; overflow:hidden; }
  .stat { background:var(--card); padding:.8rem; text-align:center; }
  .stat-v { font-family:var(--mono); font-size:1.15rem; font-weight:500; color:var(--accent); }
  .stat-k { font-size:9px; color:var(--muted); text-transform:uppercase;
    letter-spacing:.07em; margin-top:2px; }

  /* Comparison table */
  .cmp { width:100%; border-collapse:collapse; }
  .cmp th { font-family:var(--mono); font-size:9.5px; text-transform:uppercase;
    letter-spacing:.1em; color:var(--muted); padding:5px 9px;
    border-bottom:1px solid var(--bdr); text-align:left; }
  .cmp td { padding:8px 9px; font-family:var(--mono); font-size:11.5px;
    color:var(--mid); border-bottom:1px solid rgba(255,255,255,.03); }
  .cmp tr:last-child td { border-bottom:none; }
  .cmp .best td { color:var(--text); background:rgba(61,126,255,.04); }
  .hi { color:var(--teal) !important; }

  /* Notes */
  .notes { background:rgba(61,126,255,.05); border:1px solid rgba(61,126,255,.13);
    border-left:3px solid var(--accent); border-radius:7px;
    padding:.75rem .9rem; font-size:12.5px; color:var(--mid); line-height:1.75; }
  .alert { background:rgba(255,71,87,.08); border:1px solid rgba(255,71,87,.25);
    border-radius:7px; padding:.7rem .9rem; color:#ff7b87;
    font-size:12px; display:flex; gap:8px; }

  /* Empty state */
  .empty { flex:1; display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center; gap:.8rem; opacity:.4; padding:3rem; }
  .empty-ico { font-size:3.5rem; opacity:.3; }
  .empty-title { font-family:var(--serif); font-size:1.3rem; color:var(--mid); }
  .empty-sub { font-size:12px; color:var(--muted); max-width:260px; }

  /* History */
  .hist-row { display:flex; align-items:center; justify-content:space-between;
    padding:6px 9px; background:var(--input); border-radius:7px;
    font-size:11.5px; gap:.5rem; }

  /* Pill */
  .pill { display:inline-flex; align-items:center; gap:4px; font-family:var(--mono);
    font-size:10px; padding:2px 9px; border-radius:10px;
    background:rgba(255,255,255,.04); color:var(--mid); border:1px solid var(--bdr); }
  .row { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }

  @keyframes up { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
  .anim  { animation:up .3s ease both; }
  .d1 { animation-delay:.05s; } .d2 { animation-delay:.1s; } .d3 { animation-delay:.15s; }

  @media(max-width:860px) {
    .body { grid-template-columns:1fr; }
    .sidebar { border-right:none; border-bottom:1px solid var(--bdr); }
    .hdr-mid { display:none; }
  }
`;

// ── Utility components ────────────────────────────────────────────────────────
function Fld({ label, name, value, onChange, type = "number", opts, min, max, step }) {
  return (
    <div className="fld">
      <label>{label}</label>
      {opts
        ? <select name={name} value={value} onChange={onChange}>
            {opts.map(([v, l]) => <option key={v} value={v}>{l || v}</option>)}
          </select>
        : <input type={type} name={name} value={value}
                 min={min} max={max} step={step} onChange={onChange} />}
    </div>
  );
}

// CSS-only semicircle gauge — no SVG text overlap
function Gauge({ prob, risk }) {
  const cfg  = RISK[risk] || RISK["Low Risk"];
  const pct  = Math.round(prob * 100);
  // semicircle: 0% = left, 100% = right. Achieved with border coloring + rotation.
  // We clip the bottom half, rotate the fill border.
  const deg  = Math.round(pct * 1.8); // 0-100 maps to 0-180 deg
  const fillStyle = {
    borderColor: `${cfg.color} ${cfg.color} transparent transparent`,
    transform: `rotate(${-90 + deg}deg)`,
    boxShadow: `0 0 16px ${cfg.color}55`,
  };
  return (
    <div className="gauge-wrap">
      <div className="gauge-ring">
        <div className="gauge-track" />
        <div className="gauge-fill" style={fillStyle} />
      </div>
      <div className="gauge-pct" style={{ color: cfg.color }}>{pct}%</div>
      <div className="gauge-sub">Readmission Probability</div>
      <span className="risk-badge" style={{
        color: cfg.color, borderColor: cfg.color + "55", background: cfg.bg,
      }}>
        {cfg.icon} {risk.toUpperCase()}
      </span>
    </div>
  );
}

function ModelSelector({ models, selected, onSelect }) {
  return (
    <div>
      <div className="sec">Select Model</div>
      <div className="card">
        <div className="model-list">
          <div className={`mbtn ${!selected ? "active" : ""}`} onClick={() => onSelect(null)}>
            <div className="mbtn-left">
              <div className="mdot" style={{ background: "var(--accent)" }} />
              Best Model (auto)
              <span className="badge">AUTO</span>
            </div>
          </div>
          {models.map(m => (
            <div key={m.model_name}
                 className={`mbtn ${selected === m.model_name ? "active" : ""}`}
                 onClick={() => onSelect(m.model_name)}>
              <div className="mbtn-left">
                <div className="mdot" style={{ background: MODEL_COLOR[m.model_name] || "var(--mid)" }} />
                {MODEL_LABEL[m.model_name] || m.model_name}
                {m.is_best && <span className="badge">BEST</span>}
                {m.model_name === "lightgbm" && !m.is_best &&
                  <span className="badge purple">NEW</span>}
              </div>
              <div className="mbtn-right">
                AUC {m.val_roc_auc.toFixed(3)}<br/>F1 {m.val_f1.toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ComparisonTable({ models, activeName }) {
  if (!models || models.length === 0) return null;
  return (
    <div className="card anim d3">
      <div className="sec">Model Comparison</div>
      <table className="cmp">
        <thead>
          <tr>
            <th>Model</th><th>Val AUC</th><th>Val F1</th><th>Test AUC</th>
          </tr>
        </thead>
        <tbody>
          {models.map(m => (
            <tr key={m.model_name} className={m.is_best ? "best" : ""}>
              <td>
                <div style={{ display:"flex", alignItems:"center", gap:7 }}>
                  <div style={{ width:7, height:7, borderRadius:"50%",
                    background: MODEL_COLOR[m.model_name] || "var(--mid)" }} />
                  {MODEL_LABEL[m.model_name] || m.model_name}
                  {m.is_best && <span className="badge" style={{marginLeft:4}}>BEST</span>}
                  {m.model_name === activeName &&
                    <span className="badge teal" style={{marginLeft:4}}>ACTIVE</span>}
                </div>
              </td>
              <td className={m.is_best ? "hi" : ""}>{m.val_roc_auc.toFixed(4)}</td>
              <td>{m.val_f1.toFixed(4)}</td>
              <td>{m.test_roc_auc.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [form, setForm]       = useState(DEFAULTS);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [health, setHealth]   = useState(null);
  const [models, setModels]   = useState([]);
  const [selModel, setSelModel] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    apiFetch("/health").then(h => h && setHealth(h));
    apiFetch("/models/list").then(d => d && setModels(d.available || []));
  }, []);

  const onChange = useCallback(e => {
    const { name, value } = e.target;
    setForm(p => ({ ...p, [name]: value }));
  }, []);

  const onSubmit = useCallback(async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const body = {
        ...form,
        time_in_hospital:        Number(form.time_in_hospital),
        num_lab_procedures:      Number(form.num_lab_procedures),
        num_procedures:          Number(form.num_procedures),
        num_medications:         Number(form.num_medications),
        num_diagnoses:           Number(form.num_diagnoses),
        number_outpatient:       Number(form.number_outpatient),
        number_emergency:        Number(form.number_emergency),
        number_inpatient:        Number(form.number_inpatient),
        admission_type_id:       Number(form.admission_type_id),
        discharge_disposition_id: Number(form.discharge_disposition_id),
      };
      const res = await apiPredict(body, selModel);
      setResult(res);
      setHistory(p => [{ ...res, ts: new Date().toLocaleTimeString(), age: form.age }, ...p.slice(0,4)]);
    } catch(e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [form, selModel]);

  const cfg = result ? (RISK[result.risk_level] || RISK["Low Risk"]) : null;
  const activeModel = result?.model_name || selModel;

  return (
    <>
      <style>{CSS}</style>
      <div className="app">

        {/* Header */}
        <header className="hdr">
          <div className="brand">
            <div className="brand-ico">⚕</div>
            <div className="brand-name">Clinical<b>AI</b></div>
          </div>
          <div className="hdr-mid">
            <span>DATASET <b>Diabetes 130-US Hospitals</b></span>
            <span>MODELS <b>{models.length || "—"}</b></span>
            <span>FEATURES <b>19+</b></span>
          </div>
          <div className="status">
            <div className={`dot ${health?.model_loaded === false ? "off" : ""}`} />
            {health?.model_loaded ? `MODEL ONLINE · ${(health.model_name||"").toUpperCase()}` : "CONNECTING…"}
          </div>
        </header>

        <div className="body">

          {/* ── Sidebar ─────────────────────────────────────────────────── */}
          <aside className="sidebar">

            <ModelSelector models={models} selected={selModel} onSelect={setSelModel} />

            {/* Demographics */}
            <div>
              <div className="sec">Demographics</div>
              <div className="card">
                <Fld label="Age Bracket" name="age" value={form.age} onChange={onChange}
                     opts={AGE_OPTS.map(v=>[v,v])} />
                <div className="g2">
                  <Fld label="Race" name="race" value={form.race} onChange={onChange}
                       opts={[["Caucasian"],["AfricanAmerican"],["Hispanic"],["Asian"],["Other"]]} />
                  <Fld label="Gender" name="gender" value={form.gender} onChange={onChange}
                       opts={[["Male"],["Female"]]} />
                </div>
              </div>
            </div>

            {/* Admission Context */}
            <div>
              <div className="sec">Admission Context</div>
              <div className="card">
                <div className="g2">
                  <Fld label="Admission Type" name="admission_type_id"
                       value={form.admission_type_id} onChange={onChange}
                       opts={Object.entries(ADMISSION_TYPES).map(([v,l])=>[v,l])} />
                  <Fld label="# Diagnoses" name="num_diagnoses"
                       value={form.num_diagnoses} onChange={onChange} min={1} max={16} />
                </div>
                <Fld label="Discharge To" name="discharge_disposition_id"
                     value={form.discharge_disposition_id} onChange={onChange}
                     opts={Object.entries(DISCHARGE_OPTS).map(([v,l])=>[v,l])} />
              </div>
            </div>

            {/* Encounter */}
            <div>
              <div className="sec">Encounter Details</div>
              <div className="card">
                <div className="g2">
                  <Fld label="Days in Hospital" name="time_in_hospital"
                       value={form.time_in_hospital} onChange={onChange} min={1} max={14} />
                  <Fld label="Lab Procedures" name="num_lab_procedures"
                       value={form.num_lab_procedures} onChange={onChange} min={0} max={132} />
                  <Fld label="Procedures" name="num_procedures"
                       value={form.num_procedures} onChange={onChange} min={0} max={6} />
                  <Fld label="Medications" name="num_medications"
                       value={form.num_medications} onChange={onChange} min={0} max={81} />
                  <Fld label="Outpatient Visits" name="number_outpatient"
                       value={form.number_outpatient} onChange={onChange} min={0} />
                  <Fld label="Emergency Visits" name="number_emergency"
                       value={form.number_emergency} onChange={onChange} min={0} />
                  <Fld label="Prior Inpatient" name="number_inpatient"
                       value={form.number_inpatient} onChange={onChange} min={0} />
                </div>
              </div>
            </div>

            {/* Lab Results */}
            <div>
              <div className="sec">Lab Results</div>
              <div className="card">
                <div className="g2">
                  <Fld label="A1C Result" name="A1Cresult" value={form.A1Cresult} onChange={onChange}
                       opts={[["None","Not Tested"],[">7","High (>7%)"],[">8","Very High (>8%)"],["Norm","Normal"]]} />
                  <Fld label="Glucose Serum" name="max_glu_serum" value={form.max_glu_serum} onChange={onChange}
                       opts={[["None","Not Tested"],[">200","High (>200)"],[">300","Very High (>300)"],["Norm","Normal"]]} />
                </div>
              </div>
            </div>

            {/* Diagnoses */}
            <div>
              <div className="sec">Diagnoses (ICD-9)</div>
              <div className="card">
                <div className="g3">
                  <Fld label="Primary Dx"   name="diag_1" type="text" value={form.diag_1} onChange={onChange} />
                  <Fld label="Secondary Dx" name="diag_2" type="text" value={form.diag_2} onChange={onChange} />
                  <Fld label="Tertiary Dx"  name="diag_3" type="text" value={form.diag_3} onChange={onChange} />
                </div>
              </div>
            </div>

            {/* Medications */}
            <div>
              <div className="sec">Medications</div>
              <div className="card">
                <div className="g3">
                  <Fld label="Insulin" name="insulin" value={form.insulin} onChange={onChange}
                       opts={[["No","No"],["Steady","Steady"],["Up","Up"],["Down","Down"]]} />
                  <Fld label="Med Change" name="change" value={form.change} onChange={onChange}
                       opts={[["No","No Change"],["Ch","Changed"]]} />
                  <Fld label="Diabetes Med" name="diabetesMed" value={form.diabetesMed} onChange={onChange}
                       opts={[["Yes","Yes"],["No","No"]]} />
                </div>
              </div>
            </div>

            {error && <div className="alert"><span>⚠</span><span>{error}</span></div>}

            <button className="btn" onClick={onSubmit} disabled={loading}>
              {loading && <span className="spin" />}
              {loading ? "ANALYSING…"
                : `PREDICT RISK${selModel ? " · " + (MODEL_LABEL[selModel]||selModel).replace(" ★","").toUpperCase() : ""}`}
            </button>

          </aside>

          {/* ── Main Content ─────────────────────────────────────────────── */}
          <main className="main">
            {!result ? (
              <div className="empty">
                <div className="empty-ico">🏥</div>
                <div className="empty-title">No Prediction Yet</div>
                <div className="empty-sub">
                  Select a model, complete the patient form, and click Predict.
                </div>
              </div>
            ) : (
              <>
                {/* Gauge */}
                <div className="card anim" style={{ borderColor: cfg.color + "33" }}>
                  <div className="sec">Risk Assessment</div>
                  <Gauge prob={result.readmission_probability} risk={result.risk_level} />
                  <div className="row" style={{ justifyContent:"center", marginTop:".3rem" }}>
                    <span className="pill">◎ {result.confidence?.toUpperCase()} CONFIDENCE</span>
                    <span className="pill" style={{ color: MODEL_COLOR[result.model_name] }}>
                      ⚙ {(MODEL_LABEL[result.model_name]||result.model_name).toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Stats */}
                <div className="card anim d1">
                  <div className="sec">Patient Metrics</div>
                  <div className="stat-grid">
                    {[
                      ["time_in_hospital",   "Days", "d", "var(--teal)"],
                      ["num_medications",    "Meds",  "", "var(--gold)"],
                      ["number_emergency",   "ER",    "", form.number_emergency > 0 ? "var(--red)" : "var(--mid)"],
                      ["num_lab_procedures", "Labs",  "", "var(--accent)"],
                      ["number_inpatient",   "Prior Admits", "", "var(--mid)"],
                      ["num_diagnoses",      "Diagnoses", "", "var(--purple)"],
                    ].map(([k, label, suffix, color]) => (
                      <div className="stat" key={k}>
                        <div className="stat-v" style={{ color }}>{form[k]}{suffix}</div>
                        <div className="stat-k">{label}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature bar chart */}
                <div className="card anim d2">
                  <div className="sec">Clinical Profile</div>
                  <ResponsiveContainer width="100%" height={210}>
                    <BarChart
                      layout="vertical"
                      data={BAR_FEATURES.map(f => ({ name: f.label, v: Number(form[f.key]) || 0 })).filter(d => d.v > 0)}
                      margin={{ left:8, right:18, top:4, bottom:4 }}
                    >
                      <XAxis type="number" tick={{ fill:"#445570", fontSize:10 }}
                             axisLine={false} tickLine={false} />
                      <YAxis dataKey="name" type="category" width={120}
                             tick={{ fill:"#8898b8", fontSize:11 }}
                             axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{
                        background:"#111928", border:"1px solid rgba(255,255,255,.07)",
                        borderRadius:7, fontSize:12 }}
                        cursor={{ fill:"rgba(255,255,255,.03)" }} />
                      <Bar dataKey="v" radius={[0,4,4,0]}>
                        {BAR_FEATURES.map((_, i) =>
                          <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Comparison */}
                <ComparisonTable models={models} activeName={activeModel} />

                {/* Notes */}
                <div className="card anim d3">
                  <div className="sec">Clinical Decision Support Notes</div>
                  <div className="notes">{result.clinical_notes}</div>
                </div>

                {/* History */}
                {history.length > 1 && (
                  <div className="card anim">
                    <div className="sec">Recent Predictions</div>
                    <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                      {history.map((h, i) => {
                        const hc = RISK[h.risk_level] || RISK["Low Risk"];
                        return (
                          <div className="hist-row" key={i}>
                            <span style={{ color:"var(--muted)", fontFamily:"var(--mono)", fontSize:10 }}>{h.ts}</span>
                            <span style={{ color:"var(--mid)", fontSize:11 }}>{h.age}</span>
                            <span style={{ fontFamily:"var(--mono)", fontSize:10, color: MODEL_COLOR[h.model_name] }}>
                              {MODEL_LABEL[h.model_name]||h.model_name}
                            </span>
                            <span style={{ fontFamily:"var(--mono)", fontWeight:600, color:hc.color }}>
                              {Math.round(h.readmission_probability*100)}%
                            </span>
                            <span className="risk-badge" style={{
                              color:hc.color, borderColor:hc.color+"44",
                              background:hc.bg, padding:"2px 8px", fontSize:9
                            }}>
                              {h.risk_level}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </main>
        </div>
      </div>
    </>
  );
}