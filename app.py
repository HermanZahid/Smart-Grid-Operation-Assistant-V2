
import json
import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Renewable Grid Operations Agent", page_icon="⚡", layout="wide")

REQUIRED_COLUMNS = [
    "hour", "load_mw", "solar_mw", "wind_mw",
    "battery_capacity_mwh", "max_charge_mw", "max_discharge_mw",
    "initial_soc_percent", "min_soc_percent", "max_soc_percent",
    "charge_efficiency", "discharge_efficiency", "grid_import_limit_mw"
]


# ---------- POWER-SYSTEM MODEL ----------

def validate_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, "Missing columns: " + ", ".join(missing)
    if len(df) != 24:
        return False, "The demo expects exactly 24 hourly rows."
    if sorted(df["hour"].astype(int).tolist()) != list(range(24)):
        return False, "The hour column must contain 0 through 23."
    return True, ""


def simulate(df, battery_capacity=None, max_charge=None, max_discharge=None):
    d = df.sort_values("hour").reset_index(drop=True).copy()

    capacity = float(battery_capacity or d["battery_capacity_mwh"].iloc[0])
    charge_limit = float(max_charge or d["max_charge_mw"].iloc[0])
    discharge_limit = float(max_discharge or d["max_discharge_mw"].iloc[0])

    soc = float(d["initial_soc_percent"].iloc[0]) / 100
    min_soc = float(d["min_soc_percent"].iloc[0]) / 100
    max_soc = float(d["max_soc_percent"].iloc[0]) / 100
    eta_c = float(d["charge_efficiency"].iloc[0])
    eta_d = float(d["discharge_efficiency"].iloc[0])
    grid_limit = float(d["grid_import_limit_mw"].iloc[0])

    rows = []

    for _, r in d.iterrows():
        load = float(r["load_mw"])
        solar = float(r["solar_mw"])
        wind = float(r["wind_mw"])
        renewable = solar + wind
        net_load = load - renewable

        charge = discharge = grid_import = curtailment = unmet = 0.0

        if net_load < 0:
            surplus = -net_load
            max_energy_charge = max(0, (max_soc - soc) * capacity / eta_c)
            charge = min(charge_limit, surplus, max_energy_charge)
            soc += charge * eta_c / capacity
            curtailment = max(0, surplus - charge)
        else:
            max_energy_discharge = max(0, (soc - min_soc) * capacity * eta_d)
            discharge = min(discharge_limit, net_load, max_energy_discharge)
            soc -= discharge / (capacity * eta_d)
            grid_import = max(0, net_load - discharge)
            unmet = max(0, grid_import - grid_limit)

        soc = min(max(soc, min_soc), max_soc)

        rows.append({
            "hour": int(r["hour"]),
            "load_mw": load,
            "solar_mw": solar,
            "wind_mw": wind,
            "renewable_mw": renewable,
            "net_load_mw": net_load,
            "battery_charge_mw": charge,
            "battery_discharge_mw": discharge,
            "soc_percent": soc * 100,
            "grid_import_mw": grid_import,
            "grid_limit_mw": grid_limit,
            "curtailment_mwh": curtailment,
            "unmet_load_mwh": unmet,
        })

    out = pd.DataFrame(rows)
    out["renewable_penetration_percent"] = out["renewable_mw"] / out["load_mw"] * 100
    out["net_load_ramp_mw"] = out["net_load_mw"].diff().fillna(0)
    return out


def metrics(r):
    idx = r["net_load_ramp_mw"].idxmax()
    return {
        "peak_load": float(r["load_mw"].max()),
        "peak_net_load": float(r["net_load_mw"].max()),
        "max_grid_import": float(r["grid_import_mw"].max()),
        "grid_limit": float(r["grid_limit_mw"].iloc[0]),
        "curtailment": float(r["curtailment_mwh"].sum()),
        "unmet": float(r["unmet_load_mwh"].sum()),
        "min_soc": float(r["soc_percent"].min()),
        "max_soc": float(r["soc_percent"].max()),
        "renewable_penetration": float(r["renewable_mw"].sum() / r["load_mw"].sum() * 100),
        "max_ramp": float(r["net_load_ramp_mw"].max()),
        "ramp_hour": int(r.loc[idx, "hour"]),
    }


def critical_hours(r):
    m = metrics(r)
    threshold = max(10, m["max_ramp"] * 0.75)

    mask = (
        (r["grid_import_mw"] > r["grid_limit_mw"] * 0.95)
        | (r["soc_percent"] <= m["min_soc"] + 0.01)
        | (r["curtailment_mwh"] > 0)
        | (r["net_load_ramp_mw"] >= threshold)
    )
    c = r.loc[mask].copy()

    reasons = []
    for _, x in c.iterrows():
        reason = []
        if x["grid_import_mw"] > x["grid_limit_mw"] * 0.95:
            reason.append("high grid import")
        if x["soc_percent"] <= m["min_soc"] + 0.01:
            reason.append("battery near minimum SOC")
        if x["curtailment_mwh"] > 0:
            reason.append("renewable curtailment")
        if x["net_load_ramp_mw"] >= threshold:
            reason.append("large net-load ramp")
        reasons.append(", ".join(reason))

    c["reason"] = reasons
    return c


# ---------- AGENT TOOLS ----------

def grid_analysis(r):
    m = metrics(r)
    return {
        "peak_load_mw": round(m["peak_load"], 1),
        "peak_net_load_mw": round(m["peak_net_load"], 1),
        "max_grid_import_mw": round(m["max_grid_import"], 1),
        "grid_import_limit_mw": round(m["grid_limit"], 1),
        "renewable_penetration_percent": round(m["renewable_penetration"], 1),
        "curtailment_mwh": round(m["curtailment"], 1),
        "unmet_load_mwh": round(m["unmet"], 1),
        "max_upward_net_load_ramp_mw": round(m["max_ramp"], 1),
        "ramp_end_hour": m["ramp_hour"],
    }


def battery_analysis(r):
    m = metrics(r)
    charge = r["battery_charge_mw"]
    discharge = r["battery_discharge_mw"]

    return {
        "initial_soc_percent": round(float(r["soc_percent"].iloc[0]), 1),
        "minimum_soc_percent": round(m["min_soc"], 1),
        "minimum_soc_hour": int(r.loc[r["soc_percent"].idxmin(), "hour"]),
        "maximum_soc_percent": round(m["max_soc"], 1),
        "maximum_soc_hour": int(r.loc[r["soc_percent"].idxmax(), "hour"]),
        "charging_hours": int((charge > 0).sum()),
        "discharging_hours": int((discharge > 0).sum()),
        "total_charge_mwh": round(float(charge.sum()), 1),
        "total_discharge_mwh": round(float(discharge.sum()), 1),
        "max_discharge_mw": round(float(discharge.max()), 1),
        "hours_at_discharge_limit": int((discharge >= discharge.max() - 0.01).sum()) if discharge.max() > 0 else 0,
        "assessment": (
            "Battery reaches its minimum SOC and becomes energy-constrained."
            if m["min_soc"] <= 20.01
            else "Battery does not reach its minimum SOC."
        ),
    }


def problem_analysis(r):
    m = metrics(r)
    c = critical_hours(r)
    problems = []

    if m["max_grid_import"] > m["grid_limit"]:
        problems.append("Grid import exceeds the configured limit.")
    if m["curtailment"] > 0:
        problems.append("Renewable energy is curtailed during surplus periods.")
    if m["min_soc"] <= 20.01:
        problems.append("Battery reaches its minimum SOC.")
    if m["max_ramp"] > 10:
        problems.append("A significant upward net-load ramp occurs.")

    if not problems:
        problems.append("No major constraint was detected by the screening rules.")

    return {
        "problems": problems,
        "critical_hours": c["hour"].astype(int).tolist(),
        "critical_hour_count": int(len(c)),
    }


def recommendation(r):
    m = metrics(r)
    recs = []

    if m["min_soc"] <= 20.01:
        recs.append("Preserve more battery SOC before the evening ramp or increase battery energy capacity.")
    if m["curtailment"] > 0:
        recs.append("Use midday renewable surplus for charging; additional storage could reduce curtailment.")
    if m["max_grid_import"] > m["grid_limit"]:
        recs.append("Increase flexible evening resources through storage, demand flexibility, or other flexible generation.")
    if m["max_ramp"] > 10:
        recs.append("Prepare flexible resources for the evening net-load ramp.")

    return {"recommendations": recs or ["Continue monitoring renewable variability and battery SOC."]}


TOOLS = [
    {"type": "function", "function": {
        "name": "grid_analysis",
        "description": "Analyze peak load, grid import, renewable penetration, curtailment and net-load ramp.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "battery_analysis",
        "description": "Analyze battery SOC, charging, discharging and battery limitations.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "problem_analysis",
        "description": "Identify critical operating hours and main grid problems.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "recommendation",
        "description": "Generate engineering-oriented corrective strategy candidates.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


def execute_tool(name, r):
    return {
        "grid_analysis": grid_analysis,
        "battery_analysis": battery_analysis,
        "problem_analysis": problem_analysis,
        "recommendation": recommendation,
    }[name](r)


def ask_agent(question, base_r, scenario_r=None):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from Streamlit Secrets.")

    scenario_text = ""
    if scenario_r is not None:
        scenario_text = "\nBase case metrics:\n" + json.dumps(metrics(base_r))
        scenario_text += "\nScenario metrics:\n" + json.dumps(metrics(scenario_r))

    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": """
You are a renewable power-system operations analyst.
Investigate the user's question with the available engineering tools.
Do not invent numerical values. Use calculated tool results.
Explain physical causes and give concise, practical corrective strategies.
When scenario data is supplied, explicitly compare base case and scenario.
"""},
        {"role": "user", "content": question + scenario_text},
    ]

    first = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )

    used = []
    tool_messages = []

    if first.choices[0].message.tool_calls:
        messages.append(first.choices[0].message)

        for call in first.choices[0].message.tool_calls:
            name = call.function.name
            result = execute_tool(name, base_r)
            used.append((name, result))
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

        messages.extend(tool_messages)

        final = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.2,
        )
        answer = final.choices[0].message.content
    else:
        answer = first.choices[0].message.content

    return answer, used


# ---------- STREAMLIT UI ----------

st.title("⚡ Renewable Grid Operations Agent")
st.caption("AI-assisted decision support for renewable-integrated power-system operation")

uploaded = st.file_uploader("Upload the 24-hour grid input CSV", type=["csv"])

if uploaded is None:
    st.info("Upload grid_input_24h.csv to start.")
    st.stop()

df = pd.read_csv(uploaded)
ok, error = validate_data(df)

if not ok:
    st.error(error)
    st.stop()

base = simulate(df)
bm = metrics(base)

# 1. GRID HEALTH
st.header("1. Grid Health Dashboard")

health = [
    ("Peak Load", bm["peak_load"], "MW", bm["peak_load"] <= 150),
    ("Peak Grid Import", bm["max_grid_import"], "MW", bm["max_grid_import"] <= bm["grid_limit"]),
    ("Renewable Curtailment", bm["curtailment"], "MWh", bm["curtailment"] < 20),
    ("Minimum Battery SOC", bm["min_soc"], "%", bm["min_soc"] > 20.01),
    ("Unmet Load", bm["unmet"], "MWh", bm["unmet"] <= 0.01),
]

cols = st.columns(5)
for col, (label, value, unit, healthy) in zip(cols, health):
    col.metric(("🟢 " if healthy else "🔴 ") + label, f"{value:.1f} {unit}")

health_ok = all(x[3] for x in health)
if health_ok:
    st.success("Overall grid health: NORMAL")
else:
    st.warning("Overall grid health: ATTENTION REQUIRED")

# 2. CRITICAL HOURS
st.header("2. Critical Hours")
critical = critical_hours(base)

if critical.empty:
    st.success("No critical hours detected.")
else:
    st.dataframe(
        critical[[
            "hour", "load_mw", "renewable_mw", "net_load_mw",
            "battery_discharge_mw", "soc_percent", "grid_import_mw",
            "curtailment_mwh", "net_load_ramp_mw", "reason"
        ]].round(1),
        use_container_width=True,
        hide_index=True,
    )

st.info(
    f"Largest upward net-load ramp: {bm['max_ramp']:.1f} MW "
    f"ending at {bm['ramp_hour']:02d}:00."
)

# 3. SYSTEM OPERATION
st.header("3. System Operation")

st.subheader("Power balance")
st.line_chart(base.set_index("hour")[[
    "load_mw", "solar_mw", "wind_mw",
    "battery_discharge_mw", "grid_import_mw"
]])

st.subheader("Battery SOC and renewable curtailment")
st.line_chart(base.set_index("hour")[["soc_percent", "curtailment_mwh"]])

with st.expander("Intelligent battery analysis", expanded=True):
    ba = battery_analysis(base)
    st.write(f"**SOC:** {ba['minimum_soc_percent']:.1f}% minimum at {ba['minimum_soc_hour']:02d}:00; "
             f"{ba['maximum_soc_percent']:.1f}% maximum at {ba['maximum_soc_hour']:02d}:00.")
    st.write(f"**Operation:** {ba['charging_hours']} charging hours, {ba['discharging_hours']} discharging hours.")
    st.write(f"**Energy:** {ba['total_charge_mwh']:.1f} MWh charged, {ba['total_discharge_mwh']:.1f} MWh discharged.")
    st.write(f"**Constraint:** {ba['assessment']}")

# 4. WHAT-IF
st.header("4. What-if Scenario Simulator")
st.write("Change battery parameters and compare the scenario with the base case.")

c1, c2, c3 = st.columns(3)
capacity = c1.slider("Battery capacity (MWh)", 50, 200, int(df["battery_capacity_mwh"].iloc[0]), 10)
charge_limit = c2.slider("Max charge power (MW)", 10, 60, int(df["max_charge_mw"].iloc[0]), 5)
discharge_limit = c3.slider("Max discharge power (MW)", 10, 60, int(df["max_discharge_mw"].iloc[0]), 5)

scenario = simulate(
    df,
    battery_capacity=capacity,
    max_charge=charge_limit,
    max_discharge=discharge_limit,
)
sm = metrics(scenario)

comparison = pd.DataFrame({
    "Metric": [
        "Peak grid import (MW)",
        "Renewable curtailment (MWh)",
        "Minimum battery SOC (%)",
        "Unmet load (MWh)",
        "Maximum upward ramp (MW)",
    ],
    "Base case": [
        bm["max_grid_import"], bm["curtailment"], bm["min_soc"],
        bm["unmet"], bm["max_ramp"],
    ],
    "Scenario": [
        sm["max_grid_import"], sm["curtailment"], sm["min_soc"],
        sm["unmet"], sm["max_ramp"],
    ],
})
comparison["Change"] = comparison["Scenario"] - comparison["Base case"]
st.dataframe(comparison.round(1), use_container_width=True, hide_index=True)

# 5. AI AGENT
st.header("5. AI Grid Operations Agent")

questions = [
    "Why is the grid most stressed during the evening?",
    "When does the battery become constrained and why?",
    "What causes renewable curtailment?",
    "What is the most critical operating hour?",
    "Compare the current what-if scenario with the base case.",
]

st.write("Suggested questions:")
button_cols = st.columns(len(questions))
for i, q in enumerate(questions):
    if button_cols[i].button(q, key=f"q{i}", use_container_width=True):
        st.session_state["question"] = q

question = st.text_input(
    "AI Operator Chat",
    value=st.session_state.get("question", ""),
    placeholder="Ask an operational question about the grid..."
)

include_scenario = st.checkbox("Give the AI access to the current what-if scenario", value=True)

if st.button("Investigate with AI Agent", type="primary"):
    if not question.strip():
        st.warning("Enter a question or choose a suggested question.")
    else:
        try:
            with st.status("Agent investigating...", expanded=True) as status:
                answer, used_tools = ask_agent(
                    question,
                    base,
                    scenario if include_scenario else None,
                )

                st.write("### 🔧 Agent tool activity")
                if used_tools:
                    for name, output in used_tools:
                        st.write(f"**✓ {name}**")
                        st.json(output)
                else:
                    st.write("The model answered without requesting a tool.")

                status.update(label="Investigation complete", state="complete")

            st.subheader("AI Diagnosis and Recommendation")
            st.write(answer)

        except Exception as e:
            st.error(f"Groq request failed: {e}")

st.caption("Educational prototype only — not for real-time or safety-critical grid operation.")
