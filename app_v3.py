import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time

# =========================================================
# Pretty UI: CSS + layout
# =========================================================
st.set_page_config(page_title="CourseFit Studio", layout="wide")

st.markdown("""
<style>
:root{
  --bg0: rgba(255,255,255,0.03);
  --bd0: rgba(255,255,255,0.10);
  --bd1: rgba(255,255,255,0.18);
  --txt0: rgba(255,255,255,0.92);
  --txt1: rgba(255,255,255,0.70);
  --ok: rgba(52, 211, 153, 0.25);
  --bad: rgba(248, 113, 113, 0.20);
  --warn: rgba(251, 191, 36, 0.20);
}
.block-container {padding-top: 1.2rem;}
/* Hero */
.hero {
  border: 1px solid var(--bd0);
  border-radius: 22px;
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(99,102,241,0.20), rgba(16,185,129,0.12), rgba(236,72,153,0.12));
}
.hero h1 {margin: 0; font-size: 1.65rem;}
.hero p {margin: 6px 0 0 0; color: var(--txt1);}

/* Cards */
.card{
  border: 1px solid var(--bd0);
  background: var(--bg0);
  border-radius: 18px;
  padding: 14px 14px;
}
.card .label{font-size: 0.85rem; color: var(--txt1);}
.card .value{font-size: 1.15rem; font-weight: 700; color: var(--txt0); margin-top: 2px;}
.pill{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--bd0);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
  color: var(--txt1);
  margin-right: 6px;
  margin-top: 6px;
}
.pill.ok{background: var(--ok); border-color: rgba(52,211,153,0.35); color: rgba(255,255,255,0.90);}
.pill.bad{background: var(--bad); border-color: rgba(248,113,113,0.35); color: rgba(255,255,255,0.90);}
.pill.warn{background: var(--warn); border-color: rgba(251,191,36,0.35); color: rgba(255,255,255,0.90);}

/* Section titles */
.h2 {font-size: 1.1rem; font-weight: 750; margin: 4px 0 8px 0;}
.small {color: var(--txt1); font-size: 0.90rem;}
hr {border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 16px 0;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "northwestern_tgs_scheduling_dataset_3.xlsx"  # keep app_v3.py next to this file

# =========================================================
# Helpers
# =========================================================
DAY_MAP = {
    "M": {"Mon"},
    "T": {"Tue"},
    "W": {"Wed"},
    "Th": {"Thu"},
    "F": {"Fri"},
    "Sa": {"Sat"},
    "Su": {"Sun"},
}

def parse_days(days_str: str) -> set[str]:
    if not isinstance(days_str, str) or not days_str.strip():
        return set()
    s = days_str.strip()
    out = set()
    i = 0
    while i < len(s):
        if s[i:i+2] == "Th":
            out |= DAY_MAP["Th"]
            i += 2
        else:
            ch = s[i]
            if ch in DAY_MAP:
                out |= DAY_MAP[ch]
            i += 1
    return out

def parse_time_str(t) -> time | None:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if isinstance(t, time):
        return t
    if not isinstance(t, str) or not t.strip():
        return None
    try:
        return datetime.strptime(t.strip(), "%H:%M").time()
    except Exception:
        return None

def time_overlap(a_start: time, a_end: time, b_start: time, b_end: time) -> bool:
    if None in (a_start, a_end, b_start, b_end):
        return False
    return (a_start < b_end) and (b_start < a_end)

def is_on_campus(modality: str | None) -> bool:
    if not isinstance(modality, str):
        return True
    m = modality.strip().lower()
    return not ("online" in m or "remote" in m)

def parse_prereqs(prereq_str: str, prereq_type: str | None) -> list[list[str]]:
    if not isinstance(prereq_str, str) or not prereq_str.strip():
        return []
    prereq_str = prereq_str.strip()

    if prereq_type == "any_one_required":
        parts = [p.strip() for p in prereq_str.split(" OR ")]
        parts = [p for p in parts if p]
        return [parts] if parts else []

    parts = [p.strip() for p in prereq_str.split(";")]
    parts = [p for p in parts if p]
    return [[p] for p in parts] if parts else []

def prereq_missing(prereq_groups: list[list[str]], completed: set[str]) -> tuple[bool, list[str]]:
    missing = []
    for group in prereq_groups:
        if len(group) == 1:
            c = group[0]
            if c not in completed:
                missing.append(c)
        else:
            if not any(c in completed for c in group):
                missing.append(" OR ".join(group))
    return (len(missing) == 0), missing

def minmax_norm(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = float(s.min()), float(s.max())
    if np.isclose(mx, mn):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

@st.cache_data(show_spinner=False)
def load_data(path: str):
    deg = pd.read_excel(path, sheet_name="Degree_Requirements")
    sched = pd.read_excel(path, sheet_name="Course_Scheduling")
    students = pd.read_excel(path, sheet_name="Student_Profiles")
    ctec = pd.read_excel(path, sheet_name="CTEC_Evaluations")

    # normalize
    for df in (sched, students, ctec):
        if "course_id" in df.columns:
            df["course_id"] = df["course_id"].astype(str).str.strip()

    students["student_name"] = students["student_name"].astype(str).str.strip()
    sched["quarter"] = sched["quarter"].astype(str).str.strip()
    students["quarter_taken"] = students["quarter_taken"].astype(str).str.strip()

    for col in ["section","title","department","days","start_time","end_time","time_block","modality","location","building_name","building_address","instructor"]:
        if col in sched.columns:
            sched[col] = sched[col].astype(str).str.strip()

    sched["_start"] = sched["start_time"].apply(parse_time_str)
    sched["_end"] = sched["end_time"].apply(parse_time_str)
    return deg, sched, students, ctec

def get_ctec_score(ctec: pd.DataFrame, course_id: str, instructor: str | None) -> float | None:
    if ctec.empty:
        return None
    df = ctec[ctec["course_id"] == course_id].copy()
    if instructor and isinstance(instructor, str) and instructor.strip():
        df2 = df[df["instructor"].astype(str).str.strip() == instructor.strip()]
        if not df2.empty and "overall_teaching" in df2.columns:
            return float(df2["overall_teaching"].mean())
    if df.empty or "overall_teaching" not in df.columns:
        return None
    return float(df["overall_teaching"].mean())

def build_current_blocks(sched: pd.DataFrame, students: pd.DataFrame, student_name: str, quarter: str):
    s_rows = students[students["student_name"] == student_name]
    inprog = s_rows[(s_rows["course_status"] == "in_progress") & (s_rows["quarter_taken"] == quarter)]
    blocks = []
    for cid in inprog["course_id"].tolist():
        rows = sched[(sched["course_id"] == cid) & (sched["quarter"] == quarter)]
        if rows.empty:
            continue
        r = rows.iloc[0]  # demo simplification
        blocks.append({
            "course_id": cid,
            "title": str(r.get("title", "")).strip(),
            "days": parse_days(r.get("days", "")),
            "start": r.get("_start"),
            "end": r.get("_end"),
            "time_block": str(r.get("time_block", "")).strip(),
            "modality": str(r.get("modality", "")).strip(),
            "building_name": str(r.get("building_name", "")).strip(),
        })
    return blocks

def compute_day_spans(blocks):
    day_to_times = {}
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        if b["start"] is None or b["end"] is None:
            continue
        for d in b["days"]:
            day_to_times.setdefault(d, []).append((b["start"], b["end"]))
    total_minutes = 0.0
    for _, intervals in day_to_times.items():
        starts = [t[0] for t in intervals]
        ends = [t[1] for t in intervals]
        earliest = min(starts)
        latest = max(ends)
        total_minutes += (datetime.combine(datetime.today(), latest) - datetime.combine(datetime.today(), earliest)).seconds / 60
    return float(total_minutes)

def compute_unique_days_on_campus(blocks):
    days = set()
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        days |= set(b["days"])
    return float(len(days))

def compute_unique_buildings(blocks):
    bld = set()
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        name = str(b.get("building_name", "")).strip()
        if name and name.lower() != "nan":
            bld.add(name)
    return float(len(bld))

def section_conflicts(section_row, current_blocks):
    days = parse_days(section_row.get("days", ""))
    start = section_row.get("_start")
    end = section_row.get("_end")
    conflict_with = []
    for b in current_blocks:
        if days and b["days"] and (days & b["days"]):
            if time_overlap(start, end, b["start"], b["end"]):
                conflict_with.append(f'{b["course_id"]} ({b["time_block"]})')
    return conflict_with

def check_eligibility(student_name: str, course_id: str, sched: pd.DataFrame, students: pd.DataFrame):
    s_rows = students[students["student_name"] == student_name]
    if s_rows.empty:
        return False, ["Student not found"], [], set()

    completed = set(s_rows.loc[s_rows["course_status"] == "completed", "course_id"].tolist())
    c_rows = sched[sched["course_id"] == course_id]
    if c_rows.empty:
        return False, ["Course not found"], [], completed

    c0 = c_rows.iloc[0]
    title = str(c0.get("title", "")).strip()

    reasons_bad, reasons_ok = [], []

    if course_id in completed:
        reasons_bad.append(f"Already completed: {course_id} — {title}")
        return False, reasons_bad, reasons_ok, completed

    prereq_groups = parse_prereqs(c0.get("prerequisites", None), c0.get("prereq_type", None))
    satisfied, missing = prereq_missing(prereq_groups, completed)
    if not satisfied:
        reasons_bad.append("Missing prerequisites: " + ", ".join(missing))
    else:
        reasons_ok.append("Prerequisites: satisfied" if prereq_groups else "Prerequisites: none")

    coreq = str(c0.get("corequisites", "")).strip()
    if coreq and coreq.lower() != "nan":
        reasons_ok.append(f"Corequisites noted: {coreq} (not enforced in demo)")

    return (len(reasons_bad) == 0), reasons_bad, reasons_ok, completed

# =========================================================
# App
# =========================================================
st.markdown("""
<div class="hero">
  <h1>CourseFit Studio</h1>
  <p>Prototype UI: eligibility check + section recommendation based on priorities (fewer trips, less time on campus, less walking, higher CTECs).</p>
</div>
""", unsafe_allow_html=True)

try:
    deg, sched, students, ctec = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Can't find `{DATA_PATH}`. Put `app_v3.py` in the same folder as the Excel file, or update DATA_PATH in the code.")
    st.stop()

# ---- Top controls row
student_names = sorted(students["student_name"].dropna().unique().tolist())
quarters = sorted(sched["quarter"].dropna().unique().tolist())

courses = sched[["course_id", "title", "department", "level"]].drop_duplicates().copy()
courses["label"] = courses["course_id"] + " — " + courses["title"].astype(str)
courses = courses.sort_values(["department", "level", "course_id"])

left, mid, right = st.columns([1.1, 1.4, 1.1])
with left:
    student_name = st.selectbox("Student", student_names, index=0)
with mid:
    course_label = st.selectbox("Course", courses["label"].tolist(), index=0)
with right:
    quarter = st.selectbox("Quarter", quarters, index=min(0, len(quarters)-1))

course_id = course_label.split(" — ")[0].strip()

# ---- Preferences as a "panel"
with st.expander("🎛️ Preferences & priorities", expanded=True):
    a, b, c, d, e = st.columns([1,1,1,1,1.1])
    with a:
        w_trips = st.slider("Fewer trips", 0, 5, 3, help="Minimize unique days on campus")
    with b:
        w_time = st.slider("Less time on campus", 0, 5, 2, help="Minimize day spans (earliest to latest class per day)")
    with c:
        w_walk = st.slider("Less walking", 0, 5, 1, help="Proxy: minimize number of distinct buildings")
    with d:
        w_ctec = st.slider("Higher CTECs", 0, 5, 3, help="Maximize overall teaching score")
    with e:
        enforce_no_conflict = st.toggle("Hard block conflicts", value=True, help="If on, time-conflicting sections are rejected")

# ---- Action button
run = st.button("✨ Run analysis", type="primary", use_container_width=True)

if not run:
    st.caption("Tip: tweak sliders and rerun to see the recommended section change.")
    st.stop()

# =========================================================
# Step 1: Eligibility
# =========================================================
eligible, reasons_bad, reasons_ok, completed = check_eligibility(student_name, course_id, sched, students)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='card'><div class='label'>Student</div><div class='value'>{student_name}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'><div class='label'>Course</div><div class='value'>{course_id}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'><div class='label'>Quarter</div><div class='value'>{quarter}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='card'><div class='label'>Eligibility</div><div class='value'>{'Eligible ✅' if eligible else 'Not eligible ❌'}</div></div>", unsafe_allow_html=True)

# Reasons as pills
pill_line = ""
if eligible:
    pill_line += "<span class='pill ok'>Eligible</span>"
else:
    pill_line += "<span class='pill bad'>Not eligible</span>"
pill_line += "<span class='pill'>Demo rules: prereqs + already-taken</span>"
if enforce_no_conflict:
    pill_line += "<span class='pill warn'>Hard conflict blocking ON</span>"
else:
    pill_line += "<span class='pill'>Hard conflict blocking OFF</span>"
st.markdown(pill_line, unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='h2'>Why / what’s blocking</div>", unsafe_allow_html=True)
    if reasons_bad:
        for r in reasons_bad:
            st.markdown(f"- {r}")
    else:
        st.markdown("- Nothing blocking it (demo rules).")
with c2:
    st.markdown("<div class='h2'>Good signals / notes</div>", unsafe_allow_html=True)
    if reasons_ok:
        for r in reasons_ok:
            st.markdown(f"- {r}")
    else:
        st.markdown("- —")

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================================================
# Step 2: Section recommendation (in the chosen quarter)
# =========================================================
sec_rows = sched[(sched["course_id"] == course_id) & (sched["quarter"] == quarter)].copy()
if sec_rows.empty:
    st.warning(f"No sections found for {course_id} in {quarter}.")
    st.stop()

current_blocks = build_current_blocks(sched, students, student_name, quarter)

rows = []
for _, r in sec_rows.iterrows():
    conflict_with = section_conflicts(r, current_blocks)
    add_block = {
        "course_id": course_id,
        "title": str(r.get("title", "")).strip(),
        "days": parse_days(r.get("days", "")),
        "start": r.get("_start"),
        "end": r.get("_end"),
        "time_block": str(r.get("time_block", "")).strip(),
        "modality": str(r.get("modality", "")).strip(),
        "building_name": str(r.get("building_name", "")).strip(),
    }
    combined = current_blocks + [add_block]

    trips = compute_unique_days_on_campus(combined)
    time_on = compute_day_spans(combined)
    walk = compute_unique_buildings(combined)

    instructor = str(r.get("instructor", "")).strip()
    ctec_score = get_ctec_score(ctec, course_id, instructor)

    rows.append({
        "section": str(r.get("section", "")).strip(),
        "days": str(r.get("days", "")).strip(),
        "start_time": str(r.get("start_time", "")).strip(),
        "end_time": str(r.get("end_time", "")).strip(),
        "time_block": str(r.get("time_block", "")).strip(),
        "modality": str(r.get("modality", "")).strip(),
        "instructor": instructor,
        "location": str(r.get("location", "")).strip(),
        "building": str(r.get("building_name", "")).strip(),
        "address": str(r.get("building_address", "")).strip(),
        "conflicts": (len(conflict_with) > 0),
        "conflict_with": ", ".join(conflict_with) if conflict_with else "",
        "days_on_campus": trips,
        "time_on_campus_min": time_on,
        "unique_buildings": walk,
        "ctec_overall": ctec_score,
    })

df = pd.DataFrame(rows)

if enforce_no_conflict:
    df["blocked"] = df["conflicts"].astype(bool)
else:
    df["blocked"] = False

# Fill CTEC missing with mean (neutral-ish)
score_df = df.copy()
if "ctec_overall" in score_df.columns:
    if score_df["ctec_overall"].notna().any():
        score_df["ctec_overall"] = score_df["ctec_overall"].fillna(score_df["ctec_overall"].mean())
    else:
        score_df["ctec_overall"] = np.nan

# normalize penalties
score_df["p_trips"] = minmax_norm(score_df["days_on_campus"])
score_df["p_time"] = minmax_norm(score_df["time_on_campus_min"])
score_df["p_walk"] = minmax_norm(score_df["unique_buildings"])
score_df["p_ctec"] = (1.0 - minmax_norm(score_df["ctec_overall"])) if score_df["ctec_overall"].notna().any() else 0.0

den = max(1e-9, float(w_trips + w_time + w_walk + w_ctec))
score_df["weighted_penalty"] = (
    w_trips * score_df["p_trips"] +
    w_time  * score_df["p_time"] +
    w_walk  * score_df["p_walk"] +
    w_ctec  * score_df["p_ctec"]
) / den

score_df["suitability"] = (1.0 - score_df["weighted_penalty"]) * 100.0
score_df.loc[score_df["blocked"], "suitability"] = -1.0

score_df = score_df.sort_values(["suitability"], ascending=False)
best = score_df.iloc[0]

# Headline recommendation card
st.markdown("<div class='h2'>Recommended section</div>", unsafe_allow_html=True)

if best["suitability"] < 0:
    st.warning("Every available section is blocked by time conflicts (with hard blocking ON).")
else:
    L, R = st.columns([1.2, 1])
    with L:
        st.markdown(f"""
<div class="card">
  <div class="label">Best match for your priorities</div>
  <div class="value">Section {best['section']} • {best['days']} • {best['start_time']}–{best['end_time']}</div>
  <div class="small">{best['modality']} • {best['location']} {('(' + best['building'] + ')') if best['building'] else ''}</div>
  <div class="small">Instructor: {best['instructor'] or '—'} • CTEC: {('%.2f' % best['ctec_overall']) if pd.notna(best['ctec_overall']) else '—'}</div>
  <div style="margin-top:10px;">
    <span class="pill ok">Suitability {best['suitability']:.0f}/100</span>
    {("<span class='pill bad'>Has conflict</span>" if best['conflicts'] else "<span class='pill ok'>No conflict</span>")}
  </div>
</div>
""", unsafe_allow_html=True)
        if best["conflicts"]:
            st.error("Time conflict with: " + (best["conflict_with"] or "—"))
    with R:
        st.markdown("<div class='card'>"
                    "<div class='label'>Score breakdown</div>"
                    f"<div class='small'>Lower penalty is better • weights: trips {w_trips}, time {w_time}, walk {w_walk}, CTEC {w_ctec}</div>"
                    "<hr/>"
                    f"<div class='small'>Trips penalty: <b>{best['p_trips']:.2f}</b></div>"
                    f"<div class='small'>Time penalty: <b>{best['p_time']:.2f}</b></div>"
                    f"<div class='small'>Walking penalty: <b>{best['p_walk']:.2f}</b></div>"
                    f"<div class='small'>CTEC penalty (inverse): <b>{best['p_ctec']:.2f}</b></div>"
                    "</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Tabs: sections / schedule / history
tab1, tab2, tab3 = st.tabs(["🧾 Sections & scores", "🗓️ Student schedule", "📜 Student history"])

with tab1:
    view = score_df.copy()
    view = view.sort_values(["blocked", "suitability"], ascending=[True, False])
    cols = ["section","days","start_time","end_time","time_block","modality","instructor","ctec_overall",
            "conflicts","conflict_with","days_on_campus","time_on_campus_min","unique_buildings","suitability"]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)

with tab2:
    if current_blocks:
        cur_df = pd.DataFrame([{
            "course_id": b["course_id"],
            "title": b["title"],
            "days": " ".join(sorted(list(b["days"]))),
            "time_block": b["time_block"],
            "modality": b["modality"],
            "building": b["building_name"],
        } for b in current_blocks])
        st.dataframe(cur_df, use_container_width=True, hide_index=True)
        st.caption("Demo assumption: if a student is taking a course, we assume they are in the first listed section for that course.")
    else:
        st.info("No in-progress courses found for this student in this quarter.")

with tab3:
    s_rows = students[students["student_name"] == student_name].copy()
    s_rows = s_rows.sort_values(["quarter_taken", "course_status", "course_id"])
    show_cols = [c for c in ["course_id","quarter_taken","course_status","grade","cumulative_gpa","notes"] if c in s_rows.columns]
    st.dataframe(s_rows[show_cols], use_container_width=True, hide_index=True)

with st.expander("🔎 Assumptions (demo)"):
    st.write("• Prereqs evidence: Student_Profiles.course_status == 'completed'")
    st.write("• Current schedule: Student_Profiles.course_status == 'in_progress' in selected quarter")
    st.write("• Walking: proxy = number of distinct buildings (no geocoding)")
    st.write("• CTEC: CTEC_Evaluations.overall_teaching (instructor-specific if available; else course average)")
