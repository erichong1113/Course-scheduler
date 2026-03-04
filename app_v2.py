import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time

# =========================
# Page config + light styling
# =========================
st.set_page_config(page_title="Course Fit + Preference Planner (Demo)", layout="wide")
st.markdown("""
<style>
.small {font-size: 0.9rem; opacity: 0.85;}
.card {padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.2); background: rgba(255,255,255,0.02);}
.kpi {font-size: 1.8rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "northwestern_tgs_scheduling_dataset_3.xlsx"  # put this next to app_v2.py

# =========================
# Helpers
# =========================
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
    """Convert days like 'MW', 'TTh', 'MWF', 'F' into a set like {'Mon','Wed'}."""
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
    """
    Return prerequisites as groups:
      - all_required: [[A],[B],[C]] meaning A and B and C
      - any_one_required: [[A,B,C]] meaning A or B or C
    """
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
    sched["section"] = sched["section"].astype(str).str.strip()
    sched["title"] = sched["title"].astype(str).str.strip()
    sched["days"] = sched["days"].astype(str).str.strip()

    # best-effort time parsing (keep original strings too)
    sched["_start"] = sched["start_time"].apply(parse_time_str)
    sched["_end"] = sched["end_time"].apply(parse_time_str)
    return deg, sched, students, ctec

def get_ctec_score(ctec: pd.DataFrame, course_id: str, instructor: str | None) -> float | None:
    if ctec.empty:
        return None
    df = ctec[ctec["course_id"] == course_id].copy()
    if instructor and isinstance(instructor, str) and instructor.strip():
        df2 = df[df["instructor"].astype(str).str.strip() == instructor.strip()]
        if not df2.empty:
            return float(df2["overall_teaching"].mean())
    if df.empty:
        return None
    return float(df["overall_teaching"].mean())

def build_current_blocks(sched: pd.DataFrame, students: pd.DataFrame, student_name: str, quarter: str):
    """Return schedule blocks for the student's in-progress courses in a quarter."""
    s_rows = students[students["student_name"] == student_name]
    inprog = s_rows[(s_rows["course_status"] == "in_progress") & (s_rows["quarter_taken"] == quarter)]
    blocks = []
    for cid in inprog["course_id"].tolist():
        rows = sched[(sched["course_id"] == cid) & (sched["quarter"] == quarter)]
        if rows.empty:
            continue
        # assume first section as the enrolled one (demo)
        r = rows.iloc[0]
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
    """Sum of (latest_end - earliest_start) per day across blocks (on-campus only). Minutes."""
    day_to_times = {}
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        if b["start"] is None or b["end"] is None:
            continue
        for d in b["days"]:
            day_to_times.setdefault(d, []).append((b["start"], b["end"]))
    total_minutes = 0
    for d, intervals in day_to_times.items():
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
        if name:
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
    completed = set(s_rows.loc[s_rows["course_status"] == "completed", "course_id"].tolist())

    # use any row of this course as "catalog"
    c_rows = sched[sched["course_id"] == course_id]
    if c_rows.empty:
        return False, ["Course not found"], [], completed

    c0 = c_rows.iloc[0]
    title = str(c0.get("title", "")).strip()

    reasons_bad = []
    reasons_ok = []

    if course_id in completed:
        reasons_bad.append(f"Already completed: {course_id} — {title}")
        return False, reasons_bad, reasons_ok, completed

    prereq_groups = parse_prereqs(c0.get("prerequisites", None), c0.get("prereq_type", None))
    satisfied, missing = prereq_missing(prereq_groups, completed)
    if not satisfied:
        reasons_bad.append("Missing prerequisites: " + ", ".join(missing))
    else:
        reasons_ok.append("Prerequisites: satisfied" if prereq_groups else "Prerequisites: none")

    # coreqs as soft note
    coreq = str(c0.get("corequisites", "")).strip()
    if coreq and coreq.lower() != "nan":
        reasons_ok.append(f"Corequisites noted: {coreq} (not enforced in demo)")

    eligible = len(reasons_bad) == 0
    return eligible, reasons_bad, reasons_ok, completed

# =========================
# UI
# =========================
st.title("📚 Course Fit + Preference Planner (Prototype)")
st.caption("Demo: pick a student + course + quarter, then we check eligibility and recommend the best section based on your priorities.")

try:
    deg, sched, students, ctec = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Can't find `{DATA_PATH}`. Put `app_v2.py` in the same folder as the Excel file, or update DATA_PATH in the code.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Inputs")

    student_names = sorted(students["student_name"].dropna().unique().tolist())
    quarters = sorted(sched["quarter"].dropna().unique().tolist())

    # course list from scheduling table
    courses = sched[["course_id", "title", "department", "level"]].drop_duplicates().copy()
    courses["label"] = courses["course_id"] + " — " + courses["title"].astype(str)
    courses = courses.sort_values(["department", "level", "course_id"])

    student_name = st.selectbox("Student", student_names)
    course_label = st.selectbox("Course", courses["label"].tolist())
    quarter = st.selectbox("Quarter (for section recommendation)", quarters)

    st.divider()
    st.subheader("Preferences (weights)")
    st.caption("0 = don't care, 5 = care a lot")

    w_trips = st.slider("Minimize days on campus (fewer trips)", 0, 5, 3)
    w_time = st.slider("Minimize time on campus (shorter day spans)", 0, 5, 2)
    w_walk = st.slider("Minimize walking (proxy: fewer buildings)", 0, 5, 1)
    w_ctec = st.slider("Maximize CTECs (overall teaching)", 0, 5, 3)

    enforce_no_conflict = st.toggle("Hard block: reject time-conflicting sections", value=True)

    run = st.button("Analyze + Recommend", type="primary", use_container_width=True)

# Main
course_id = course_label.split(" — ")[0].strip()

if not run:
    st.info("Pick inputs on the left, then click **Analyze + Recommend**.")
    st.stop()

eligible, reasons_bad, reasons_ok, completed = check_eligibility(student_name, course_id, sched, students)

top = st.container()
with top:
    colA, colB, colC = st.columns([1.3, 1, 1])
    with colA:
        st.markdown(f"<div class='card'><div class='small'>Student</div><div class='kpi'>{student_name}</div></div>", unsafe_allow_html=True)
    with colB:
        st.markdown(f"<div class='card'><div class='small'>Course</div><div class='kpi'>{course_id}</div></div>", unsafe_allow_html=True)
    with colC:
        st.markdown(f"<div class='card'><div class='small'>Quarter</div><div class='kpi'>{quarter}</div></div>", unsafe_allow_html=True)

st.divider()

# Eligibility result
if eligible:
    st.success("✅ Eligible (based on demo rules)")
else:
    st.error("❌ Not eligible (based on demo rules)")

c1, c2 = st.columns(2)
with c1:
    st.subheader("What’s blocking it")
    if reasons_bad:
        for r in reasons_bad:
            st.write("• " + r)
    else:
        st.write("• Nothing blocking it (demo rules).")
with c2:
    st.subheader("Good signals / notes")
    if reasons_ok:
        for r in reasons_ok:
            st.write("• " + r)
    else:
        st.write("• —")

# If not eligible, still show sections, but don't "recommend"
st.divider()

# Sections in quarter
sec_rows = sched[(sched["course_id"] == course_id) & (sched["quarter"] == quarter)].copy()
if sec_rows.empty:
    st.warning(f"No sections found for {course_id} in {quarter}.")
    st.stop()

current_blocks = build_current_blocks(sched, students, student_name, quarter)

# Compute features per section
rows = []
for _, r in sec_rows.iterrows():
    conflict_with = section_conflicts(r, current_blocks)
    # Build combined blocks to compute "after adding this section"
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
        "location": str(r.get("location", "")).strip(),
        "building": str(r.get("building_name", "")).strip(),
        "modality": str(r.get("modality", "")).strip(),
        "instructor": instructor,
        "conflicts": (len(conflict_with) > 0),
        "conflict_with": ", ".join(conflict_with) if conflict_with else "",
        "days_on_campus": trips,
        "time_on_campus_min": time_on,
        "unique_buildings": walk,
        "ctec_overall": ctec_score,
    })

df = pd.DataFrame(rows)

# Optionally hard-block conflicting sections
if enforce_no_conflict:
    df["blocked"] = df["conflicts"].astype(bool)
else:
    df["blocked"] = False

# Scoring (normalize across available sections)
w = {
    "days_on_campus": w_trips,
    "time_on_campus_min": w_time,
    "unique_buildings": w_walk,
    "ctec_overall": w_ctec,
}

score_df = df.copy()

# Fill CTEC missing with mean (neutral-ish)
if "ctec_overall" in score_df.columns:
    if score_df["ctec_overall"].notna().any():
        score_df["ctec_overall"] = score_df["ctec_overall"].fillna(score_df["ctec_overall"].mean())
    else:
        score_df["ctec_overall"] = np.nan

# penalties: lower is better
score_df["p_trips"] = minmax_norm(score_df["days_on_campus"])
score_df["p_time"] = minmax_norm(score_df["time_on_campus_min"])
score_df["p_walk"] = minmax_norm(score_df["unique_buildings"])

# for CTEC we want maximize => penalty is inverse
if score_df["ctec_overall"].notna().any():
    score_df["p_ctec"] = 1.0 - minmax_norm(score_df["ctec_overall"])
else:
    score_df["p_ctec"] = 0.0  # if no data, don't penalize

# weighted penalty
den = max(1e-9, float(w_trips + w_time + w_walk + w_ctec))
score_df["weighted_penalty"] = (
    w_trips * score_df["p_trips"] +
    w_time  * score_df["p_time"] +
    w_walk  * score_df["p_walk"] +
    w_ctec  * score_df["p_ctec"]
) / den

# hard-block: push score down
score_df["suitability"] = (1.0 - score_df["weighted_penalty"]) * 100.0
score_df.loc[score_df["blocked"], "suitability"] = -1.0

# pick best
best = score_df.sort_values(["suitability"], ascending=False).iloc[0]

# Summary KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Current in-progress courses (this quarter)", len(current_blocks))
k2.metric("Best section", str(best["section"]) if best["suitability"] >= 0 else "None")
k3.metric("Suitability score", f'{best["suitability"]:.0f}/100' if best["suitability"] >= 0 else "Blocked")
k4.metric("Hard conflict blocking", "On" if enforce_no_conflict else "Off")

st.subheader("Recommended section (based on your weights)")
if best["suitability"] < 0:
    st.warning("Every available section conflicts with the student's current in-progress schedule (with hard blocking on).")
else:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"""
<div class="card">
  <div class="kpi">Section {best['section']}</div>
  <div class="small">{best['days']} • {best['start_time']}–{best['end_time']} • {best['modality']}</div>
  <div class="small">{best['location']} {('(' + best['building'] + ')') if best['building'] else ''}</div>
  <div class="small">Instructor: {best['instructor'] or '—'} • CTEC overall: {('%.2f' % best['ctec_overall']) if pd.notna(best['ctec_overall']) else '—'}</div>
</div>
""", unsafe_allow_html=True)
        if best["conflicts"]:
            st.error("Time conflict with: " + (best["conflict_with"] or "—"))
        else:
            st.success("No time conflict found (demo check).")
    with right:
        st.markdown("**Score breakdown (lower penalty is better):**")
        st.write(f"• Days on campus penalty: {best['p_trips']:.2f} (weight {w_trips})")
        st.write(f"• Time on campus penalty: {best['p_time']:.2f} (weight {w_time})")
        st.write(f"• Walking penalty (proxy): {best['p_walk']:.2f} (weight {w_walk})")
        st.write(f"• CTEC penalty (inverse): {best['p_ctec']:.2f} (weight {w_ctec})")

st.divider()

tab1, tab2, tab3 = st.tabs(["All sections", "Student schedule (this quarter)", "Student history"])

with tab1:
    show = score_df.copy()
    show = show.sort_values(["blocked", "suitability"], ascending=[True, False])
    cols = ["section","days","start_time","end_time","modality","instructor","ctec_overall","conflicts","conflict_with",
            "days_on_campus","time_on_campus_min","unique_buildings","suitability"]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)

with tab2:
    if current_blocks:
        cur_df = pd.DataFrame([{
            "course_id": b["course_id"],
            "title": b["title"],
            "days": "".join(sorted(list(b["days"]))),
            "time_block": b["time_block"],
            "modality": b["modality"],
            "building": b["building_name"],
        } for b in current_blocks])
        st.dataframe(cur_df, use_container_width=True, hide_index=True)
    else:
        st.write("No in-progress courses found for this student in this quarter.")

with tab3:
    s_rows = students[students["student_name"] == student_name].copy()
    s_rows = s_rows.sort_values(["quarter_taken", "course_status", "course_id"])
    st.dataframe(
        s_rows[["course_id","quarter_taken","course_status","grade","cumulative_gpa","notes"]],
        use_container_width=True,
        hide_index=True
    )

with st.expander("Assumptions (for the demo)"):
    st.write("• Uses Student_Profiles.course_status == 'completed' as prerequisites evidence.")
    st.write("• Uses Student_Profiles.course_status == 'in_progress' within the selected quarter as the student's current schedule.")
    st.write("• If a student is enrolled in a course, we assume they are in the first listed section for that course (demo simplification).")
    st.write("• 'Walking' is approximated by the number of distinct buildings (since we don't geocode addresses here).")
    st.write("• CTEC score uses CTEC_Evaluations.overall_teaching (instructor-specific if available; else course average).")
