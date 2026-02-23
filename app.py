import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, time

# =========================
# Config
# =========================
st.set_page_config(page_title="Course Fit Checker (Demo)", layout="wide")

DATA_PATH = "northwestern_tgs_scheduling_dataset_1.xlsx"  # put app.py next to the xlsx

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

    # Handle common compact patterns
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
    """Parse 'HH:MM' into datetime.time."""
    if not isinstance(t, str) or not t.strip():
        return None
    try:
        return datetime.strptime(t.strip(), "%H:%M").time()
    except Exception:
        return None

def time_overlap(a_start: time, a_end: time, b_start: time, b_end: time) -> bool:
    """True if time intervals overlap (strict)."""
    if None in (a_start, a_end, b_start, b_end):
        return False
    return (a_start < b_end) and (b_start < a_end)

def normalize_course_id(x: str) -> str:
    return str(x).strip()

def parse_prereqs(prereq_str: str, prereq_type: str | None) -> list[list[str]]:
    """
    Return prerequisites as groups (CNF-ish):
      - all_required: [[A],[B],[C]] meaning A and B and C
      - any_one_required: [[A,B,C]] meaning A or B or C
      - blank/NaN: []
    """
    if not isinstance(prereq_str, str) or not prereq_str.strip():
        return []
    prereq_str = prereq_str.strip()

    if prereq_type == "any_one_required":
        parts = [p.strip() for p in prereq_str.split(" OR ")]
        parts = [p for p in parts if p]
        return [parts] if parts else []

    # default: all_required (also covers weird strings with ';')
    parts = [p.strip() for p in prereq_str.split(";")]
    parts = [p for p in parts if p]
    return [[p] for p in parts] if parts else []

def prereq_missing(prereq_groups: list[list[str]], completed: set[str]) -> tuple[bool, list[str]]:
    """
    prereq_groups:
      [[A],[B]] -> need all groups satisfied
      [[A,B]]   -> need at least one in group
    Returns: (satisfied, missing_list)
    """
    missing = []
    for group in prereq_groups:
        if len(group) == 1:
            course = group[0]
            if course not in completed:
                missing.append(course)
        else:
            # OR group
            if not any(c in completed for c in group):
                missing.append(" OR ".join(group))
    return (len(missing) == 0), missing

@st.cache_data(show_spinner=False)
def load_data(path: str):
    xl = pd.ExcelFile(path)
    degree = pd.read_excel(path, sheet_name="Degree_Requirements")
    catalog = pd.read_excel(path, sheet_name="Course_Catalog")
    sched = pd.read_excel(path, sheet_name="Course_Scheduling")
    students = pd.read_excel(path, sheet_name="Student_Profiles")
    # normalize ids
    catalog["course_id"] = catalog["course_id"].astype(str).str.strip()
    sched["course_id"] = sched["course_id"].astype(str).str.strip()
    students["course_id"] = students["course_id"].astype(str).str.strip()
    students["student_name"] = students["student_name"].astype(str).str.strip()
    sched["quarter"] = sched["quarter"].astype(str).str.strip()
    students["quarter_taken"] = students["quarter_taken"].astype(str).str.strip()
    return degree, catalog, sched, students

def check_fit(student_name: str, course_id: str, quarter: str | None,
              catalog: pd.DataFrame, sched: pd.DataFrame, students: pd.DataFrame):
    """Core logic: prerequisites, already-taken, offered, time conflicts."""
    reasons_bad = []
    reasons_ok = []

    # Student records
    s_rows = students[students["student_name"] == student_name]
    if s_rows.empty:
        return {"eligible": False, "status": "Not found", "reasons_bad": ["Student not found"], "reasons_ok": [], "section_suggestions": []}

    completed = set(s_rows.loc[s_rows["course_status"] == "completed", "course_id"].tolist())
    inprog = s_rows.loc[s_rows["course_status"] == "in_progress", ["course_id", "quarter_taken"]].copy()

    # Course info
    c_row = catalog[catalog["course_id"] == course_id]
    if c_row.empty:
        return {"eligible": False, "status": "Not found", "reasons_bad": ["Course not found"], "reasons_ok": [], "section_suggestions": []}

    c_row = c_row.iloc[0]
    title = str(c_row.get("title", "")).strip()

    # Already completed?
    if course_id in completed:
        reasons_bad.append(f"Already completed: {course_id} — {title}")
        # If already completed, we stop here (still show other info if you want)
        return {"eligible": False, "status": "Already completed", "reasons_bad": reasons_bad, "reasons_ok": [], "section_suggestions": []}

    # Prereqs
    prereq_groups = parse_prereqs(c_row.get("prerequisites", None), c_row.get("prereq_type", None))
    satisfied, missing = prereq_missing(prereq_groups, completed)
    if not satisfied:
        reasons_bad.append("Missing prerequisites: " + ", ".join(missing))
    else:
        if prereq_groups:
            reasons_ok.append("Prerequisites: satisfied")
        else:
            reasons_ok.append("Prerequisites: none")

    # Quarter offering + conflicts
    section_suggestions = []
    if quarter and quarter.strip():
        q = quarter.strip()
        offered_rows = sched[(sched["course_id"] == course_id) & (sched["quarter"] == q)].copy()
        if offered_rows.empty:
            reasons_bad.append(f"Not offered in {q}")
        else:
            reasons_ok.append(f"Offered in {q}")

            # Build student's in-progress schedule for that quarter
            current_courses = inprog[inprog["quarter_taken"] == q]["course_id"].tolist()

            # Gather schedule blocks for student's current courses
            cur_blocks = []
            for cc in current_courses:
                cc_rows = sched[(sched["course_id"] == cc) & (sched["quarter"] == q)]
                if cc_rows.empty:
                    continue
                # assume their section is the first one
                r = cc_rows.iloc[0]
                cur_blocks.append({
                    "course_id": cc,
                    "days": parse_days(r.get("days", "")),
                    "start": parse_time_str(r.get("start_time", "")),
                    "end": parse_time_str(r.get("end_time", "")),
                    "time_block": str(r.get("time_block", "")).strip()
                })

            # Evaluate each offered section of target course
            conflicts_all_sections = True
            for _, r in offered_rows.iterrows():
                sec = str(r.get("section", "")).strip()
                days = parse_days(r.get("days", ""))
                start = parse_time_str(r.get("start_time", ""))
                end = parse_time_str(r.get("end_time", ""))
                tb = str(r.get("time_block", "")).strip()
                conflict_with = []

                for b in cur_blocks:
                    if days and b["days"] and (days & b["days"]):
                        if time_overlap(start, end, b["start"], b["end"]):
                            conflict_with.append(f'{b["course_id"]} ({b["time_block"]})')

                if not conflict_with:
                    conflicts_all_sections = False
                    section_suggestions.append({
                        "section": sec,
                        "days": str(r.get("days", "")).strip(),
                        "start_time": str(r.get("start_time", "")).strip(),
                        "end_time": str(r.get("end_time", "")).strip(),
                        "location": str(r.get("location", "")).strip(),
                        "instructor": str(r.get("instructor", "")).strip(),
                        "modality": str(r.get("modality", "")).strip(),
                        "note": "No conflict found"
                    })
                else:
                    section_suggestions.append({
                        "section": sec,
                        "days": str(r.get("days", "")).strip(),
                        "start_time": str(r.get("start_time", "")).strip(),
                        "end_time": str(r.get("end_time", "")).strip(),
                        "location": str(r.get("location", "")).strip(),
                        "instructor": str(r.get("instructor", "")).strip(),
                        "modality": str(r.get("modality", "")).strip(),
                        "note": "Conflicts with: " + ", ".join(conflict_with)
                    })

            if conflicts_all_sections and current_courses:
                reasons_bad.append("Time conflict in this quarter (every available section conflicts with current courses)")
            elif current_courses:
                reasons_ok.append("Checked time conflicts vs current courses")

    # Final
    eligible = (len(reasons_bad) == 0)
    status = "Eligible" if eligible else "Not eligible"
    return {
        "eligible": eligible,
        "status": status,
        "reasons_bad": reasons_bad,
        "reasons_ok": reasons_ok,
        "section_suggestions": section_suggestions
    }


# =========================
# UI
# =========================
st.title("Course Fit Checker (Prototype)")
st.caption("Pick a student + a course, and this will explain whether the course is a good fit (demo rules: prerequisites + already-taken + optional quarter/time conflicts).")

# Load data
try:
    degree, catalog, sched, students = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Can't find `{DATA_PATH}`. Put `app.py` in the same folder as the Excel file, or update DATA_PATH in the code.")
    st.stop()

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")

    student_names = sorted(students["student_name"].dropna().unique().tolist())
    course_opts = catalog[["course_id", "title"]].dropna().copy()
    course_opts["label"] = course_opts["course_id"] + " — " + course_opts["title"].astype(str)

    quarters = sorted(sched["quarter"].dropna().unique().tolist())

    student_name = st.selectbox("Student", student_names)
    course_label = st.selectbox("Course", course_opts["label"].tolist())
    use_quarter = st.toggle("Also check quarter offering + time conflicts", value=True)
    quarter = st.selectbox("Quarter", quarters) if use_quarter else None

    run = st.button("Check fit", type="primary", use_container_width=True)

# Main content
if run:
    # resolve course_id
    course_id = course_label.split(" — ")[0].strip()
    result = check_fit(student_name, course_id, quarter, catalog, sched, students)

    # Big result
    if result["eligible"]:
        st.success(f"✅ Eligible: {course_label}")
    else:
        st.error(f"❌ Not eligible: {course_label}")

    # Reasons
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Why / what's blocking it")
        if result["reasons_bad"]:
            for r in result["reasons_bad"]:
                st.write("• " + r)
        else:
            st.write("• Nothing blocking it (based on demo rules).")

    with col2:
        st.subheader("What looks good")
        if result["reasons_ok"]:
            for r in result["reasons_ok"]:
                st.write("• " + r)
        else:
            st.write("• —")

    # Section suggestions (if quarter used)
    if use_quarter and quarter:
        st.subheader("Sections in this quarter")
        if result["section_suggestions"]:
            df = pd.DataFrame(result["section_suggestions"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No section info to show (either not offered or no schedule rows).")

    # Quick peek: student's history
    st.subheader("Student course history")
    s_rows = students[students["student_name"] == student_name].copy()
    s_rows = s_rows.sort_values(["quarter_taken", "course_status", "course_id"])
    st.dataframe(
        s_rows[["course_id", "quarter_taken", "course_status", "grade", "cumulative_gpa", "notes"]],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("Choose a student + course on the left, then click **Check fit**.")
