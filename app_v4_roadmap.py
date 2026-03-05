import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

# =========================================================
# Pretty UI
# =========================================================
st.set_page_config(page_title="CourseFit Roadmap Planner", layout="wide")

st.markdown("""
<style>
:root{
  --bg0: rgba(255,255,255,0.03);
  --bd0: rgba(255,255,255,0.10);
  --txt0: rgba(255,255,255,0.92);
  --txt1: rgba(255,255,255,0.70);
  --ok: rgba(52, 211, 153, 0.25);
  --bad: rgba(248, 113, 113, 0.20);
  --warn: rgba(251, 191, 36, 0.20);
}
.block-container {padding-top: 1.2rem;}
.hero {
  border: 1px solid var(--bd0);
  border-radius: 22px;
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(99,102,241,0.20), rgba(16,185,129,0.12), rgba(236,72,153,0.12));
}
.hero h1 {margin: 0; font-size: 1.65rem;}
.hero p {margin: 6px 0 0 0; color: var(--txt1);}
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
.small {color: var(--txt1); font-size: 0.90rem;}
hr {border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 16px 0;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "northwestern_tgs_scheduling_dataset_3.xlsx"

# =========================================================
# Helpers: time, days, quarters
# =========================================================
DAY_MAP = {"M": {"Mon"}, "T": {"Tue"}, "W": {"Wed"}, "Th": {"Thu"}, "F": {"Fri"}, "Sa": {"Sat"}, "Su": {"Sun"}}
TERM_ORDER = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}

def parse_days(days_str: str) -> set[str]:
    if not isinstance(days_str, str) or not days_str.strip():
        return set()
    s = days_str.strip()
    out = set()
    i = 0
    while i < len(s):
        if s[i:i+2] == "Th":
            out |= DAY_MAP["Th"]; i += 2
        else:
            ch = s[i]
            if ch in DAY_MAP:
                out |= DAY_MAP[ch]
            i += 1
    return out

def parse_time_str(t) -> Optional[time]:
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

def is_on_campus(modality: Optional[str]) -> bool:
    if not isinstance(modality, str):
        return True
    m = modality.strip().lower()
    return not ("online" in m or "remote" in m)

def parse_quarter_key(q: str) -> Tuple[int,int,str]:
    if not isinstance(q, str):
        return (0, 0, str(q))
    s = q.strip().replace("_", "-").replace(" ", "-")
    parts = [p for p in s.split("-") if p]
    year = None
    term = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = int(p)
        if p.capitalize() in TERM_ORDER:
            term = p.capitalize()
    if year is None:
        try:
            year = int(parts[0])
        except Exception:
            year = 0
    if term is None:
        for t in TERM_ORDER:
            if t.lower() in q.lower():
                term = t
                break
    term_rank = TERM_ORDER.get(term, 99)
    return (year, term_rank, q)

# =========================================================
# Prereqs
# =========================================================
def parse_prereqs(prereq_str: str, prereq_type: Optional[str]) -> List[List[str]]:
    if not isinstance(prereq_str, str) or not prereq_str.strip() or prereq_str.strip().lower() == "nan":
        return []
    prereq_str = prereq_str.strip()

    if prereq_type == "any_one_required":
        parts = [p.strip() for p in prereq_str.split(" OR ")]
        parts = [p for p in parts if p]
        return [parts] if parts else []

    parts = [p.strip() for p in prereq_str.split(";")]
    parts = [p for p in parts if p]
    return [[p] for p in parts] if parts else []

def prereq_satisfied(groups: List[List[str]], completed: Set[str]) -> Tuple[bool, List[str]]:
    missing = []
    for g in groups:
        if len(g) == 1:
            if g[0] not in completed:
                missing.append(g[0])
        else:
            if not any(c in completed for c in g):
                missing.append(" OR ".join(g))
    return (len(missing) == 0), missing

# =========================================================
# Metrics + scoring
# =========================================================
def compute_unique_days_on_campus(blocks) -> float:
    days = set()
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        days |= set(b["days"])
    return float(len(days))

def compute_unique_buildings(blocks) -> float:
    bld = set()
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        name = str(b.get("building_name", "")).strip()
        if name and name.lower() != "nan":
            bld.add(name)
    return float(len(bld))

def compute_day_spans_minutes(blocks) -> float:
    day_to_times = {}
    for b in blocks:
        if not is_on_campus(b.get("modality")):
            continue
        if b["start"] is None or b["end"] is None:
            continue
        for d in b["days"]:
            day_to_times.setdefault(d, []).append((b["start"], b["end"]))
    total = 0.0
    for intervals in day_to_times.values():
        starts = [t[0] for t in intervals]; ends = [t[1] for t in intervals]
        earliest, latest = min(starts), max(ends)
        total += (datetime.combine(datetime.today(), latest) - datetime.combine(datetime.today(), earliest)).seconds / 60
    return float(total)

def normalize_penalty(trips: float, time_min: float, buildings: float, ctec: Optional[float]) -> Tuple[float,float,float,float]:
    p_trips = min(1.0, trips / 5.0)
    p_time = min(1.0, time_min / 600.0)
    p_walk = min(1.0, buildings / 6.0)
    if ctec is None or (isinstance(ctec, float) and np.isnan(ctec)):
        p_ctec = 0.5
    else:
        p_ctec = 1.0 - min(1.0, float(ctec) / 6.0)
    return p_trips, p_time, p_walk, p_ctec

def weighted_suitability(trips, time_min, buildings, ctec, w_trips, w_time, w_walk, w_ctec) -> float:
    p_trips, p_time, p_walk, p_ctec = normalize_penalty(trips, time_min, buildings, ctec)
    den = max(1e-9, float(w_trips + w_time + w_walk + w_ctec))
    penalty = (w_trips*p_trips + w_time*p_time + w_walk*p_walk + w_ctec*p_ctec) / den
    return float((1.0 - penalty) * 100.0)

# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(path: str):
    sched = pd.read_excel(path, sheet_name="Course_Scheduling")
    students = pd.read_excel(path, sheet_name="Student_Profiles")
    ctec = pd.read_excel(path, sheet_name="CTEC_Evaluations")

    for df in (sched, students, ctec):
        if "course_id" in df.columns:
            df["course_id"] = df["course_id"].astype(str).str.strip()

    students["student_name"] = students["student_name"].astype(str).str.strip()
    students["quarter_taken"] = students["quarter_taken"].astype(str).str.strip()
    sched["quarter"] = sched["quarter"].astype(str).str.strip()

    for col in ["section","title","department","level","days","start_time","end_time","time_block","modality",
                "location","building_name","building_address","instructor","credits",
                "prerequisites","prereq_type","corequisites"]:
        if col in sched.columns:
            sched[col] = sched[col].astype(str).str.strip()

    sched["_start"] = sched["start_time"].apply(parse_time_str)
    sched["_end"] = sched["end_time"].apply(parse_time_str)
    return sched, students, ctec

def get_ctec(ctec: pd.DataFrame, course_id: str, instructor: str) -> Optional[float]:
    if ctec.empty:
        return None
    df = ctec[ctec["course_id"] == course_id]
    if df.empty or "overall_teaching" not in df.columns:
        return None
    if instructor and isinstance(instructor, str) and instructor.strip():
        df2 = df[df["instructor"].astype(str).str.strip() == instructor.strip()]
        if not df2.empty:
            return float(df2["overall_teaching"].mean())
    return float(df["overall_teaching"].mean())

# =========================================================
# Scheduling primitives
# =========================================================
def section_to_block(r: pd.Series) -> Dict:
    return {
        "course_id": str(r.get("course_id","")).strip(),
        "title": str(r.get("title","")).strip(),
        "section": str(r.get("section","")).strip(),
        "days": parse_days(str(r.get("days","")).strip()),
        "start": r.get("_start"),
        "end": r.get("_end"),
        "time_block": str(r.get("time_block","")).strip(),
        "modality": str(r.get("modality","")).strip(),
        "location": str(r.get("location","")).strip(),
        "building_name": str(r.get("building_name","")).strip(),
        "instructor": str(r.get("instructor","")).strip(),
        "credits": float(r.get("credits")) if str(r.get("credits","")).replace(".","",1).isdigit() else None,
    }

def conflicts_with_blocks(block: Dict, blocks: List[Dict]) -> List[str]:
    conflict = []
    for b in blocks:
        if block["days"] and b["days"] and (block["days"] & b["days"]):
            if time_overlap(block["start"], block["end"], b["start"], b["end"]):
                conflict.append(f'{b["course_id"]} ({b.get("time_block","")})')
    return conflict

def build_fixed_blocks(students: pd.DataFrame, sched: pd.DataFrame, student_name: str, quarter: str) -> List[Dict]:
    s = students[(students["student_name"] == student_name) &
                 (students["course_status"] == "in_progress") &
                 (students["quarter_taken"] == quarter)]
    blocks = []
    for cid in s["course_id"].tolist():
        rows = sched[(sched["course_id"] == cid) & (sched["quarter"] == quarter)]
        if rows.empty:
            continue
        blocks.append(section_to_block(rows.iloc[0]))
    return blocks

def completed_set(students: pd.DataFrame, student_name: str) -> Set[str]:
    s = students[(students["student_name"] == student_name) & (students["course_status"] == "completed")]
    return set(s["course_id"].tolist())

# =========================================================
# Roadmap planning
# =========================================================
@dataclass
class PlannedSection:
    course_id: str
    title: str
    section: str
    quarter: str
    days: str
    start_time: str
    end_time: str
    modality: str
    instructor: str
    location: str
    building: str
    ctec: Optional[float]
    conflicts: str

@dataclass
class QuarterPlan:
    quarter: str
    fixed_blocks: List[Dict]
    chosen_blocks: List[Dict]
    chosen_sections: List[PlannedSection]
    kpis: Dict[str, float]

@dataclass
class Roadmap:
    plans: List[QuarterPlan]
    total_score: float

def quarter_plan_score(blocks: List[Dict], ctecs: List[Optional[float]],
                       w_trips, w_time, w_walk, w_ctec) -> Tuple[float, Dict[str,float]]:
    trips = compute_unique_days_on_campus(blocks)
    tmin = compute_day_spans_minutes(blocks)
    bld = compute_unique_buildings(blocks)
    vals = [v for v in ctecs if v is not None and not (isinstance(v,float) and np.isnan(v))]
    avg_ctec = float(np.mean(vals)) if vals else np.nan
    score = weighted_suitability(trips, tmin, bld, avg_ctec, w_trips, w_time, w_walk, w_ctec)
    return score, {"days_on_campus": trips, "time_on_campus_min": tmin, "unique_buildings": bld, "avg_ctec": avg_ctec, "score": score}

def build_candidate_sections(sched: pd.DataFrame, ctec: pd.DataFrame, quarter: str, completed: Set[str]) -> pd.DataFrame:
    df = sched[sched["quarter"] == quarter].copy()
    df = df[~df["course_id"].isin(list(completed))]
    df["_ctec"] = df.apply(lambda r: get_ctec(ctec, str(r["course_id"]).strip(), str(r.get("instructor","")).strip()), axis=1)
    return df

def course_prereq_ok(course_rows: pd.DataFrame, completed_by_time: Set[str]) -> Tuple[bool, List[str]]:
    r0 = course_rows.iloc[0]
    prereq_str = str(r0.get("prerequisites","")).strip()
    prereq_type = str(r0.get("prereq_type","")).strip()
    if prereq_type.lower() == "nan":
        prereq_type = None
    groups = parse_prereqs(prereq_str, prereq_type)
    ok, missing = prereq_satisfied(groups, completed_by_time)
    return ok, missing

def greedy_fill_one_quarter(course_rows_by_id: Dict[str, pd.DataFrame],
                            fixed_blocks: List[Dict],
                            target_n: int,
                            w_trips, w_time, w_walk, w_ctec,
                            enforce_no_conflict: bool,
                            seed: Optional[Dict] = None) -> Tuple[List[Dict], List[PlannedSection], float]:
    chosen_blocks: List[Dict] = []
    chosen_sections: List[PlannedSection] = []
    used_courses = set()

    def all_blocks():
        return fixed_blocks + chosen_blocks

    # seed
    if seed:
        conf = conflicts_with_blocks(seed, all_blocks())
        if not (enforce_no_conflict and conf):
            chosen_blocks.append(seed)
            used_courses.add(seed["course_id"])

    while len(chosen_blocks) < target_n:
        best_pick = None
        best_score = -1e9

        for cid, rows in course_rows_by_id.items():
            if cid in used_courses:
                continue

            # choose best section for this course given current blocks
            best_sec = None
            best_sec_score = -1e9
            best_conf = []

            for _, r in rows.iterrows():
                blk = section_to_block(r)
                conf = conflicts_with_blocks(blk, all_blocks())
                if enforce_no_conflict and conf:
                    continue

                tentative_blocks = all_blocks() + [blk]
                score, _ = quarter_plan_score(tentative_blocks, [r.get("_ctec", None)], w_trips, w_time, w_walk, w_ctec)
                if score > best_sec_score:
                    best_sec_score = score
                    best_sec = (blk, r)
                    best_conf = conf

            if best_sec is None:
                continue

            if best_sec_score > best_score:
                best_score = best_sec_score
                best_pick = (best_sec[0], best_sec[1], best_conf)

        if best_pick is None:
            break

        blk, r, conf = best_pick
        chosen_blocks.append(blk)
        used_courses.add(blk["course_id"])
        chosen_sections.append(PlannedSection(
            course_id=blk["course_id"],
            title=str(r.get("title","")).strip(),
            section=str(r.get("section","")).strip(),
            quarter=str(r.get("quarter","")).strip(),
            days=str(r.get("days","")).strip(),
            start_time=str(r.get("start_time","")).strip(),
            end_time=str(r.get("end_time","")).strip(),
            modality=str(r.get("modality","")).strip(),
            instructor=str(r.get("instructor","")).strip(),
            location=str(r.get("location","")).strip(),
            building=str(r.get("building_name","")).strip(),
            ctec=r.get("_ctec", None),
            conflicts=", ".join(conf) if conf else ""
        ))

    final_blocks = all_blocks()
    final_ctecs = [s.ctec for s in chosen_sections]
    q_score, _ = quarter_plan_score(final_blocks, final_ctecs, w_trips, w_time, w_walk, w_ctec)
    return chosen_blocks, chosen_sections, q_score

def plan_roadmap(sched: pd.DataFrame, students: pd.DataFrame, ctec: pd.DataFrame,
                 student_name: str, start_quarter: str, num_quarters: int,
                 target_courses_per_q: int,
                 w_trips, w_time, w_walk, w_ctec,
                 enforce_no_conflict: bool,
                 beam_width: int = 8,
                 top_k_outputs: int = 3) -> List[Roadmap]:

    quarters_all = sorted(sched["quarter"].dropna().unique().tolist(), key=parse_quarter_key)
    if start_quarter not in quarters_all:
        start_quarter = quarters_all[0]
    start_idx = quarters_all.index(start_quarter)
    q_list = quarters_all[start_idx:start_idx + num_quarters]
    if not q_list:
        return []

    completed0 = completed_set(students, student_name)
    beam = [([], set(completed0), 0.0)]  # plans_so_far, completed_by_time, total_score

    for q in q_list:
        new_beam = []
        for plans_so_far, completed_by_time, total_score in beam:
            fixed = build_fixed_blocks(students, sched, student_name, q)

            cand = build_candidate_sections(sched, ctec, q, completed0)

            # group by course_id; filter by prereqs
            course_rows_by_id = {}
            for cid, rows in cand.groupby("course_id"):
                ok, _missing = course_prereq_ok(rows, completed_by_time)
                if ok:
                    course_rows_by_id[cid] = rows

            if not course_rows_by_id:
                q_score, kpis = quarter_plan_score(fixed, [], w_trips, w_time, w_walk, w_ctec)
                qplan = QuarterPlan(quarter=q, fixed_blocks=fixed, chosen_blocks=[], chosen_sections=[], kpis=kpis)
                new_beam.append((plans_so_far + [qplan], set(completed_by_time), total_score + q_score))
                continue

            # seeds: pick good-looking sections to diversify
            seed_blocks = []
            for cid, rows in course_rows_by_id.items():
                rr = rows.copy()
                rr["_start_min"] = rr["_start"].apply(lambda t: t.hour*60+t.minute if isinstance(t,time) else 9999)
                rr = rr.sort_values(["_ctec","_start_min"], ascending=[False, True]).head(1)
                seed_blocks.append(section_to_block(rr.iloc[0]))

            seeds = seed_blocks[:beam_width] if seed_blocks else [None]

            for seed in seeds[:beam_width]:
                chosen_blocks, chosen_sections, q_score = greedy_fill_one_quarter(
                    course_rows_by_id=course_rows_by_id,
                    fixed_blocks=fixed,
                    target_n=target_courses_per_q,
                    w_trips=w_trips, w_time=w_time, w_walk=w_walk, w_ctec=w_ctec,
                    enforce_no_conflict=enforce_no_conflict,
                    seed=seed
                )

                all_blocks = fixed + chosen_blocks
                q_score, kpis = quarter_plan_score(all_blocks, [s.ctec for s in chosen_sections], w_trips, w_time, w_walk, w_ctec)
                qplan = QuarterPlan(quarter=q, fixed_blocks=fixed, chosen_blocks=chosen_blocks, chosen_sections=chosen_sections, kpis=kpis)

                new_completed = set(completed_by_time) | set([b["course_id"] for b in chosen_blocks])
                new_beam.append((plans_so_far + [qplan], new_completed, total_score + q_score))

        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]

    # to roadmaps, dedupe by signature
    roadmaps: List[Roadmap] = []
    seen = set()
    for plans, _, total in beam:
        sig = tuple((p.quarter, tuple(sorted([s.course_id + ":" + s.section for s in p.chosen_sections]))) for p in plans)
        if sig in seen:
            continue
        seen.add(sig)
        roadmaps.append(Roadmap(plans=plans, total_score=float(total)))

    roadmaps.sort(key=lambda r: r.total_score, reverse=True)
    return roadmaps[:top_k_outputs]

# =========================================================
# UI
# =========================================================
st.markdown("""
<div class="hero">
  <h1>CourseFit Roadmap Planner</h1>
  <p>Generates a sample plan for the <b>next 3 quarters</b> (courses + sections), optimized for your priorities.</p>
</div>
""", unsafe_allow_html=True)

try:
    sched, students, ctec = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Can't find `{DATA_PATH}`. Put `app_v4_roadmap.py` in the same folder as the Excel file, or update DATA_PATH in the code.")
    st.stop()

student_names = sorted(students["student_name"].dropna().unique().tolist())
quarters_all = sorted(sched["quarter"].dropna().unique().tolist(), key=parse_quarter_key)

c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.2])
with c1:
    student_name = st.selectbox("Student", student_names, index=0)
with c2:
    start_quarter = st.selectbox("Start quarter", quarters_all, index=max(0, len(quarters_all)-3))
with c3:
    target_courses_per_q = st.slider("Courses / quarter", 1, 4, 2)
with c4:
    enforce_no_conflict = st.toggle("Hard block time conflicts", value=True)

with st.expander("🎛️ Preferences & priorities", expanded=True):
    a,b,c,d,e = st.columns([1,1,1,1,1.2])
    with a:
        w_trips = st.slider("Fewer trips", 0, 5, 3)
    with b:
        w_time = st.slider("Less time on campus", 0, 5, 2)
    with c:
        w_walk = st.slider("Less walking", 0, 5, 1)
    with d:
        w_ctec = st.slider("Higher CTECs", 0, 5, 3)
    with e:
        beam_width = st.slider("Planner diversity", 3, 15, 8)

run = st.button("🧭 Generate 3-quarter roadmap", type="primary", use_container_width=True)
if not run:
    st.caption("Tip: change weights and regenerate to see different roadmaps.")
    st.stop()

roadmaps = plan_roadmap(
    sched=sched, students=students, ctec=ctec,
    student_name=student_name, start_quarter=start_quarter, num_quarters=3,
    target_courses_per_q=target_courses_per_q,
    w_trips=w_trips, w_time=w_time, w_walk=w_walk, w_ctec=w_ctec,
    enforce_no_conflict=enforce_no_conflict,
    beam_width=beam_width,
    top_k_outputs=3
)

if not roadmaps:
    st.warning("No roadmap could be generated (not enough quarter data or no offered courses).")
    st.stop()

opt_labels = [f"Option {i+1} — Total score {rm.total_score:.0f}" for i, rm in enumerate(roadmaps)]
choice = st.selectbox("Roadmap options", opt_labels, index=0)
rm = roadmaps[opt_labels.index(choice)]

k1,k2,k3,k4 = st.columns(4)
k1.markdown(f"<div class='card'><div class='label'>Student</div><div class='value'>{student_name}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'><div class='label'>Start quarter</div><div class='value'>{start_quarter}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'><div class='label'>Courses/quarter</div><div class='value'>{target_courses_per_q}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='card'><div class='label'>Total roadmap score</div><div class='value'>{rm.total_score:.0f}</div></div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

cols = st.columns(3)
for i, qp in enumerate(rm.plans):
    with cols[i]:
        st.markdown(
            f"<div class='card'><div class='label'>Quarter</div><div class='value'>{qp.quarter}</div>"
            f"<div class='small'>Score: <b>{qp.kpis['score']:.0f}</b> • Days: <b>{qp.kpis['days_on_campus']:.0f}</b> • "
            f"Time: <b>{qp.kpis['time_on_campus_min']:.0f} min</b> • Buildings: <b>{qp.kpis['unique_buildings']:.0f}</b></div>"
            "</div>", unsafe_allow_html=True
        )
        if qp.fixed_blocks:
            st.markdown("<span class='pill warn'>Has fixed in-progress schedule</span>", unsafe_allow_html=True)

        if not qp.chosen_sections:
            st.info("No new courses selected this quarter (prereqs/offering/conflicts).")
        else:
            for s in qp.chosen_sections:
                badges = []
                if s.modality and ("online" in s.modality.lower() or "remote" in s.modality.lower()):
                    badges.append("<span class='pill ok'>Online</span>")
                if s.conflicts:
                    badges.append("<span class='pill bad'>Conflict</span>")
                else:
                    badges.append("<span class='pill ok'>No conflict</span>")
                if s.ctec is not None and not (isinstance(s.ctec,float) and np.isnan(s.ctec)):
                    badges.append(f"<span class='pill'>CTEC {float(s.ctec):.2f}</span>")

                st.markdown(
                    f"<div class='card' style='margin-top:10px;'>"
                    f"<div class='label'>{s.course_id} • Section {s.section}</div>"
                    f"<div class='value'>{s.title}</div>"
                    f"<div class='small'>{s.days} • {s.start_time}–{s.end_time} • {s.modality}</div>"
                    f"<div class='small'>Instructor: {s.instructor or '—'} • {s.location} {('(' + s.building + ')') if s.building else ''}</div>"
                    f"<div>{''.join(badges)}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

st.markdown("<hr/>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧾 Roadmap table", "🔎 Planner logic (MVP)"])
with tab1:
    rows = []
    for qp in rm.plans:
        for s in qp.chosen_sections:
            rows.append({
                "quarter": s.quarter,
                "course_id": s.course_id,
                "title": s.title,
                "section": s.section,
                "days": s.days,
                "start": s.start_time,
                "end": s.end_time,
                "modality": s.modality,
                "instructor": s.instructor,
                "building": s.building,
                "ctec": s.ctec,
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No planned courses to show.")

with tab2:
    st.write("• Plans quarter-by-quarter for the next 3 quarters starting from your selected start quarter.")
    st.write("• Only uses courses offered in that quarter.")
    st.write("• Filters out courses the student already completed.")
    st.write("• Enforces prerequisites using completed + courses planned in earlier quarters.")
    st.write("• Picks up to N courses per quarter and chooses sections that avoid time conflicts (if hard blocking is on).")
    st.write("• Scores each quarter using your weights: fewer trips (days on campus), less time on campus, less walking (proxy: fewer buildings), higher CTECs.")
    st.caption("Demo simplification: for any in-progress course, we assume the student is in the first listed section of that course.")
