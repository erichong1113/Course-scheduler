# CourseFit – Roadmap Planner

CourseFit is a prototype course planning tool that generates a **3-quarter academic roadmap** for a student based on scheduling constraints, prerequisites, and user preferences.

Instead of simply checking whether a single course fits a student's schedule, this system automatically recommends a **set of courses and sections for the next three quarters**, while optimizing factors such as time on campus, walking distance, and course evaluation scores.

The interface is built using **Streamlit** and runs locally as a lightweight planning demo.

---

## Features

- Generate a **3-quarter course roadmap**
- Automatically select **course sections and schedules**
- Enforce **prerequisite constraints**
- Avoid **time conflicts**
- Optimize schedules using preference weights:
  - fewer trips to campus
  - less time spent on campus
  - less walking between buildings
  - higher CTEC (course evaluation) scores

---

## Dataset

The prototype uses a simulated dataset containing:

- Course scheduling information  
- Student course history  
- Prerequisite relationships  
- Course evaluation scores (CTEC)

Required dataset file: northwestern_tgs_scheduling_dataset_3.xlsx

---

## Running the App

Install the required dependencies:

```bash
pip install streamlit pandas openpyxl

Run the roadmap planner:streamlit run app_v4_roadmap_fixed.py
