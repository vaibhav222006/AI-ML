# Autonomous Adaptive Tutor (SDG 4: Quality Education)

An AI-powered adaptive tutoring system designed for under-resourced classrooms. It diagnoses individual student knowledge gaps using supervised machine learning and delivers personalized practice questions alongside relatable real-world analogies.

##  Project Features
- **Supervised ML Diagnostic Engine:** Trained `RandomForestClassifier` model predicting 4 student mastery levels (Remedial, Beginner, Intermediate, Advanced).
- **Adaptive Question Bank:** Dynamic question selection matched to student mastery level across multiple subjects (Math, Civics, CS, Grammar, Physics).
- **Real-World Analogy Engine:** Translates complex technical academic concepts into intuitive real-world analogies.
- **SDG 4 Alignment:** Promotes equitable and quality education by bridging the student-teacher ratio gap.

##  Repository File Structure
- `main.py` - Complete Python Machine Learning and adaptive tutoring script.
- `student_dataset.csv` - Generated dataset of 120 student historical quiz responses.
- `question_bank.csv` - Multidisciplinary question bank with technical explanations and analogies.
- `Lean_Canvas.png` - Visual Lean Canvas diagram for project business alignment.

##  How to Run the Pipeline
```bash
# 1. Install required dependencies
pip install pandas numpy scikit-learn

# 2. Run the main adaptive tutor script
python adaptive_tutor_pipeline.py
