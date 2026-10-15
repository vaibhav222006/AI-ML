import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


class StudentDiagnosticEngine:
    """Supervised Machine Learning Engine to diagnose student mastery level and knowledge gaps."""

    MASTERY_METADATA = {
        1: {"label": "Remedial Level", "strategy": "Provide real-world relatable analogies & basic foundational concept reviews."},
        2: {"label": "Beginner Level", "strategy": "Offer step-by-step guided practice with instant hints."},
        3: {"label": "Intermediate Level", "strategy": "Present standard curriculum questions with speed and accuracy targets."},
        4: {"label": "Advanced Level", "strategy": "Deliver challenging multi-step analytical problem sets."}
    }

    def __init__(self, dataset_path="student_dataset.csv"):
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.topic_encoder = LabelEncoder()
        self.error_encoder = LabelEncoder()
        self.is_trained = False

    def load_and_preprocess(self):
        """Loads dataset from path with automated fallback if missing."""
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
        else:
            print(f"Warning: '{self.dataset_path}' not found. Initializing fallback dataset.")
            self.df = pd.DataFrame([
                {"topic": "Algebra", "recent_score_pct": 35, "quiz_attempts": 5, "time_spent_mins": 30, "primary_error_type": "Conceptual Misunderstanding", "mastery_level_code": 1},
                {"topic": "Algebra", "recent_score_pct": 55, "quiz_attempts": 3, "time_spent_mins": 20, "primary_error_type": "Formula Misapplication", "mastery_level_code": 2},
                {"topic": "Algebra", "recent_score_pct": 75, "quiz_attempts": 2, "time_spent_mins": 15, "primary_error_type": "Calculation Error", "mastery_level_code": 3},
                {"topic": "Algebra", "recent_score_pct": 92, "quiz_attempts": 1, "time_spent_mins": 10, "primary_error_type": "None", "mastery_level_code": 4}
            ])

        self.df['topic_encoded'] = self.topic_encoder.fit_transform(self.df['topic'])
        self.df['error_encoded'] = self.error_encoder.fit_transform(self.df['primary_error_type'])

    def train(self):
        """Trains the Supervised Learning Model on student performance features."""
        self.load_and_preprocess()

        feature_cols = ['topic_encoded', 'recent_score_pct', 'quiz_attempts', 'time_spent_mins', 'error_encoded']
        X = self.df[feature_cols]
        y = self.df['mastery_level_code']

        self.model.fit(X, y)
        self.is_trained = True

        if len(self.df) >= 10:
            cv_scores = cross_val_score(self.model, X, y, cv=3)
            print(f"Supervised Student Diagnostic Model Trained. (CV Accuracy: {np.mean(cv_scores)*100:.1f}%)")
        else:
            print("Supervised Student Diagnostic Model Trained Successfully.")

    def diagnose_student(self, student_performance):
        """Predicts student mastery level and identifies error patterns."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call .train() first.")

        topic_enc = self.topic_encoder.transform([student_performance['topic']])[0] if student_performance['topic'] in self.topic_encoder.classes_ else 0
        error_enc = self.error_encoder.transform([student_performance['primary_error_type']])[0] if student_performance['primary_error_type'] in self.error_encoder.classes_ else 0

        features_df = pd.DataFrame([{
            'topic_encoded': topic_enc,
            'recent_score_pct': student_performance['recent_score_pct'],
            'quiz_attempts': student_performance['quiz_attempts'],
            'time_spent_mins': student_performance['time_spent_mins'],
            'error_encoded': error_enc
        }])

        predicted_code = int(self.model.predict(features_df)[0])
        probabilities = self.model.predict_proba(features_df)[0]
        confidence = float(np.max(probabilities) * 100)

        meta = self.MASTERY_METADATA.get(predicted_code, {"label": "Unknown", "strategy": "Standard Review"})

        return {
            "student_id": student_performance.get("student_id", "STU-NEW"),
            "topic": student_performance['topic'],
            "recent_score": f"{student_performance['recent_score_pct']}%",
            "diagnosed_level_code": predicted_code,
            "mastery_status": meta["label"],
            "model_confidence": f"{confidence:.1f}%",
            "recommended_pedagogy": meta["strategy"],
            "identified_error_pattern": student_performance['primary_error_type']
        }

class AdaptiveQuestionRecommender:
    """Content & Analogy Recommendation Engine using Pandas filtering and adaptive mapping."""

    def __init__(self, question_bank_path="question_bank.csv"):
        self.question_bank_path = question_bank_path
        self.qb_df = self._load_question_bank()

    def _load_question_bank(self):
        if os.path.exists(self.question_bank_path):
            return pd.read_csv(self.question_bank_path)
        else:
            return pd.DataFrame([
                {
                    "question_id": "Q-CIV-101",
                    "topic": "Indian Civics & Governance",
                    "difficulty_level": 1,
                    "concept": "Power Sharing",
                    "question_text": "What is horizontal power sharing in a democracy?",
                    "correct_answer": "Sharing power among different organs of government (Legislature, Executive, Judiciary)",
                    "technical_explanation": "Power is distributed at the same level among Legislature, Executive, and Judiciary to ensure checks and balances.",
                    "real_world_analogy": "It's like a cricket team where the Captain, Wicketkeeper, and Coach have equal authority in different decisions so no single person controls everything."
                }
            ])

    def generate_adaptive_lesson(self, diagnostic_result):
        """Selects the best difficulty question and renders a tailored real-world analogy."""
        topic = diagnostic_result['topic']
        level = diagnostic_result['diagnosed_level_code']

        filtered_q = self.qb_df[self.qb_df['topic'] == topic]
        if filtered_q.empty:
            filtered_q = self.qb_df

        filtered_q = filtered_q.assign(diff_diff=np.abs(filtered_q['difficulty_level'] - level))
        selected_question = filtered_q.sort_values(by='diff_diff').iloc[0]

        output = [
            "=" * 65,
            "         AUTONOMOUS ADAPTIVE TUTOR: PERSONALIZED STUDY PLAN      ",
            "=" * 65,
            f" Student ID           : {diagnostic_result['student_id']}",
            f" Subject/Topic        : {diagnostic_result['topic']}",
            f" Diagnostic Status     : {diagnostic_result['mastery_status']} ({diagnostic_result['recent_score']})",
            f" Error Pattern        : {diagnostic_result['identified_error_pattern']}",
            f" Pedagogical Strategy : {diagnostic_result['recommended_pedagogy']}",
            "-" * 65,
            "\nRECOMMENDED ADAPTIVE PRACTICE QUESTION:",
            f"   [ID: {selected_question['question_id']}] (Difficulty Level: {selected_question['difficulty_level']})",
            f"   Concept       : {selected_question['concept']}",
            f"   Question      : {selected_question['question_text']}",
            f"   Correct Answer: {selected_question['correct_answer']}",
            "\n TECHNICAL EXPLANATION:",
            f"   {selected_question['technical_explanation']}",
            "\n TAILORED REAL-WORLD ANALOGY (FOR EASY UNDERSTANDING):",
            f"   {selected_question['real_world_analogy']}",
            "=" * 65
        ]

        return "\n".join(output)

if __name__ == "__main__":
   
    tutor_engine = StudentDiagnosticEngine(dataset_path="student_dataset.csv")
    tutor_engine.train()

    student_profile = {
        "student_id": "STU-1042",
        "topic": "Indian Civics & Governance",
        "recent_score_pct": 38,
        "quiz_attempts": 5,
        "time_spent_mins": 35,
        "primary_error_type": "Conceptual Misunderstanding"
    }

    print("\n--- STUDENT PERFORMANCE INPUT ---")
    print(json.dumps(student_profile, indent=2))

    diagnostic_result = tutor_engine.diagnose_student(student_profile)
    print("\n--- TUTOR DIAGNOSTIC OUTPUT (JSON) ---")
    print(json.dumps(diagnostic_result, indent=2))

    recommender = AdaptiveQuestionRecommender(question_bank_path="question_bank.csv")
    lesson_plan = recommender.generate_adaptive_lesson(diagnostic_result)

    print("\n--- ADAPTIVE TUTOR LESSON OUTPUT ---")
    print(lesson_plan)