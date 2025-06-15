import streamlit as st
from transformers import pipeline
import wikipedia
import nltk
import google.generativeai as genai

genai.configure(api_key='')

model = genai.GenerativeModel('gemini-1.5-flash')
# Instantiate the generative model
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.corpus import brown

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load Hugging Face pipeline for Q&A
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# === App UI ===
st.set_page_config(page_title="AI in Education Tutor", layout="centered")
st.title("🎓 AI in Education: Virtual Tutor")

# 1. Personalized Learning
st.header("📘 Personalized Learning Suggestion")
score = st.slider("Enter your last test score (0-100):", 0, 100, 0)

def recommend_topic(score):
    if score <= 30:
        return "Recommended Topic: Basics of Python"
    elif score <= 40:
        return "Recommended Topic: If Else Statement "
    elif score <= 50:
        return "Recommended Topic: Functions and Loops"
    elif score <= 60:
        return "Recommended Topic: List, Tuples, Dist"
    elif score <= 70:
        return "Recommended Topic: Make some projects and practice it Well"
    else:
        return "Recommended Topic: Object-Oriented Programming"

st.success(recommend_topic(score))

st.header("💡 Ask the AI Tutor a Question")

user_question = st.text_input("Type your question (e.g., What is Python?)")

if user_question:
    # Try finding the most relevant page using search
    # search_results = wikipedia.search(user_question)
    response = model.generate_content(f"give me answer in 5 line for this question {user_question}")
    st.markdown(f"**Tutor Answer:** {response.text}")

# === Self-Grading Quiz with Online Answer Key ===
st.header("🧠 Self-Grading (Answers fetched from Wikipedia)")

questions = [
    "What is Python?",
    "What is artificial intelligence?",
    "What is a function in programming?"
]

student_answers = []
correct_answers = []
score = 0

if st.checkbox("Attempt quiz with AI-checked answers"):
    st.markdown("### Answer the questions:")

    for i, question in enumerate(questions):
        student_input = st.text_input(f"{i+1}. {question}", key=f"q{i}")
        student_answers.append(student_input)

    if st.button("Submit Answers"):
        st.markdown("---\n### 📖 Correct Answers & Grading:")

        for i, question in enumerate(questions):
            try:
                prompt = f"""
You're an AI teacher. A student has answered a question.

**Goal:** Provide the correct answer in 1 line and evaluate the student's response as:
- **1**: Fully correct
- **0.5**: Partially correct or contextually related
- **0**: Wrong

### Question:
{question}

### Student's Answer:
{student_answers[i]}

Please respond in this format only:
Correct Answer: <one-line correct answer>
Score: <1 / 0.5 / 0>
Explanation: <brief if score is less than 1>
"""

                result = model.generate_content(prompt)
                response = result.text.strip().splitlines()

                correct_answer = response[0].replace("Correct Answer: ", "").strip()
                score_line = response[1].replace("Score: ", "").strip()
                explanation = response[2].replace("Explanation: ", "").strip() if len(response) > 2 else ""

                try:
                    current_score = float(score_line)
                except:
                    current_score = 0

                correct_answers.append(correct_answer)
                score += current_score

                st.markdown(f"**Q{i+1}: {question}**")
                st.write(f"Your Answer: `{student_answers[i]}`")
                st.write(f"Correct Answer: `{correct_answer}`")

                if current_score == 1:
                    st.success("✅ Fully Correct (1 mark)")
                elif current_score == 0.5:
                    st.info(f"🟡 Partially Correct (0.5 mark) — {explanation}")
                else:
                    st.error(f"❌ Incorrect (0 mark) — {explanation}")

                st.markdown("---")

            except Exception as e:
                st.error(f"Error fetching answer: {e}")

        percentage = (score / len(questions)) * 100
        st.subheader(f"Final Score: {score}/{len(questions)} ({percentage:.2f}%)")
        if percentage < 50:
            st.warning("You might want to review the material.")
        elif percentage < 75:
            st.info("Good job! A bit more practice will help.")
        else:
            st.success("Excellent work! Keep it up!")
