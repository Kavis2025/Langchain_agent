from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


criteria = """
# Prompt Quality Scoring Criteria

1. Clarity (0–10): Checks whether the prompt is easy to understand and has a clear goal.

2. Specificity / Details (0–10): Evaluates whether sufficient details and requirements are provided.

3. Context (0–10): Checks if background information, audience, or use case is mentioned.

4. Output Format & Constraints (0–10): Checks whether expected output format, tone, or length is specified.

5. Persona defined (0–10): Confirms whether a prompt assigns a specific role.

Final Score Calculation: The final score should be the average of the five criteria.
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert Prompt Engineer and Evaluator.

Your task is to evaluate the user's prompt against the following scoring criteria:

{criteria}

### Instructions:
For EACH criterion:
1. Name it exactly
2. Give score (0–10)
3. Provide 1–2 sentence justification

### Output Format STRICTLY:
- Markdown table: Criteria | Score (0–10) | Justification
- Final Score (average, 1 decimal)
- Overall Assessment (2–3 sentences)
- 2–3 actionable improvements

DO NOT deviate from format.
"""),
    ("human", "Evaluate this prompt:\n\n{user_prompt}")
])


chain = prompt_template | llm


while True:
    user_prompt = input("Enter prompt > ").strip()

    if user_prompt.lower() == "exit":
        print("Exiting...")
        break

    response = chain.invoke({
        "criteria": criteria,
        "user_prompt": user_prompt
    })

    print("\n" + "="*80)
    print(response.content)
    print("="*80 + "\n")