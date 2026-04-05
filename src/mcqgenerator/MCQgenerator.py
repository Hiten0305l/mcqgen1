import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import PromptTemplate

# load environment variables
load_dotenv()

# get GROQ API key
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template
quiz_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone"],
    template="""
You are an expert MCQ creator.

Create exactly {number} multiple-choice questions about {subject}.
Use a {tone} tone.

STRICT FORMAT FOR EVERY QUESTION:

Q1. Question here
A) Option
B) Option
C) Option
D) Option
Correct Answer: A) correct option

---

Text:
{text}

IMPORTANT:
- Generate EXACTLY {number} questions
- Follow format strictly
- Do not add extra explanation outside format
"""
)

# function to generate MCQs using Groq
def generate_mcqs(data):

    formatted_prompt = quiz_prompt.format(
        text=data["text"],
        number=data["number"],
        subject=data["subject"],
        tone=data["tone"]
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content


# wrapper to mimic LangChain chain.invoke()
class GroqChain:

    def invoke(self, data):
        return generate_mcqs(data)


# this will be imported in StreamlitAPP.py
chain = GroqChain()