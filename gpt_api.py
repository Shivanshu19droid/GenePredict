import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Set your Gemini API key securely
genai.configure(api_key=os.getenv("YOUR_GEMINI_API_KEY"))

# Initialize Gemini model (text-only)
model = genai.GenerativeModel("models/gemini-1.5-flash")

def fetch_disease_info(disease_name):
    try:
        prompt = f"Explain the disease '{disease_name}' in simple, non-technical terms. Keep it short and informative."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error fetching disease information: {e}"



    



