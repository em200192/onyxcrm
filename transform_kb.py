# transform_kb.py
import streamlit as st
import json
import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

# --- CONFIGURATION ---
load_dotenv()
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
RAW_KB_PATH = CACHE_DIR / "gl_guide_kb.json"  # Input file
FINAL_KB_PATH = CACHE_DIR / "gl_guide_final.json"  # Output file


# --- LLM SETUP (Copied from your main app) ---
@st.cache_resource
def get_llm_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found in .env file")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.0)


def _json_from_model_text(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


@st.cache_resource
def create_enrichment_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""You are an expert technical writer creating a knowledge base for an ERP system.
Based on the following topic title and content, your task is to:
1. Write a concise, one-sentence summary of the topic's purpose.
2. Generate a list of 3 to 5 different, realistic questions a user might ask to find this information. The questions should be in Arabic.

Topic Title: "{title}"
Topic Content Snippet: "{content}"

Respond with a valid JSON object only, with the keys "summary" and "faqs" (a list of strings).
Example Response:
{{"summary": "This section explains how to define and code new bank accounts.", "faqs": ["كيف اضيف بنك جديد؟", "اين اجد شاشة تعريف البنوك؟", "ما هي خطوات ترميز حساب بنكي؟"]}}

JSON Response:""",
        input_variables=["title", "content"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)


# --- MAIN TRANSFORMATION LOGIC ---
def transform_knowledge_base():
    if not RAW_KB_PATH.exists():
        print(f"Error: Raw knowledge base file not found at '{RAW_KB_PATH}'")
        print("Please run the 'Build GL User Guide' process in the Streamlit app first.")
        return

    with open(RAW_KB_PATH, "r", encoding="utf-8") as f:
        raw_sections = json.load(f)

    print(f"Loaded {len(raw_sections)} raw sections. Starting AI enrichment process...")

    enrichment_chain = create_enrichment_chain()
    enriched_sections = []

    for i, section in enumerate(raw_sections):
        title = section.get("title")
        body = section.get("body", "")
        print(f"  -> Enriching topic {i + 1}/{len(raw_sections)}: {title}")

        try:
            # Use a snippet of the body for context to keep the prompt efficient
            content_snippet = body[:800]
            response_text = enrichment_chain.invoke({"title": title, "content": content_snippet})["text"]
            enriched_data = _json_from_model_text(response_text)

            # Add the new, smart data to the section
            section["summary"] = enriched_data.get("summary", "")
            section["faqs"] = enriched_data.get("faqs", [])

            enriched_sections.append(section)
        except Exception as e:
            print(f"    -> WARNING: Could not enrich '{title}'. Using basic data. Error: {e}")
            # Even if enrichment fails, keep the original section
            section["summary"] = ""
            section["faqs"] = []
            enriched_sections.append(section)

    with open(FINAL_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_sections, f, ensure_ascii=False, indent=2)

    print(f"\nTransformation complete. Enriched knowledge base saved to '{FINAL_KB_PATH}'")


if __name__ == "__main__":
    transform_knowledge_base()