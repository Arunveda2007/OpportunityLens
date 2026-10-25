
import os
import json
import time
import pymupdf

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"


# Extract text from an uploaded PDF
def extract_pdf(pdf_bytes):

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:

        for page_number, page in enumerate(pdf):

            pages.append({
                "page": page_number + 1,
                "text": page.get_text()
            })

    return pages


# Generate a response using Gemini
def generate_gemini_response(prompt, json_mode=False):

    config = None

    if json_mode:
        config = {
            "response_mime_type": "application/json"
        }

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as error:

            message = str(error).upper()

            temporary_error = (
                "503" in message
                or "UNAVAILABLE" in message
                or "429" in message
                or "RESOURCE_EXHAUSTED" in message
            )

            if not temporary_error or attempt == 2:
                raise

            time.sleep(2 * (attempt + 1))


# Extract structured internship details
def extract_internship_details(pdf_text):

    prompt = f"""
    Analyze the following internship advertisement.

    Extract these fields:
    - organization
    - stipend
    - duration
    - mode
    - disciplines
    - eligibility
    - deadline

    Return valid JSON using exactly these keys:
    organization, stipend, duration, mode,
    disciplines, eligibility, deadline.

    If information is missing, write "Not specified".
    Do not invent information.

    Internship advertisement:
    {pdf_text}
    """

    response_text = generate_gemini_response(
        prompt,
        json_mode=True
    )

    try:
        details = json.loads(response_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini did not return valid JSON. "
            "Please try again."
        ) from error

    required_keys = [
        "organization",
        "stipend",
        "duration",
        "mode",
        "disciplines",
        "eligibility",
        "deadline"
    ]

    for key in required_keys:
        details.setdefault(key, "Not specified")

    return details
