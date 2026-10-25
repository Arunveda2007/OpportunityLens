
from extractor import generate_gemini_response


def check_eligibility(internship_details, student_profile):

    prompt = f"""
    You are OpportunityLens, an internship eligibility checker.

    Compare the internship requirements with the student's profile.

    INTERNSHIP DETAILS:
    {internship_details}

    STUDENT PROFILE:
    {student_profile}

    Return your report in this format:

    Eligibility: Eligible, Not Eligible, or Uncertain

    Matching Requirements:
    - List the requirements the student meets.

    Missing Requirements:
    - List the requirements the student does not meet.

    Unknown Information:
    - List information that cannot be verified.

    Explanation:
    - Briefly explain your conclusion.

    Rules:
    1. Use only the information provided.
    2. Never assume missing information.
    3. If a mandatory requirement cannot be verified,
       use Uncertain rather than guessing.
    4. Do not claim the student is eligible if a mandatory
       requirement is clearly not satisfied.
    5. Compare the graduation year and CGPA against the
       stated requirements whenever applicable.
    6. Distinguish mandatory requirements from preferred skills.
    7. Do not assume that missing project experience means
       the student lacks a particular skill.
    """

    return generate_gemini_response(prompt)
