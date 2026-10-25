
import streamlit as st

from extractor import extract_pdf, extract_internship_details
from rag import answer_question
from eligibility import check_eligibility


st.set_page_config(
    page_title="OpportunityLens",
    page_icon="🎯",
    layout="centered"
)

st.title("OpportunityLens 🎯")

st.write(
    "Analyze internship PDFs, ask questions, "
    "and check your eligibility."
)


# Upload internship PDF
pdf = st.file_uploader(
    "Upload internship PDF",
    type=["pdf"]
)


if pdf:

    pdf_bytes = pdf.getvalue()

    # Analyze the internship
    if st.button("Analyze Internship"):

        with st.spinner(
            "Analyzing internship..."
        ):

            try:

                pages = extract_pdf(pdf_bytes)

                internship_text = "\n".join(
                    page["text"]
                    for page in pages
                )

                if not internship_text.strip():

                    st.warning(
                        "No readable text was found in the PDF. "
                        "It may be a scanned document."
                    )

                else:

                    details = extract_internship_details(
                        internship_text
                    )

                    st.session_state["internship_details"] = details
                    st.session_state["analyzed_pdf_name"] = pdf.name

                    st.success(
                        "Internship analysis completed!"
                    )

            except Exception as error:

                st.error(
                    f"Internship analysis error: {error}"
                )

    # Show extracted details only for the analyzed PDF
    if (
        "internship_details" in st.session_state
        and st.session_state.get("analyzed_pdf_name") == pdf.name
    ):

        st.subheader("Internship Details")

        st.json(
            st.session_state["internship_details"]
        )

    st.divider()

    # Question-answering section
    st.subheader("Ask a Question")

    question = st.text_input(
        "Ask anything about this internship",
        key="internship_question"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            with st.spinner(
                "Finding the answer in your PDF..."
            ):

                try:

                    answer = answer_question(
                        pdf_bytes,
                        question
                    )

                    st.subheader("Answer")
                    st.write(answer)

                except Exception as error:

                    st.error(
                        f"Question-answering error: {error}"
                    )

    st.divider()

    # Eligibility checker
    st.header("🎯 Check Your Eligibility")

    st.write(
        "Compare your profile with the internship requirements."
    )

    with st.form("eligibility_form"):

        student_name = st.text_input(
            "Your name"
        )

        degree = st.text_input(
            "Degree and branch",
            placeholder="B.Tech CSE"
        )

        graduation_year = st.text_input(
            "Expected graduation year",
            placeholder="2029"
        )

        cgpa = st.text_input(
            "CGPA",
            placeholder="9.5 / 10"
        )

        skills = st.text_input(
            "Technical skills",
            placeholder="Python, Machine Learning, SQL"
        )

        experience = st.text_area(
            "Projects or experience",
            placeholder="Describe your projects and experience"
        )

        check_button = st.form_submit_button(
            "Check My Eligibility"
        )

    if check_button:

        if not degree.strip():

            st.warning(
                "Enter your degree and branch first."
            )

        elif (
            "internship_details" not in st.session_state
            or st.session_state.get("analyzed_pdf_name") != pdf.name
        ):

            st.warning(
                "Click Analyze Internship first for this PDF."
            )

        else:

            student_profile = {
                "Name": student_name,
                "Degree": degree,
                "Graduation Year": graduation_year,
                "CGPA": cgpa,
                "Skills": skills,
                "Projects and Experience": experience
            }

            with st.spinner(
                "Checking your eligibility..."
            ):

                try:

                    result = check_eligibility(
                        st.session_state["internship_details"],
                        student_profile
                    )

                    st.subheader(
                        "Your Eligibility Report"
                    )

                    st.markdown(result)

                except Exception as error:

                    st.error(
                        f"Eligibility error: {error}"
                    )

else:

    st.info(
        "Upload an internship PDF to get started."
    )
