import fitz


def extract_pdf_text(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page in document:

        text = page.get_text("text")

        if text.strip():
            pages.append(text)

    document.close()

    return "\n".join(pages)