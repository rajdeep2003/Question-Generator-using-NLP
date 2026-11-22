from reportlab.lib.pagesizes import A4

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet


def export_question_paper(results, output_path):

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AUTOMATIC QUESTION PAPER",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            f"Total Questions: {len(results)}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    for i, item in enumerate(results, 1):

        content.append(
            Paragraph(
                f"<b>Q{i}. {item['question']}</b>",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 8))

        content.append(
            Paragraph(
                f"Bloom's Level: {item['bloom']} | "
                f"Difficulty: {item['difficulty']}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 15))

    content.append(PageBreak())

    content.append(
        Paragraph("ANSWER KEY", styles["Title"])
    )

    content.append(Spacer(1, 20))

    for i, item in enumerate(results, 1):

        content.append(
            Paragraph(
                f"<b>Q{i}.</b> {item['answer']}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 12))

    document.build(content)

    print("PDF exported:", output_path)