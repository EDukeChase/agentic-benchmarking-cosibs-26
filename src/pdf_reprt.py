# Use for generating a PDF report of the benchmarking results


# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
# )
# from reportlab.lib.styles import getSampleStyleSheet

# from src.schemas import BenchmarkReport


# def save_pdf(report: BenchmarkReport, filename="benchmark_report.pdf"):
#     styles = getSampleStyleSheet()

#     doc = SimpleDocTemplate(filename)
#     elements = []

#     elements.append(Paragraph("Benchmark Report", styles["Title"]))
#     elements.append(Spacer(1, 20))

#     elements.append(Paragraph("<b>Overall Summary</b>", styles["Heading1"]))
#     elements.append(Paragraph(report.summary, styles["BodyText"]))
#     elements.append(Spacer(1, 12))

#     elements.append(Paragraph("<b>Recommendations</b>", styles["Heading1"]))
#     elements.append(Paragraph(report.recommendations, styles["BodyText"]))
#     elements.append(Spacer(1, 20))

#     elements.append(Paragraph("<b>Model Results</b>", styles["Heading1"]))
#     elements.append(Spacer(1, 12))

#     for model in report.entries:
#         elements.append(
#             Paragraph(f"<b>{model.model_name}</b>", styles["Heading2"])
#         )

#         elements.append(
#             Paragraph(f"<b>Rationale:</b> {model.rationale}", styles["BodyText"])
#         )

#         elements.append(
#             Paragraph(f"<b>Documentation:</b> {model.documentation}", styles["BodyText"])
#         )

#         elements.append(
#             Paragraph(
#                 f"""
#                 Accuracy: {model.accuracy}<br/>
#                 F1: {model.f1}<br/>
#                 AUROC: {model.auroc}<br/>
#                 Status: {model.status}
#                 """,
#                 styles["BodyText"],
#             )
#         )

#         elements.append(Spacer(1, 15))

#     doc.build(elements)