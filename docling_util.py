from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode

def process_pdf(pdf_path: str) -> str:
    """
    Convert PDF to markdown into output_md/<stem>.md and return md path.
    """
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=False,
        generate_picture_images=True,
        images_scale=2
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(source=pdf_path)

    out_dir = Path("output_md")
    out_dir.mkdir(exist_ok=True)

    md_path = out_dir / f"{Path(pdf_path).stem}.md"
    result.document.save_as_markdown(
        md_path,
        image_mode=ImageRefMode.REFERENCED,
        include_annotations=True
    )
    return str(md_path).replace("\\", "/")
