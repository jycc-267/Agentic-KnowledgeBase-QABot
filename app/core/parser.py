import re
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Chunking parameters: 500 chars with 50-char overlap to avoid cutting facts
# at boundaries. Separators prefer Markdown structure over splitting mid-word.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


def slugify(text: str) -> str:
    """Convert heading text to a URL-safe slug (e.g. 'Refund Timeline' -> 'refund-timeline')."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def load_markdown_sections(path: Path) -> list[Document]:
    """Parse a single Markdown file into section-level Documents.

    Each section is delimited by headings (# through ######).  The returned
    Documents carry metadata:
      - source:  e.g. "refund_policy.md"
      - heading: e.g. "Refund Timeline"
      - section_id: e.g. "refund_policy.md#refund-timeline"
    """
    text = path.read_text(encoding="utf-8")
    filename = path.name
    lines = text.split("\n")

    sections: list[Document] = []
    current_heading = filename.replace(".md", "").replace("_", " ").title()
    current_slug = slugify(current_heading)
    current_lines: list[str] = []

    def _flush():
        content = "\n".join(current_lines).strip()
        if not content:
            return
        section_id = f"{filename}#{current_slug}"
        sections.append(Document(
            page_content=content,
            metadata={
                "source": filename,
                "heading": current_heading,
                "section_id": section_id,
            },
        ))

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            _flush()
            current_heading = match.group(2)
            current_slug = slugify(current_heading)
            current_lines = [line]
        else:
            current_lines.append(line)

    _flush()
    return sections


def split_sections_to_chunks(sections: list[Document]) -> list[Document]:
    """Split section Documents into smaller chunks, preserving parent metadata.

    Every chunk inherits its parent section's metadata plus a ``parent_section_id``
    field that enables resolution back to the full section during retrieval.
    """
    chunks: list[Document] = []
    for section in sections:
        section_chunks = splitter.split_documents([section])
        for chunk in section_chunks:
            # Carry forward the parent section_id so chunk hits can be
            # resolved back to the full section for LLM context.
            chunk.metadata = {
                **chunk.metadata,
                "parent_section_id": section.metadata["section_id"],
            }
            chunks.append(chunk)
    return chunks
