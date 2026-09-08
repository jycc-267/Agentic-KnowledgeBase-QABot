import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Chunking parameters: 500 chars with 50-char overlap to avoid cutting facts
# at boundaries. Separators prefer Markdown structure over splitting mid-word.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


def slugify(text: str) -> str:
    """Convert heading text to a URL-safe slug (e.g. 'Refund Timeline' -> 'refund-timeline').
    
    Why: Creating stable, readable anchor IDs allows us to generate deep links directly
    to the relevant heading in the frontend, improving the user experience.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def load_markdown_sections(path: Path) -> list[Document]:
    """Parse a single Markdown file into section-level Documents.

    What: Reads a markdown file line by line and splits it at every heading level (# to ######).
    Each chunk of text under a heading becomes a distinct `Document` with metadata
    tracking its source file and generated section ID.

    Why: BM25 (keyword search) and the LLM context window both operate better on 
    complete, coherent sections rather than arbitrarily sliced text. This gives them
    the full context of a single topic.
    """
    # Step 1: Read the entire markdown file content
    text = path.read_text(encoding="utf-8")
    filename = path.name
    
    # Step 2: Split the content into individual lines for line-by-line processing
    lines = text.split("\n")

    sections: list[Document] = []
    
    # Step 3: Initialize heading stack with the document title
    doc_title = filename.replace(".md", "").replace("_", " ").title()
    
    # heading_stack is a stack that changes over time as the code parses through the markdown file line-by-line.
    # It starts with the document title as the root (level 1) and pushes/pops as it encounters
    # headings of different levels, maintaining the current heading hierarchy.
    heading_stack: list[tuple[int, str]] = [(1, doc_title)]
    
    # Step 4: Generate a URL-safe slug for the initial section
    current_slug = slugify(doc_title)
    current_lines: list[str] = []

    # Step 5: Define a helper function to save the currently accumulated lines into a LangChain Document
    def _flush():
        """
        What: This function encapsulates the logic for converting the `current_lines` buffer into a structured
        LangChain Document with complete metadata (including the hierarchical `heading_path` and `section_id`).
        It acts as a "section factory" called every time a new heading is encountered or at the end of the file.
        """
        content = "\n".join(current_lines).strip()
        if not content:
            return
        section_id = f"{filename}#{current_slug}"
        
        # Current path is just the text components from the stack
        heading_path = [h for _, h in heading_stack]
        
        sections.append(Document(
            page_content=content,
            metadata={
                "source": filename,
                "heading": heading_path[-1] if heading_path else "",
                "heading_path": heading_path,
                "section_id": section_id,
            },
        ))

    # Step 6: Iterate through each line to detect headings and build sections
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            # Step 7a: If a heading is found, flush the previous section and start a new one
            _flush()
            
            # Step 7b: Determine the heading level (number of #) and text
            level = len(match.group(1))
            heading_text = match.group(2)
            
            # Step 7c: Pop the stack so the new heading is placed at the correct depth relative to its parents
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
                
            # Step 7d: Push the current heading onto the stack to form the new hierarchical path
            heading_stack.append((level, heading_text))
            
            # Step 7e: Update the slug and reset the line accumulator for the new section
            current_slug = slugify(heading_text)
            current_lines = [line]
        else:
            # Step 8: If it's not a heading, simply append the text line to the current section
            current_lines.append(line)

    # Step 9: After the loop finishes, flush any remaining text as the final section
    _flush()
    return sections


def split_sections_to_chunks(sections: list[Document]) -> list[Document]:
    """Split section Documents into smaller chunks, preserving parent metadata.

    What: Takes large section Documents and breaks them down using the `splitter` 
    (RecursiveCharacterTextSplitter) into ~500 character chunks. 
    It explicitly copies the `parent_section_id` into each chunk's metadata.

    Why: Dense embeddings (FAISS) perform much better on shorter, focused text snippets
    (~500 chars). By preserving the `parent_section_id`, we can retrieve the precise 
    small chunk, but then inflate the context back to the full section before passing 
    it to the LLM (preventing context fragmentation).
    """
    chunks: list[Document] = []
    
    # Step 1: Iterate over each full section Document
    for section in sections:
        
        # Step 2: Break the section into smaller chunks using LangChain's splitter
        section_chunks = splitter.split_documents([section])
        
        # Step 3: Iterate over the newly generated chunks for this section
        for chunk in section_chunks:
            
            # Step 4: Inject the parent section ID into each chunk's metadata
            # This allows chunk hits to be resolved back to the full section for LLM context.
            chunk.metadata = {
                **chunk.metadata,
                "parent_section_id": section.metadata["section_id"],
            }
            
            # Step 5: Append the modified chunk to the final list
            chunks.append(chunk)
            
    return chunks
