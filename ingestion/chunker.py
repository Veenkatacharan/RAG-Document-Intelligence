import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_sections(text: str) -> list[str]:
    """
    Tries to split text into logical sections based on structural patterns.
    Looks for:
    - Lines that look like headings (short, no punctuation at end, followed by content)
    - Numbered list resets (No  Title patterns indicating a new table)
    - All caps or title case short lines

    Returns a list of section strings.
    """
    # split on double newlines first (paragraph breaks)
    raw_sections = re.split(r'\n{2,}', text)

    sections = []
    current_section = []

    for block in raw_sections:
        stripped = block.strip()
        if not stripped:
            continue

        # detect if this block looks like a section heading
        # headings are typically short (under 60 chars) and don't end in punctuation
        is_heading = (
            len(stripped) < 60 and
            not stripped.endswith(('.', ',', ';', ':')) and
            '\n' not in stripped and
            not stripped[0].isdigit()
        )

        if is_heading and current_section:
            # save what we have and start a new section
            sections.append('\n\n'.join(current_section))
            current_section = [stripped]
        else:
            current_section.append(stripped)

    # don't forget the last section
    if current_section:
        sections.append('\n\n'.join(current_section))

    return sections


def chunk_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[dict]:
    """
    Improved chunker that:
    1. First detects logical sections within each page
    2. Keeps sections together where possible
    3. Falls back to character splitting only if a section is too large

    Args:
        pages: output from load_all_pdfs()
        chunk_size: maximum characters per chunk
        chunk_overlap: overlap between chunks when splitting large sections
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ".", " "]
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:
        # first pass — detect sections
        sections = detect_sections(page["text"])

        for section in sections:
            if len(section.strip()) < 30:
                continue

            # if section fits within chunk_size, keep it whole
            if len(section) <= chunk_size:
                all_chunks.append({
                    "chunk_index": chunk_index,
                    "text": section.strip(),
                    "source": page["source"],
                    "page": page["page"]
                })
                chunk_index += 1

            else:
                # section is too large — split it with overlap
                splits = splitter.split_text(section)
                for split in splits:
                    if len(split.strip()) < 30:
                        continue
                    all_chunks.append({
                        "chunk_index": chunk_index,
                        "text": split.strip(),
                        "source": page["source"],
                        "page": page["page"]
                    })
                    chunk_index += 1

    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ingestion.loader import load_all_pdfs

    pages = load_all_pdfs("data")
    chunks = chunk_pages(pages)

    print(f"\nFirst 5 chunks:\n")
    for chunk in chunks[:5]:
        print(f"--- Chunk {chunk['chunk_index']} | {chunk['source']} p.{chunk['page']} ---")
        print(chunk["text"])
        print(f"Length: {len(chunk['text'])} characters\n")
