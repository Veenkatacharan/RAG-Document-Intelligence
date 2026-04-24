import fitz  # this is PyMuPDF
import os

def load_pdf(file_path: str) -> list[dict]:
    """
    Takes a path to a PDF file.
    Returns a list of dicts, one per page:
    { "page": 1, "text": "...", "source": "filename.pdf" }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at: {file_path}")

    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # skip pages that are blank or nearly blank
        if len(text.strip()) < 50:
            continue

        pages.append({
            "page": page_num + 1,
            "text": text.strip(),
            "source": os.path.basename(file_path)
        })

    doc.close()
    print(f"Loaded {len(pages)} pages from {os.path.basename(file_path)}")
    return pages


def load_all_pdfs(folder_path: str) -> list[dict]:
    """
    Loads every PDF in a folder.
    Returns all pages from all documents combined.
    """
    all_pages = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {folder_path}")
        return []

    for filename in pdf_files:
        full_path = os.path.join(folder_path, filename)
        pages = load_pdf(full_path)
        all_pages.extend(pages)

    print(f"\nTotal: {len(all_pages)} pages loaded from {len(pdf_files)} PDF(s)")
    return all_pages


if __name__ == "__main__":
    # quick test — drop any PDF into your data/ folder and run this
    pages = load_all_pdfs("data")

    # print the first 3 pages to verify it's working
    for page in pages[:3]:
        print(f"\n--- {page['source']} | Page {page['page']} ---")
        print(page['text'][:300])  # first 300 characters
        print("...")
