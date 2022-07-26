# Script to merge all PDFs in a directory
# Usage: python merge_pdfs.py <input_dir> <output_file>
# or import the funktion merge_pdfs
# You need to install PyPDF2 if it's not already installed:
# pip install PyPDF2


from PyPDF2 import PdfFileMerger
from pathlib import Path
import sys

def merge_pdfs(input_dir, output_file):
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

    if not isinstance(output_file, Path):
        output_file = Path(output_file)

    assert input_dir.is_dir(), "Input directory is not a directory"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = [str(file) for file in input_dir.glob("*.pdf") if file.is_file()]
    
    assert len(pdf_files) > 0, f"There are no PDF files in '{str(input_dir)}' to merge."
    
    pdf_files = sorted(pdf_files)

    merger = PdfFileMerger()

    for pdf in pdf_files:
        merger.append(pdf)
    
    with open(output_file, "wb") as fout:
        merger.write(fout)

    merger.close()

    print("PDFs merged successfully")


if __name__ == "__main__":
    merge_pdfs(input_dir=sys.argv[1], output_file=sys.argv[2])