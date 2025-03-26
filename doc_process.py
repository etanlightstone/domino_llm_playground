# document_processor.py

def read_file_contents(file_path):
    """Reads and returns the contents of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_document(file_path, chunk_size, overlap_size):
    """
    Processes a document by reading its content and splitting it into chunks.
    
    Parameters:
    - file_path: Path to the document file.
    - chunk_size: Size of each chunk.
    - overlap_size: Overlap size between consecutive chunks.
    
    Returns:
    - A list of text chunks.
    """
    if chunk_size <= 0 or overlap_size < 0:
        raise ValueError("Chunk size must be positive and overlap size must be non-negative.")
    
    content = read_file_contents(file_path)
    length = len(content)
    chunks = []
    
    for i in range(0, length, chunk_size - overlap_size):
        end = min(i + chunk_size, length)
        chunks.append(content[i:end])
    
    return chunks