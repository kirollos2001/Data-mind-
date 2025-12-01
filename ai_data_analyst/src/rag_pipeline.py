import google.generativeai as genai
from pypdf import PdfReader
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# -----------------------------
# 1) Configure Google Gemini API
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)
model_name = "gemini-2.5-pro"

try:
    model = genai.GenerativeModel(model_name)
except Exception as e:
    print(f"Error initializing model {model_name}: {e}")
    print("Falling back to gemini-1.5-pro")
    model_name = "gemini-1.5-pro"
    model = genai.GenerativeModel(model_name)


# ----------------------------------
# 2) Load PDF and extract all text
# ----------------------------------
def load_pdf(pdf_path):
    print(f"📄 Loading PDF from: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        return ""
    
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in tqdm(reader.pages, desc="Reading pages", unit="page"):
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    print(f"✅ PDF loaded. Total characters: {len(full_text):,}")
    return full_text

# ---------------------------------------------------------
# 3) Helper function: Count tokens using Gemini tokenizer
# ---------------------------------------------------------
# Caching to avoid redundant calls
_token_count_cache = {}

def token_count(text: str) -> int:
    """Return number of tokens according to Gemini tokenizer (with caching)."""
    # Use hash as cache key (text might be too large)
    cache_key = hash(text)
    if cache_key in _token_count_cache:
        return _token_count_cache[cache_key]
    
    try:
        resp = model.count_tokens(text)
        count = resp.total_tokens
        _token_count_cache[cache_key] = count
        return count
    except Exception as e:
        print(f"⚠️ Error counting tokens: {e}")
        return 0

# ---------------------------------------------------------
# 4) OPTIMIZED: Split text by token length
#    Strategy: Use character-based estimation + validation
# ---------------------------------------------------------
def split_by_tokens_optimized(text, max_tokens=500, overlap_tokens=50):
    """
    Optimized splitting with minimal API calls.
    
    Strategy:
    1. Estimate characters per token (~4 chars/token for English)
    2. Split by estimated character count
    3. Validate only the final chunk boundaries
    4. Adjust if needed with binary search (but much less frequently)
    """
    
    # Constants
    CHARS_PER_TOKEN = 4.0  # Conservative estimate for English
    
    words = text.split()
    chunks = []
    total_words = len(words)
    
    print(f"🔪 Starting optimized split for {total_words:,} words...")
    
    # Convert to character positions for faster processing
    # Build a word->char position map
    word_positions = []
    current_pos = 0
    for word in words:
        word_positions.append(current_pos)
        current_pos += len(word) + 1  # +1 for space
    
    start_idx = 0
    api_call_count = 0
    
    with tqdm(total=total_words, desc="Splitting text", unit="words") as pbar:
        while start_idx < total_words:
            # 1. Estimate end index based on characters
            start_char = word_positions[start_idx]
            target_chars = int(max_tokens * CHARS_PER_TOKEN)
            target_char_pos = start_char + target_chars
            
            # Find word index closest to target_char_pos
            end_idx = start_idx
            for i in range(start_idx, total_words):
                if word_positions[i] >= target_char_pos:
                    end_idx = i
                    break
            else:
                end_idx = total_words
            
            # Ensure we don't go past the end
            end_idx = min(end_idx, total_words)
            
            # 2. Verify with actual token count (single API call)
            chunk_text = " ".join(words[start_idx:end_idx])
            count = token_count(chunk_text)
            api_call_count += 1
            
            # 3. Adjust if needed (binary search, but should be rare)
            if count > max_tokens:
                # Shrink with binary search
                low = start_idx + 1
                high = end_idx
                
                while low < high:
                    mid = (low + high) // 2
                    test_text = " ".join(words[start_idx:mid])
                    test_count = token_count(test_text)
                    api_call_count += 1
                    
                    if test_count <= max_tokens:
                        low = mid + 1
                    else:
                        high = mid
                
                end_idx = low - 1
                chunk_text = " ".join(words[start_idx:end_idx])
            
            elif count < max_tokens * 0.85 and end_idx < total_words:
                # Try to expand (only if significantly under-utilized)
                # Binary search upward
                low = end_idx
                high = min(end_idx + int((max_tokens - count) / CHARS_PER_TOKEN * 0.25), total_words)
                
                best_idx = end_idx
                while low < high:
                    mid = (low + high + 1) // 2
                    test_text = " ".join(words[start_idx:mid])
                    test_count = token_count(test_text)
                    api_call_count += 1
                    
                    if test_count <= max_tokens:
                        best_idx = mid
                        low = mid
                    else:
                        high = mid - 1
                
                end_idx = best_idx
                chunk_text = " ".join(words[start_idx:end_idx])
            
            # Add chunk
            chunks.append(chunk_text)
            pbar.update(end_idx - start_idx)
            
            # Prepare next iteration with overlap
            if end_idx >= total_words:
                break
            
            # Calculate overlap in words (estimate)
            overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
            overlap_start = end_idx
            
            # Find word index for overlap
            target_overlap_char = word_positions[end_idx] - overlap_chars
            for i in range(end_idx - 1, start_idx, -1):
                if word_positions[i] <= target_overlap_char:
                    overlap_start = i
                    break
            
            start_idx = max(overlap_start, start_idx + 1)
    
    print(f"✅ Split complete: {len(chunks)} chunks with {api_call_count} API calls")
    return chunks


# -----------------------------
# 5) OPTIMIZED: Create Embeddings
# -----------------------------
def create_embeddings_batch(chunks, batch_size=100):
    """
    Generate embeddings for all chunks.
    
    Note: Gemini API embed_content processes one content at a time.
    """
    print(f"🎯 Generating embeddings for {len(chunks)} chunks...")
    
    all_embeddings = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks", unit="chunk")):
        try:
            # Embed each chunk individually
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            
            # Extract embedding
            if isinstance(result, dict) and 'embedding' in result:
                all_embeddings.append(result['embedding'])
            else:
                print(f"⚠️ Unexpected result format for chunk {i}")
                
        except Exception as e:
            print(f"⚠️ Error embedding chunk {i}: {e}")
            # Continue with other chunks
        
        # Small delay to avoid rate limiting
        if i < len(chunks) - 1:  # Don't sleep after last chunk
            time.sleep(0.1)
    
    if len(all_embeddings) == 0:
        print("❌ No embeddings generated.")
        return np.array([])
    
    embeddings_array = np.array(all_embeddings)
    print(f"✅ Generated {len(all_embeddings)} embeddings with shape {embeddings_array.shape}")
    return embeddings_array


# -----------------------------
# 6) Store in FAISS
# -----------------------------
def store_in_faiss(embeddings):
    if len(embeddings) == 0:
        print("❌ No embeddings to store.")
        return None
        
    dimension = embeddings.shape[1]
    print(f"🗄️  Creating FAISS index with dimension {dimension}")
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"✅ Stored {index.ntotal} vectors in FAISS index.")
    return index


# -----------------------------
# 7) Save Index to Disk (Optional)
# -----------------------------
def save_index(index, chunks, output_dir="rag_output"):
    """Save FAISS index and chunks for later use."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"💾 FAISS index saved to: {index_path}")
    
    # Save chunks as text
    chunks_path = os.path.join(output_dir, "chunks.txt")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i} ---\n")
            f.write(chunk)
            f.write("\n\n")
    print(f"💾 Chunks saved to: {chunks_path}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    import sys
    
    # Path from user request (or command line argument)
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = r"C:\Users\kirollos\Desktop\2003_H1_Reports.pdf"
    
    print("=" * 60)
    print("🚀 RAG Pipeline - Optimized Version")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Load
    text = load_pdf(pdf_path)
    
    if text:
        # 2. Split (Optimized)
        chunks = split_by_tokens_optimized(text, max_tokens=500, overlap_tokens=50)
        print(f"\n📊 Generated {len(chunks)} chunks")
        
        if chunks:
            print(f"📝 First chunk preview ({len(chunks[0])} chars):")
            print(chunks[0][:200] + "...\n")
            
            # 3. Embed (Batch)
            embeddings = create_embeddings_batch(chunks, batch_size=50)
            
            # 4. Index
            if len(embeddings) > 0:
                index = store_in_faiss(embeddings)
                
                # 5. Save (Optional)
                save_index(index, chunks)
                
                elapsed = time.time() - start_time
                print("\n" + "=" * 60)
                print(f"✅ Pipeline completed in {elapsed:.2f} seconds")
                print("=" * 60)
            else:
                print("❌ Failed to generate embeddings.")
        else:
            print("❌ No chunks generated.")
    else:
        print("❌ Failed to load PDF text.")
