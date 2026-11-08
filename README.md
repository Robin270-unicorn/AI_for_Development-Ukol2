# Ask Your PDFs

A Python RAG command-line app for Homework 2 in AI for Software Development. It builds a local index of text-based PDFs and answers questions with citations in the form `[filename.pdf, p. N]`.

## Setup on Windows

Install Python 3.12 for Windows x64 and Ollama. Start Ollama so that its local API is available on port 11434. Run these commands from this repository in PowerShell. [Python downloads](https://www.python.org/downloads/) · [Ollama downloads](https://ollama.com/download)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:1.5b
.\.venv\Scripts\python.exe ingest.py pdfs
.\.venv\Scripts\python.exe ask.py "How cool should the build plate be before I remove a print?"
```

On macOS or Linux, use `python3.12 -m venv .venv` and `./.venv/bin/python` in place of the Windows Python paths. Use a Python build whose SQLite module can load extensions.

The first ingestion downloads the embedding model from Hugging Face. After both models are downloaded, indexing and querying run locally. There is no API key. If the local server is unavailable, start Ollama or run `ollama serve` in a separate terminal. If the model is missing, repeat the `ollama pull` command.

## Example questions

The `pdfs/` folder contains three original two-page fictional manuals. Answer wording may vary. Compare the facts and page citations.

| Question | Expected fact and citation |
| --- | --- |
| How cool should the build plate be before I remove a print? | 35 degrees Celsius. `[print-lab-manual.pdf, p. 2]` |
| How long is a standard bike share ride? | 45 minutes. `[bike-share-guide.pdf, p. 1]` |
| What should I do if all the bike docks are full? | Select **Station full** in the app. `[bike-share-guide.pdf, p. 2]` |
| Which DOI should I use when citing a dataset? | Use the DOI for the version you used. `[research-archive-manual.pdf, p. 2]` |
| What is the phone number for the print lab supervisor? | `I don't know based on the provided documents.` |
| What is the orbital period of Neptune? | `I don't know based on the provided documents.` |

## How it works

1. **Ingest.** `ingest.py` uses pypdf to extract each page. It splits the page into chunks of up to 200 model tokens with 40 tokens of overlap. Each chunk stays on one page and carries the original filename and the 1-based PDF page number.
2. **Index.** Sentence Transformers encodes each chunk using `all-MiniLM-L6-v2`. A SQLite database stores the vectors and page metadata. Re-running ingestion replaces the previous index in one transaction.
3. **Retrieve.** `ask.py` embeds the question with the same model. The sqlite-vec extension ranks chunks by cosine similarity and returns the top four by default.
4. **Generate.** The local language model receives the question and the relevant chunks. It returns an answer and quotes from the chunks. The CLI cites the answer when it appears on one retrieved page and uses a verified supporting quote to give that answer context when available. For paraphrases, it matches the supporting quotes to their pages and checks that numbers in the answer appear in those quotes.

Run `ingest.py <folder>` again after changing the PDFs. It reads files directly inside that folder. Use `--db PATH` on both scripts to keep separate indexes. Use `ask.py --help` to see the top-k and model options.

The sample corpus and embedding model are English. Image-only pages require OCR before ingestion. Exact cosine search scans the stored vectors, which suits a small document collection. The app abstains when no retrieved chunk reaches a cosine similarity of 0.30 or it cannot trace the answer or supporting quotes to specific pages. The CLI verifies where the text appears. The model decides whether the text supports the answer.

## References

- [Sentence Transformers embedding guide](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)
- [sqlite-vec nearest-neighbor queries](https://alexgarcia.xyz/sqlite-vec/features/knn.html)
- [Ollama chat completions API](https://docs.ollama.com/api/openai-compatibility)
