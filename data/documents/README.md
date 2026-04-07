# Sample Documents Directory

Place your financial documents here for ingestion.

## Supported Formats
- PDF (`.pdf`) - 10-K reports, earnings reports, whitepapers
- Text (`.txt`) - Plain text documents
- CSV (`.csv`) - Tabular financial data
- Word (`.docx`) - Microsoft Word documents
- Markdown (`.md`) - Markdown files

## Example Documents to Include
1. **10-K Annual Reports** - SEC filings with comprehensive business info
2. **10-Q Quarterly Reports** - Quarterly financial updates
3. **Earnings Transcripts** - Quarterly earnings call transcripts
4. **Investor Presentations** - Corporate presentation decks
5. **Financial Whitepapers** - Industry analysis documents

## Ingestion
After adding documents, run:

```bash
# From project root
python scripts/ingest_documents.py --input ./data/documents

# Or using make
make ingest
```

## Tips
- Use descriptive filenames for better metadata extraction
- Include year/quarter in filenames (e.g., `AAPL_10K_2023.pdf`)
- Organize by company or document type in subdirectories
