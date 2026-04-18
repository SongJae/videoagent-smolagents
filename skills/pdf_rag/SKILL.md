---
name: pdf_rag
description: PDF RAG (Retrieval-Augmented Generation) skill for searching reference documents. Use when needing specifications, tactical procedures, equipment details, or any information from uploaded PDF manuals and guides.
---

# PDF RAG Search Skill

## Overview

This skill enables semantic search across uploaded PDF documents including manuals, field guides, intelligence reports, and tactical references. Use it to augment video analysis with contextual information.

## When to Use

- Looking up equipment specifications (range, speed, capabilities)
- Finding tactical procedures and doctrine
- Cross-referencing video observations with documentation
- Answering "why" or "how" questions beyond visual analysis
- Getting identification guides for detected objects

## Usage

```python
result = pdf_rag_skill(query="tank operational range capabilities")
```

### Parameters

- `query` (required): Semantic search query describing needed information
- `pdf_sources` (optional): List of specific PDF filenames to search
- `top_k` (optional): Number of results to return (default: 5)

## Result Structure

Returns a dictionary containing:
- `status`: "success", "no_results", or "error"
- `result`: Contains search results with:
  - `results`: List of matching passages with:
    - `rank`: Position in relevance ranking
    - `text`: Extracted text chunk
    - `relevance_score`: Similarity score (0.0-1.0)
    - `source`: PDF filename
  - `sources_used`: List of PDFs that had matches
  - `num_results`: Number of results found

## Relevance Score Guidelines

- `> 0.7`: Highly relevant, strong match
- `0.5 - 0.7`: Moderately relevant
- `< 0.5`: Weakly relevant, may be off-topic

## Best Practices

### Good Query Construction

```python
# GOOD - specific, technical queries
pdf_rag_skill("T-72 tank identification features")
pdf_rag_skill("convoy spacing tactical formation")
pdf_rag_skill("tank thermal signature detection range")
```

```python
# BAD - too vague
pdf_rag_skill("information about tanks")  # Too general
pdf_rag_skill("how many tanks in video")  # Use videodb_query instead
```

### Combining with Video Analysis

```python
# Step 1: Detect objects in video
tanks = videodb_query_skill("object_search", object_type="tank")

# Step 2: Get reference information
specs = pdf_rag_skill("tank capabilities and weaknesses")

# Step 3: Synthesize in your response
```

## Error Handling

Always check status:
```python
result = pdf_rag_skill("tank specifications")

if result["status"] == "success":
    # Use results
    for item in result["result"]["results"]:
        print(f"[{item['source']}] {item['text'][:200]}...")
elif result["status"] == "no_results":
    # Normal - not all queries match content
    print("No relevant information found in PDFs")
elif result["status"] == "error":
    # System error
    print(f"Error: {result['message']}")
```

## Source Attribution

Always cite sources when using PDF information:
```python
result = pdf_rag_skill("convoy tactics")
if result["status"] == "success":
    for r in result["result"]["results"]:
        # Include source in your response
        print(f"According to {r['source']}: {r['text']}")
```
