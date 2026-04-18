---
name: final_response
description: Final response skill for generating structured battlefield analysis reports. Use to format your final answer with clear causal relationships, evidence attribution, and professional report structure. Always use this for final answers to maintain consistency.
---

# Final Response Generation Skill

## Overview

This skill generates professional, structured reports for battlefield analysis. Use it to format your final answer with clear organization, evidence attribution, and actionable insights.

## When to Use

**Always use this skill when providing your final answer** to ensure:
- Consistent report format across all analyses
- Clear causal relationships between evidence and conclusions
- Proper attribution of sources (video, PDF, tactical map)
- Professional presentation of findings

## Report Structure

The generated report includes these sections (as applicable):

1. **Title** - Descriptive title for the analysis
2. **Executive Summary** - 2-3 sentence overview
3. **Methodology** - How the analysis was conducted
4. **Key Findings** - Numbered list of discoveries
5. **Video Evidence** - Specific video segment references
6. **Reference Documentation** - PDF citations
7. **Tactical Assessment** - Overall situation analysis
8. **Recommendations** - Actionable next steps

## Usage

```python
report = final_response_skill(
    title="Reconnaissance Analysis Report",
    summary="Analysis of drone footage reveals significant enemy armor activity...",
    findings=[
        "3 tanks detected moving eastward in sector Alpha",
        "Convoy formation suggests defensive posture",
        {"text": "Infantry support observed", "confidence": "high", "source": "video"}
    ],
    video_evidence=[
        {
            "video_id": "m-abc123",
            "segment_id": 2,
            "time_range": "00:00:30 - 00:01:00",
            "description": "Tank formation in open terrain",
            "objects": ["tank", "tank", "truck"]
        }
    ],
    pdf_references=[
        {
            "source": "field_manual.pdf",
            "text": "T-72 tanks have operational range of 500km...",
            "relevance_score": 0.85
        }
    ],
    tactical_assessment="The observed enemy movement pattern indicates preparation for defensive operations...",
    recommendations=[
        "Monitor sector Alpha for continued activity",
        "Prepare counter-measures for potential armor engagement",
        "Request aerial reconnaissance for broader coverage"
    ],
    methodology="Semantic search and object detection across 2 videos, cross-referenced with tactical doctrine"
)
```

## Parameters

### Required
- `title`: Report title (e.g., "Tank Activity Analysis Report")
- `summary`: Executive summary (2-3 sentences)
- `findings`: List of key findings (strings or dicts with text/confidence/source)

### Optional
- `video_evidence`: List of video evidence items (dicts with video_id, segment_id, time_range, description, objects)
- `pdf_references`: List of PDF citations (dicts with source, text, relevance_score)
- `tactical_assessment`: Overall tactical assessment paragraph
- `recommendations`: List of recommended actions
- `methodology`: Description of analysis approach

## Finding Format Options

Findings can be simple strings or structured dicts:

```python
# Simple string
"3 tanks detected in sector Alpha"

# Structured dict
{
    "text": "3 tanks detected in sector Alpha",
    "confidence": "high",
    "source": "video segment 2"
}
```

## Video Evidence Format

```python
{
    "video_id": "m-abc123xyz",
    "segment_id": 2,
    "time_range": "00:00:30 - 00:01:00",
    "description": "Tank formation moving through open terrain",
    "objects": ["tank", "tank", "truck"]
}
```

## PDF Reference Format

```python
{
    "source": "tank_manual.pdf",
    "text": "Relevant excerpt from the document...",
    "relevance_score": 0.85
}
```

## Result Structure

Returns:
- `status`: "success" or "error"
- `result`:
  - `report`: Formatted markdown report
  - `sections`: Metadata about included sections
- `message`: Summary of report contents

## Example Workflow

```python
# 1. Gather evidence using other skills
contexts = videodb_query_skill("get_contexts")
tanks = videodb_query_skill("object_search", object_type="tank")
specs = pdf_rag_skill("tank capabilities")
situation = wargame_query_skill("tactical_situation")

# 2. Synthesize findings
findings = []
video_evidence = []
pdf_references = []

# Process tank results
if tanks["status"] == "success":
    findings.append(tanks["result"]["summary_text"])
    for entry in tanks["result"]["timeline"]:
        video_evidence.append({
            "video_id": entry["video_id"],
            "segment_id": entry["segment_id"],
            "time_range": f"{entry['start_time_hms']} - {entry['end_time_hms']}",
            "description": entry.get("description", ""),
            "objects": [f"tank x{entry['object_count']}"]
        })

# Process PDF results
if specs["status"] == "success":
    for r in specs["result"]["results"][:3]:
        pdf_references.append({
            "source": r["source"],
            "text": r["text"],
            "relevance_score": r["relevance_score"]
        })

# 3. Generate final report
report = final_response_skill(
    title="Tank Activity Analysis",
    summary="Comprehensive analysis of detected tank activity...",
    findings=findings,
    video_evidence=video_evidence,
    pdf_references=pdf_references,
    tactical_assessment="Based on the evidence...",
    recommendations=["Continue monitoring", "Alert command"]
)

# 4. Return the formatted report
final_answer(report["result"]["report"])
```

## Best Practices

1. **Always use for final answers** - Maintains consistency
2. **Include methodology** - Explain how you analyzed
3. **Cite all sources** - Video segments, PDFs, map data
4. **Be specific** - Include segment IDs, time ranges
5. **Actionable recommendations** - Provide clear next steps
6. **Causal relationships** - Connect evidence to conclusions
