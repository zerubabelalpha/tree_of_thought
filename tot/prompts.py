


L1_PROMPT_TEMPLATE = """You are a lead academic researcher.
The main research query is: "{query}"

Your task is to propose {branching_factor} distinct, promising research sub-directions/angles to investigate.
For each sub-direction, you must provide:
1. A concise, clear explanation of the thought/angle (1-2 sentences).
2. A search query for the arXiv API. The search query should be simple, containing keywords (e.g., "surface codes error correction", do not use boolean operators like AND/OR/NOT).

Respond ONLY with a valid JSON array of objects, containing "thought" and "arxiv_query" fields.
Example format:
[
  {{
    "thought": "Analysis of topological surface codes and their error thresholds.",
    "arxiv_query": "surface codes topological error threshold"
  }}
]"""

EVAL_L1_PROMPT_TEMPLATE = """You are a senior academic reviewer.
We are research-solving the query: "{query}"
We proposed the following research direction:
"{thought}"

To verify this direction, we fetched the following papers from arXiv:
{papers_summary}

Based on the relevance of these papers and the promise of this research direction, please score this thought on a scale of 1 to 10 (10 being highly relevant, original, and promising; 1 being completely off-topic or unhelpful).
Provide a brief, jargon-free 1-sentence rationale for the score.
If no papers were found (list is empty), rate this thought accordingly, since a research path requires literature support.

Respond ONLY with a valid JSON object containing "score" and "rationale" fields.
Example format:
{{
  "score": 8,
  "rationale": "The fetched papers directly address quantum stabilizer code scaling, providing a strong literature foundation."
}}"""

ANALYSIS_PROMPT_TEMPLATE = """You are an expert scientist conducting a deep literature review.
The main research query is: "{query}"
The research direction being pursued is: "{thought}"

Here is the literature we fetched from arXiv:
{papers_details}

Please write a structured, detailed sub-analysis of this research direction based on these papers.
Identify:
1. The core methodology or techniques proposed.
2. Major findings and breakthroughs.
3. Identified limitations or open challenges.

Be specific and ground your analysis in the provided paper details (cite author names and years if available). Do not add general buzzwords."""

EVAL_L2_PROMPT_TEMPLATE = """You are an academic editor.
We are reviewing research on: "{query}"
Here is a draft sub-analysis of one research direction:
"{analysis}"

Please evaluate this analysis for clarity, depth, and relevance to the overall research query.
Score the analysis on a scale of 1 to 10 (10 being an outstanding, clear, and informative analysis; 1 being shallow, inaccurate, or irrelevant).
Provide a brief, jargon-free 1-sentence rationale.

Respond ONLY with a valid JSON object containing "score" and "rationale" fields.
Example format:
{{
  "score": 9,
  "rationale": "The analysis is well-grounded in the literature, clearly laying out the trade-offs of stabilizers."
}}"""

FINAL_PROMPT_TEMPLATE = """You are the lead author of a state-of-the-art survey paper.
The research query is: "{query}"

We have explored multiple research directions using a Tree of Thoughts approach. Here are the successful research paths we analyzed and verified with arXiv literature:

{successful_paths_summary}

Please write a comprehensive, cohesive, and professional academic-quality research report answering the research query.
The report must:
1. Start with a clear Title and an Executive Summary.
2. Have structured sections matching the successful research directions we explored.
3. Synthesize the findings across these directions, explicitly citing the authors and papers we retrieved.
4. Discuss limitations, open questions, and concrete future research recommendations.
5. Format the report beautifully using Markdown (bold headings, bullet points, numbered lists, blockquotes, etc.).

Ensure the output is written in a clear, objective, scientific tone."""
