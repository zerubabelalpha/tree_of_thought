import json
import re
from .prompts import PROPOSE_PROMPT_TEMPLATE, EVAL_PROMPT_TEMPLATE


def parse_json(text):
    """Parse JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate = match.group(1).strip() if match else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = candidate.find(start_char)
            end = candidate.rfind(end_char)
            if start != -1 and end != -1:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not extract JSON from: {text[:200]}")


def propose(llm, query, parent_thought, papers_context, branching_factor):
    """
    Generate child research directions from a parent thought.
    Returns list of dicts with 'thought' and 'arxiv_query' keys.
    """
    prompt = PROPOSE_PROMPT_TEMPLATE.format(
        query=query,
        parent_thought=parent_thought,
        papers_context=papers_context or "No papers retrieved yet.",
        branching_factor=branching_factor
    )

    try:
        response = llm.complete(
            prompt,
            system_prompt="You are a research planner. Output raw JSON only.",
            temperature=0.7
        )
        directions = parse_json(response)
        if not isinstance(directions, list):
            raise ValueError("Expected a list of research directions.")
        return directions[:branching_factor]
    except Exception:
        keywords = " ".join(query.split()[:4])
        return [
            {"thought": f"General survey of {query}", "arxiv_query": keywords},
            {"thought": f"Recent advances in {query}", "arxiv_query": f"{keywords} recent"},
        ][:branching_factor]


def evaluate(llm, query, thought, papers_summary):
    """
    Score a thought branch based on retrieved papers.
    Returns (score, rationale).
    """
    prompt = EVAL_PROMPT_TEMPLATE.format(
        query=query,
        thought=thought,
        papers_summary=papers_summary or "(No papers found)"
    )

    try:
        response = llm.complete(
            prompt,
            system_prompt="You are a review evaluator. Output raw JSON only.",
            temperature=0.2
        )
        data = parse_json(response)
        return float(data.get("score", 5.0)), data.get("rationale", "No rationale.")
    except Exception:
        has_papers = papers_summary and "(No papers" not in papers_summary
        return (5.0 if has_papers else 2.0), "Fallback score."


def select(candidates, n_select, threshold):
    """
    Greedy selection: pick top-n candidates at or above threshold.
    Returns (selected, discarded). Force-selects best if none meet threshold.
    """
    ranked = sorted(candidates, key=lambda n: n.score, reverse=True)
    selected = [n for n in ranked if n.score >= threshold][:n_select]

    if not selected and ranked:
        selected = [ranked[0]]

    discarded = [n for n in ranked if n not in selected]
    return selected, discarded
