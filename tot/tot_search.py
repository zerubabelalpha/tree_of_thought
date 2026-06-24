import os
import json
import re
import time
from datetime import datetime
from .llm_client import LLMClient
from .arxiv_client import ArxivClient, ArxivPaper
from .prompts import (
    L1_PROMPT_TEMPLATE,
    EVAL_L1_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    EVAL_L2_PROMPT_TEMPLATE,
    FINAL_PROMPT_TEMPLATE
)

class Node:
    """Represents a thought/state node in the Tree of Thoughts search."""
    
    def __init__(self, node_id: str, parent_id: str, depth: int, thought: str, arxiv_query: str = ""):
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.thought = thought
        self.arxiv_query = arxiv_query
        self.papers = []          # List of ArxivPaper objects
        self.score = 0.0          # Evaluation score (1-10)
        self.rationale = ""       # Rationale for evaluation
        self.analysis = ""        # Detailed analysis (Layer 2)
        self.status = "active"    # "active", "selected", "discarded", "final"

    def to_dict(self):
        """Converts node into a clean, simple, jargon-free dictionary."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "thought": self.thought,
            "arxiv_query": self.arxiv_query,
            "papers": [p.to_dict() for p in self.papers],
            "score": self.score,
            "rationale": self.rationale,
            "status": self.status,
            "analysis": self.analysis if self.analysis else None
        }


def parse_json_safely(text: str):
    """Attempts to robustly parse JSON from an LLM response."""
    text = text.strip()
    
    # Remove markdown code formatting blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Fallback: extract substring between first occurrence of [ or { and matching bracket/brace
        first_brace = candidate.find('{')
        first_bracket = candidate.find('[')
        
        start = -1
        end = -1
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start = first_brace
            end = candidate.rfind('}')
        elif first_bracket != -1:
            start = first_bracket
            end = candidate.rfind(']')
            
        if start != -1 and end != -1:
            try:
                return json.loads(candidate[start:end+1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not extract JSON from text: {text}")


class TreeOfThoughts:
    """Executes the Tree of Thoughts search process for research queries."""
    
    def __init__(self, llm_client: LLMClient, arxiv_client: ArxivClient):
        self.llm = llm_client
        self.arxiv = arxiv_client
        self.nodes = {}  # node_id -> Node
        self.next_node_id = 1

    def _generate_id(self) -> str:
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        return node_id

    def solve(self, query: str, max_depth: int = 2, branching_factor: int = 3, score_threshold: float = 5.0):
        """
        Runs the BFS Tree of Thoughts search, yielding status messages for the UI.
        
        Args:
            query (str): The main research prompt.
            max_depth (int): Maximum depth (1 or 2 layers of research).
            branching_factor (int): How many branches to explore at each step.
            score_threshold (float): Minimum score to continue exploring a branch.
            
        Yields:
            tuple (str, str): (current_status_markdown, final_report_markdown_or_empty)
        """
        self.nodes.clear()
        self.next_node_id = 1
        
        status_log = []
        def log(msg):
            status_log.append(msg)
            return "\n".join(status_log)

        yield log(f"Starting Tree of Thoughts Research\n* **Query**: `{query}`\n* **Depth**: `{max_depth}` | **Branching Factor**: `{branching_factor}` | **Score Threshold**: `{score_threshold}`\n\nInitializing search..."), ""

        # Step 0: Create Root Node
        root_id = self._generate_id()
        root_node = Node(
            node_id=root_id,
            parent_id="",
            depth=0,
            thought=f"Root: Researching '{query}'",
            arxiv_query=""
        )
        root_node.status = "selected"
        self.nodes[root_id] = root_node
        
        yield log("Root node initialized. Beginning Layer 1..."), ""

        # -------------------------------------------------------------
        # LAYER 1: Generate Research Directions & Query arXiv
        # -------------------------------------------------------------
        yield log("\nLayer 1: Proposing Research Directions"), ""
        
        l1_prompt = L1_PROMPT_TEMPLATE.format(query=query, branching_factor=branching_factor)

        try:
            l1_response = self.llm.complete(l1_prompt, system_prompt="You are a research planner. Output raw JSON only.", temperature=0.7)
            proposed_directions = parse_json_safely(l1_response)
            if not isinstance(proposed_directions, list):
                raise ValueError("Expected a list of research directions.")
        except Exception as e:
            yield log(f"Warning: Thought generation failed: {e}. Applying fallback directions..."), ""
            # Fallback directions based on query keywords
            keywords = " ".join(query.split()[:4])
            proposed_directions = [
                {"thought": f"Comprehensive review of {query}", "arxiv_query": keywords},
                {"thought": f"Recent methods and advancements in {query}", "arxiv_query": f"{keywords} methods"},
            ]

        l1_nodes = []
        for direction in proposed_directions[:branching_factor]:
            thought_text = direction.get("thought", "Unnamed research direction")
            arxiv_q = direction.get("arxiv_query", "")
            
            node_id = self._generate_id()
            node = Node(
                node_id=node_id,
                parent_id=root_id,
                depth=1,
                thought=thought_text,
                arxiv_query=arxiv_q
            )
            self.nodes[node_id] = node
            l1_nodes.append(node)
            
            yield log(f"Generated thought branch: *\"{thought_text}\"* (arXiv query: `{arxiv_q}`)"), ""

        # Fetch papers & Evaluate Layer 1
        yield log("\n###Fetching literature from arXiv and evaluating paths..."), ""
        
        for node in l1_nodes:
            yield log(f"Searching arXiv for: `{node.arxiv_query}`..."), ""
            papers = self.arxiv.search(node.arxiv_query, max_results=3)
            node.papers = papers
            
            yield log(f"Found {len(papers)} papers. Scoring branch..."), ""
            
            # Format papers for the prompt
            papers_summary = ""
            if not papers:
                papers_summary = "(No papers found on arXiv for this query)"
            else:
                for idx, p in enumerate(papers, 1):
                    papers_summary += f"{idx}. Title: {p.title}\n   Authors: {', '.join(p.authors)}\n   Published: {p.published}\n   Summary: {p.summary[:300]}...\n\n"
            
            eval_prompt = EVAL_L1_PROMPT_TEMPLATE.format(
                query=query,
                thought=node.thought,
                papers_summary=papers_summary
            )

            try:
                eval_response = self.llm.complete(eval_prompt, system_prompt="You are a review evaluator. Output raw JSON only.", temperature=0.2)
                eval_data = parse_json_safely(eval_response)
                node.score = float(eval_data.get("score", 5.0))
                node.rationale = eval_data.get("rationale", "No rationale provided.")
            except Exception as e:
                yield log(f"Evaluation parsing error: {e}. Using fallback score."), ""
                node.score = 5.0 if papers else 2.0
                node.rationale = "Fallback score based on papers presence."

            yield log(f"Score: **{node.score}/10** | *{node.rationale}*"), ""

        # Filter & Select Layer 1
        l1_nodes.sort(key=lambda n: n.score, reverse=True)
        selected_l1_nodes = []
        
        for node in l1_nodes:
            if node.score >= score_threshold and len(selected_l1_nodes) < max(2, branching_factor // 2 + 1):
                node.status = "selected"
                selected_l1_nodes.append(node)
                yield log(f"**Selected**: Branch `{node.node_id}` (*Score: {node.score}*)"), ""
            else:
                node.status = "discarded"
                yield log(f"**Discarded**: Branch `{node.node_id}` (*Score: {node.score}*)"), ""

        if not selected_l1_nodes:
            yield log("No branches met the score threshold. Continuing with the highest scoring branch to avoid complete failure..."), ""
            if l1_nodes:
                l1_nodes[0].status = "selected"
                selected_l1_nodes.append(l1_nodes[0])
                yield log(f"**Forced Selection**: Branch `{l1_nodes[0].node_id}` (*Score: {l1_nodes[0].score}*)"), ""
            else:
                yield log("Critical error: No branches available at all."), ""
                return

        # If max_depth is 1, we skip layer 2 and jump directly to synthesis
        if max_depth < 2:
            yield log("\nMax depth reached. Proceeding to final report synthesis..."), ""
        else:
            # -------------------------------------------------------------
            # LAYER 2: Deep Analysis & Critique
            # -------------------------------------------------------------
            yield log("\n###Layer 2: Deep Literature Analysis & Critique"), ""
            
            l2_nodes = []
            for l1_node in selected_l1_nodes:
                yield log(f"Synthesizing literature analysis for: *\"{l1_node.thought}\"*..."), ""
                
                # Format papers details
                papers_details = ""
                for idx, p in enumerate(l1_node.papers, 1):
                    papers_details += f"Paper {idx}:\nTitle: {p.title}\nAuthors: {', '.join(p.authors)}\nPublished: {p.published}\nURL: {p.url}\nAbstract: {p.summary}\n\n"
                
                analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                    query=query,
                    thought=l1_node.thought,
                    papers_details=papers_details
                )

                try:
                    analysis_text = self.llm.complete(analysis_prompt, system_prompt="You are an objective scientific analyst.", temperature=0.4)
                except Exception as e:
                    yield log(f"Analysis generation failed: {e}. Using fallback summary."), ""
                    analysis_text = f"Fallback summary of papers for research direction: {l1_node.thought}."
                
                # Create Layer 2 node
                node_id = self._generate_id()
                l2_node = Node(
                    node_id=node_id,
                    parent_id=l1_node.node_id,
                    depth=2,
                    thought=f"Analysis of {l1_node.node_id}",
                    arxiv_query=""
                )
                l2_node.papers = l1_node.papers
                l2_node.analysis = analysis_text
                self.nodes[node_id] = l2_node
                l2_nodes.append(l2_node)
                
                # Evaluate Layer 2 Analysis
                yield log(f"Evaluating analysis depth..."), ""
                
                eval_l2_prompt = EVAL_L2_PROMPT_TEMPLATE.format(
                    query=query,
                    analysis=analysis_text
                )

                try:
                    eval_l2_response = self.llm.complete(eval_l2_prompt, system_prompt="You are a review evaluator. Output raw JSON only.", temperature=0.2)
                    eval_l2_data = parse_json_safely(eval_l2_response)
                    l2_node.score = float(eval_l2_data.get("score", 5.0))
                    l2_node.rationale = eval_l2_data.get("rationale", "No rationale provided.")
                except Exception as e:
                    yield log(f"L2 evaluation parsing error: {e}. Using fallback score."), ""
                    l2_node.score = 5.0
                    l2_node.rationale = "Fallback score based on default analysis."

                yield log(f"L2 Analysis Score: **{l2_node.score}/10** | *{l2_node.rationale}*"), ""

            # Filter & Select Layer 2
            l2_nodes.sort(key=lambda n: n.score, reverse=True)
            selected_l2_nodes = []
            
            for node in l2_nodes:
                if node.score >= score_threshold:
                    node.status = "selected"
                    selected_l2_nodes.append(node)
                    # Mark parent as also selected (should already be selected)
                    yield log(f"**Selected**: L2 Analysis `{node.node_id}` (*Score: {node.score}*)"), ""
                else:
                    node.status = "discarded"
                    yield log(f"**Discarded**: L2 Analysis `{node.node_id}` (*Score: {node.score}*)"), ""

            if not selected_l2_nodes:
                yield log("No L2 analyses met the score threshold. Continuing with the highest scoring analysis to avoid failure..."), ""
                if l2_nodes:
                    l2_nodes[0].status = "selected"
                    selected_l2_nodes.append(l2_nodes[0])
                    yield log(f"**Forced Selection**: L2 Analysis `{l2_nodes[0].node_id}` (*Score: {l2_nodes[0].score}*)"), ""

        # -------------------------------------------------------------
        # LAYER 3: Synthesis & Final Report
        # -------------------------------------------------------------
        yield log("\nLayer 3: Synthesizing Final Research Report"), ""
        
        # Build successful paths representation
        successful_paths_summary = ""
        path_count = 1
        
        # Find all nodes that are selected
        for n_id, node in self.nodes.items():
            if node.depth == 1 and node.status == "selected":
                successful_paths_summary += f"### Direction {path_count}: {node.thought}\n"
                successful_paths_summary += f"* **arXiv search query**: `{node.arxiv_query}`\n"
                successful_paths_summary += "* **Retrieved literature**:\n"
                for p in node.papers:
                    successful_paths_summary += f"  - *{p.title}* by {', '.join(p.authors)} ({p.published}) - [Paper Link]({p.url}) | [PDF Link]({p.pdf_url})\n"
                
                # Check if there is a child analysis that was selected
                child_analysis = next((child for child in self.nodes.values() if child.parent_id == node.node_id and child.status == "selected"), None)
                if child_analysis and child_analysis.analysis:
                    successful_paths_summary += f"\n* **Expert sub-analysis**:\n{child_analysis.analysis}\n"
                
                successful_paths_summary += "\n---\n\n"
                path_count += 1

        final_prompt = FINAL_PROMPT_TEMPLATE.format(
            query=query,
            successful_paths_summary=successful_paths_summary
        )

        yield log("Drafting final report with unified literature synthesis..."), ""
        
        try:
            final_report = self.llm.complete(final_prompt, system_prompt="You are a professional academic writer.", temperature=0.5)
        except Exception as e:
            final_report = f"# Research Report: {query}\n\nError generating report: {e}\n\nWe successfully explored the literature, but could not synthesize a final report. Please check the JSON tree log."

        # Save search tree to file
        self._save_tree_to_file()
        
        yield log("**Deep Research Complete!** Report generated successfully and tree log saved to `research_tree.json`."), final_report

    def _save_tree_to_file(self):
        """Saves the complete tree structure to a JSON file in the background."""
        tree_data = {
            "timestamp": datetime.now().isoformat(),
            "nodes": [node.to_dict() for node in self.nodes.values()]
        }
        
        # Save as the latest run
        latest_filename = "research_tree.json"
        try:
            with open(latest_filename, "w", encoding="utf-8") as f:
                json.dump(tree_data, f, indent=2)
            
            # Also save with timestamp to avoid overwriting history
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_filename = f"research_tree_{timestamp_str}.json"
            with open(timestamp_filename, "w", encoding="utf-8") as f:
                json.dump(tree_data, f, indent=2)
                
            print(f"Saved tree structure to {latest_filename} and {timestamp_filename}")
        except Exception as e:
            print(f"Failed to save tree structure: {e}")
