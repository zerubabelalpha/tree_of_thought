import json
from datetime import datetime
from . import bfs
from .llm_client import LLMClient
from .arxiv_client import ArxivClient
from .prompts import ANALYSIS_PROMPT_TEMPLATE, FINAL_PROMPT_TEMPLATE


class Node:
    """Represents a thought/state node in the Tree of Thoughts search."""

    def __init__(self, node_id: str, parent_id: str, depth: int, thought: str, arxiv_query: str = ""):
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.thought = thought
        self.arxiv_query = arxiv_query
        self.papers = []
        self.score = 0.0
        self.rationale = ""
        self.analysis = ""
        self.status = "active"  # "active", "selected", "discarded"

    def to_dict(self):
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


class TreeOfThoughts:
    """Executes BFS Tree of Thoughts search for research queries."""

    def __init__(self, llm_client: LLMClient, arxiv_client: ArxivClient):
        self.llm = llm_client
        self.arxiv = arxiv_client
        self.nodes = {}
        self.next_node_id = 1

    def _generate_id(self) -> str:
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        return node_id

    def _format_papers_brief(self, papers) -> str:
        """Brief paper list for the propose prompt's papers context."""
        if not papers:
            return "No papers retrieved yet."
        return "\n".join(f"- {p.title} ({p.published})" for p in papers)

    def _format_papers_summary(self, papers) -> str:
        """Detailed paper summary for the evaluation prompt."""
        if not papers:
            return "(No papers found on arXiv for this query)"
        summary = ""
        for idx, p in enumerate(papers, 1):
            summary += (
                f"{idx}. Title: {p.title}\n"
                f"   Authors: {', '.join(p.authors)}\n"
                f"   Published: {p.published}\n"
                f"   Summary: {p.summary[:300]}...\n\n"
            )
        return summary

    def _build_paths_summary(self, frontier) -> str:
        """Build successful paths summary for final report synthesis."""
        summary = ""
        for i, node in enumerate(frontier, 1):
            summary += f"### Direction {i}: {node.thought}\n"
            summary += f"* **arXiv search query**: `{node.arxiv_query}`\n"
            summary += "* **Retrieved literature**:\n"
            for p in node.papers:
                summary += (
                    f"  - *{p.title}* by {', '.join(p.authors)} ({p.published})"
                    f" - [Link]({p.url}) | [PDF]({p.pdf_url})\n"
                )
            if node.analysis:
                summary += f"\n* **Expert analysis**:\n{node.analysis}\n"
            summary += "\n---\n\n"
        return summary

    def solve(self, query: str, max_depth: int = 2, branching_factor: int = 3, score_threshold: float = 5.0):
        """
        BFS Tree of Thoughts search with backtracking.
        Yields:
            tuple (str, str): (status_markdown, final_report_or_empty)
        """
        self.nodes.clear()
        self.next_node_id = 1
        n_select = max(1, branching_factor // 2 + 1)

        status_log = []
        def log(msg):
            status_log.append(msg)
            return "\n".join(status_log)

        yield log(
            f"Starting Tree of Thoughts Research\n\n"
            f"* **Query**: `{query}`\n"
            f"* **Depth**: `{max_depth}` | **Branching Factor**: `{branching_factor}` "
            f"| **Score Threshold**: `{score_threshold}`\n\n"
            f"Initializing BFS search..."
        ), ""

        # Root node
        root_id = self._generate_id()
        root = Node(root_id, "", 0, query)
        root.status = "selected"
        self.nodes[root_id] = root

        frontier = [root]
        backtrack_pool = []

        yield log("Root node created. Beginning exploration...\n"), ""

        # ===================== BFS Loop =====================
        for depth in range(1, max_depth + 1):
            yield log(f"## Layer {depth}: Expanding {len(frontier)} frontier node(s)"), ""

            candidates = []

            for parent in frontier:
                papers_context = self._format_papers_brief(parent.papers)

                yield log(f"Proposing directions from: *\"{parent.thought[:100]}\"*..."), ""
                directions = bfs.propose(
                    self.llm, query, parent.thought, papers_context, branching_factor
                )

                for d in directions:
                    thought_text = d.get("thought", "Unnamed direction")
                    arxiv_q = d.get("arxiv_query", "")

                    node_id = self._generate_id()
                    node = Node(node_id, parent.node_id, depth, thought_text, arxiv_q)
                    self.nodes[node_id] = node

                    yield log(f"Generated thought branch: *\"{thought_text}\"* (arXiv query: `{arxiv_q}`)"), ""

                    # Fetch papers
                    yield log(f"Searching arXiv for: `{arxiv_q}`..."), ""
                    node.papers = self.arxiv.search(arxiv_q, max_results=3)
                    yield log(f"Found {len(node.papers)} papers. Scoring branch..."), ""

                    # Evaluate
                    papers_summary = self._format_papers_summary(node.papers)
                    node.score, node.rationale = bfs.evaluate(
                        self.llm, query, node.thought, papers_summary
                    )
                    yield log(f"Score: **{node.score}/10** | *{node.rationale}*"), ""

                    candidates.append(node)

            # Selection
            selected, discarded = bfs.select(candidates, n_select, score_threshold)
            backtrack_pool.extend(discarded)

            for n in discarded:
                n.status = "discarded"
                yield log(f"**Discarded**: Branch `{n.node_id}` (Score: {n.score})"), ""
            for n in selected:
                n.status = "selected"
                yield log(f"**Selected**: Branch `{n.node_id}` (Score: {n.score})"), ""

            # Backtracking: if all selected nodes scored below threshold,
            # check if a previously discarded node from an earlier depth is better
            if selected and selected[0].score < score_threshold and backtrack_pool:
                backtrack_pool.sort(key=lambda n: n.score, reverse=True)
                if backtrack_pool[0].score > selected[0].score:
                    revived = backtrack_pool.pop(0)
                    revived.status = "selected"
                    selected.append(revived)
                    yield log(
                        f"**Backtrack**: Revived `{revived.node_id}` "
                        f"(Score: {revived.score}) from depth {revived.depth}"
                    ), ""

            if not selected:
                yield log("Critical error: No candidates could be generated."), ""
                return

            frontier = selected

        # ============ Post-BFS: Deep Analysis ============
        yield log("\n## Deep Analysis of selected research paths"), ""

        for node in frontier:
            yield log(f"Synthesizing literature analysis for: *\"{node.thought[:100]}\"*..."), ""

            papers_details = ""
            for idx, p in enumerate(node.papers, 1):
                papers_details += (
                    f"Paper {idx}:\nTitle: {p.title}\n"
                    f"Authors: {', '.join(p.authors)}\nPublished: {p.published}\n"
                    f"URL: {p.url}\nAbstract: {p.summary}\n\n"
                )

            analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                query=query,
                thought=node.thought,
                papers_details=papers_details or "(No papers available)"
            )

            try:
                node.analysis = self.llm.complete(
                    analysis_prompt,
                    system_prompt="You are an objective scientific analyst.",
                    temperature=0.4
                )
            except Exception as e:
                node.analysis = f"Analysis could not be generated: {e}"

            yield log(f"Analysis complete for `{node.node_id}`."), ""

        # ============ Final Synthesis ============
        yield log("\n## Synthesizing Final Research Report"), ""

        paths_summary = self._build_paths_summary(frontier)
        final_prompt = FINAL_PROMPT_TEMPLATE.format(
            query=query,
            successful_paths_summary=paths_summary
        )

        yield log("Drafting final report with unified literature synthesis..."), ""

        try:
            final_report = self.llm.complete(
                final_prompt,
                system_prompt="You are a professional academic writer.",
                temperature=0.5
            )
        except Exception as e:
            final_report = (
                f"# Research Report: {query}\n\n"
                f"Error generating report: {e}\n\n"
                f"Check research_tree.json for explored paths."
            )

        self._save_tree_to_file()

        yield log(
            "**Deep Research Complete!** Report generated successfully "
            "and tree log saved to `research_tree.json`."
        ), final_report

    def _save_tree_to_file(self):
        """Saves the complete tree structure to a JSON file."""
        tree_data = {
            "timestamp": datetime.now().isoformat(),
            "nodes": [node.to_dict() for node in self.nodes.values()]
        }

        latest_filename = "research_tree.json"
        try:
            with open(latest_filename, "w", encoding="utf-8") as f:
                json.dump(tree_data, f, indent=2)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_filename = f"research_tree_{timestamp_str}.json"
            with open(timestamp_filename, "w", encoding="utf-8") as f:
                json.dump(tree_data, f, indent=2)

            print(f"Saved tree structure to {latest_filename} and {timestamp_filename}")
        except Exception as e:
            print(f"Failed to save tree structure: {e}")
