from __future__ import annotations
import json
import re
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method


def parse_llm_json(text: str) -> dict:
    """
    Robust JSON parser for LLM outputs that handles common formatting issues:
    - Markdown code blocks (```json, ```)
    - Escaped quotes (\")
    - Trailing commas
    - Single quotes instead of double quotes
    - Incomplete/truncated JSON
    - Extracts key-value pairs even from malformed JSON
    """
    if not text or not text.strip():
        return {}

    original_text = text
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Unescape common escape sequences that appear in raw output
    text = text.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")

    # Try standard JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Replace single quotes with double quotes
    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    # Try to fix incomplete JSON by adding missing closing quotes and braces
    if "{" in text:
        # Find the last complete value and try to close it
        attempt = text.rstrip()

        # If ends mid-string (no closing quote), add closing quote
        if attempt.count('"') % 2 == 1:
            attempt += '"'

        # Add missing closing brace if needed
        if not attempt.endswith("}"):
            attempt += "}"

        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    # Fallback: Extract key-value pairs using regex
    # Looks for patterns like "key": "value" or 'key': 'value'
    result = {}

    # Pattern to match JSON key-value pairs
    patterns = [
        r'["\'](\w+)["\']\s*:\s*["\']([^"\']*)["\']',  # Standard JSON
        r'(\w+)\s*:\s*["\']([^"\']*)["\']',  # Unquoted keys
        r'["\'](\w+)["\']\s*:\s*([^,}\]]+)',  # Values without quotes
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for key, value in matches:
            if key and value:
                result[key] = value.strip()

    # If we found any key-value pairs, return them
    if result:
        return result

    # Absolute fallback: return empty dict
    return {}


@register_method("qa_expand")
class QAExpand(BaseReformulator):
    """
    QA-Expand Pipeline (from paper):
    1. Generate 3 sub-questions from original query (LLM call #1)
    2. Generate 3 pseudo-answers from the 3 sub-questions (LLM call #2) - NO contexts
    3. Filter and refine answers based on relevance to query (LLM call #3)
    4. Expand query: (Q × 3) + refined_answers
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = False  # No contexts needed - pure generative expansion
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"  # (Q × 3) + refined_answers
    DEFAULT_QUERY_REPEATS = 3  # K=3 (fixed)

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # K is fixed at 3 for QA-Expand
        k = 3

        # Get configurable parameters
        max_tokens = self.cfg.params.get("max_tokens", self.cfg.llm.get("max_tokens", 256))
        default_temp = self.cfg.llm.get("temperature", 0.8)

        # Temperature settings (can override per-step, defaults to global temperature)
        temp_subq = self.cfg.params.get("temperature_subq", default_temp)
        temp_answer = self.cfg.params.get("temperature_answer", default_temp)
        temp_refine = self.cfg.params.get("temperature_refine", default_temp)

        # Prompt IDs (customizable for different datasets)
        prompt_subq = self.cfg.params.get("prompt_subq", "qa_expand.subq.v1")
        prompt_answer = self.cfg.params.get("prompt_answer", "qa_expand.answer.v1")
        prompt_refine = self.cfg.params.get("prompt_refine", "qa_expand.refine.v1")

        # Step 1: Generate 3 sub-questions (LLM call #1)
        # Input: {query}
        # Output: Raw text with 3 questions
        msgs = self.prompts.render(prompt_subq, query=q.text)
        subqs_raw = self.llm.chat(msgs, temperature=temp_subq, max_tokens=max_tokens)

        # Format questions as JSON for next step
        # Try to parse if LLM already returned JSON, otherwise split by newlines
        questions_dict = parse_llm_json(subqs_raw)

        if questions_dict and any(k.startswith("question") for k in questions_dict.keys()):
            # LLM returned JSON format already
            questions_json = json.dumps(questions_dict)
        else:
            # Parse from text - split by newlines and clean up
            lines = [line.strip() for line in subqs_raw.split("\n") if line.strip()]
            # Remove common prefixes like "1.", "Q1:", "-", etc.
            cleaned_lines = []
            for line in lines:
                line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line)  # Remove "1.", "1)", "-", "*"
                line = re.sub(r"^[Qq]\d+[\:\.]?\s*", "", line)  # Remove "Q1:", "q1."
                if line:
                    cleaned_lines.append(line)

            # Take first 3 questions
            questions_json = json.dumps(
                {
                    "question1": cleaned_lines[0] if len(cleaned_lines) > 0 else "",
                    "question2": cleaned_lines[1] if len(cleaned_lines) > 1 else "",
                    "question3": cleaned_lines[2] if len(cleaned_lines) > 2 else "",
                }
            )

        # Step 2: Generate 3 pseudo-answers from LLM's internal knowledge (LLM call #2)
        # Input: {questions} (JSON format)
        # Output: JSON with {"answer1": "...", "answer2": "...", "answer3": "..."}
        # NOTE: No contexts used - LLM generates answers from its own knowledge
        msgs = self.prompts.render(prompt_answer, questions=questions_json)
        answers_json = self.llm.chat(msgs, temperature=temp_answer, max_tokens=max_tokens)

        # Step 3: Filter and refine answers (LLM call #3)
        # Input: {query}, {answers} (JSON format)
        # Output: Refined JSON with relevant answers only
        msgs = self.prompts.render(prompt_refine, query=q.text, answers=answers_json)
        refined_answers_json = self.llm.chat(msgs, temperature=temp_refine, max_tokens=max_tokens)

        # Parse refined JSON and extract answer values for concatenation
        refined_dict = parse_llm_json(refined_answers_json)
        if refined_dict:
            # Extract answer values in order (answer1, answer2, answer3)
            answer_values = [
                str(v) for k, v in sorted(refined_dict.items()) if v and str(v).strip()
            ]
            refined_text = " ".join(answer_values)
        else:
            # Fallback if JSON parsing fails - use raw text
            refined_text = refined_answers_json

        # Step 4: Expand query: (Q × 3) + refined_answers
        reformulated = self.concatenate_result(q.text, refined_text, query_repeats=k)

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "subquestions_raw": subqs_raw,
                "questions_json": questions_json,
                "answers_json": answers_json,
                "refined_answers_json": refined_answers_json,
                "refined_text": refined_text,
                "prompts_used": {
                    "subq": prompt_subq,
                    "answer": prompt_answer,
                    "refine": prompt_refine,
                },
            },
        )
