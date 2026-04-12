from __future__ import annotations
from typing import List, Optional, Dict
from .base import QueryItem, MethodConfig
from .llm import OpenAICompatibleClient
from .prompts import PromptBank
from .registry import METHODS

# Import all methods to ensure they're registered
from ..methods.genqr import GENQR
from ..methods.genqr_ensemble import GenQREnsemble
from ..methods.query2doc import Query2Doc
from ..methods.qa_expand import QAExpand
from ..methods.mugi import MuGI
from ..methods.lamer import LameR
from ..methods.query2e import Query2E
from ..methods.csqe import CSQE
from ..methods.thinkqe import ThinkQE


def build_llm(cfg: MethodConfig):
    llm_cfg = cfg.llm
    return OpenAICompatibleClient(
        model=llm_cfg["model"], base_url=llm_cfg.get("base_url"), api_key=llm_cfg.get("api_key")
    )


def run_method(
    method_name: str,
    cfg: MethodConfig,
    queries: List[QueryItem],
    prompt_bank_path: str,
    ctx_map: Optional[Dict[str, List[str]]] = None,
):
    Method = METHODS[method_name]
    llm = build_llm(cfg)
    pb = PromptBank(prompt_bank_path)
    reformulator = Method(cfg, llm, pb)
    return reformulator.reformulate_batch(queries, ctx_map)
