from querygym.core.base import MethodConfig, QueryItem
from querygym.core.prompts import PromptBank
from querygym.methods.genqr_ensemble import GenQREnsemble
from pathlib import Path


class DummyLLM:
    def chat(self, messages, **kwargs):
        return "bertopic"


def test_genqr_ensemble():
    cfg = MethodConfig(
        name="genqr_ensemble", params={"repeat_query_weight": 2}, llm={"model": "dummy"}
    )
    llm = DummyLLM()
    pb = PromptBank(Path(__file__).parents[1] / "querygym" / "prompt_bank.yaml")
    meth = GenQREnsemble(cfg, llm, pb)
    res = meth.reformulate(QueryItem("Q1", "bertopic"))
    assert "bertopic bertopic" in res.reformulated
