import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
import json
import os

from dspy import LM,ReAct,Example,Prediction
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from typing import Optional




load_dotenv()

vector_store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../chroma_persist")


chroma_client = PersistentClient(path=vector_store_path)
# emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="blockcure_collection")
print(collection.count())
exit()
lm = LM(
            "openrouter/openai/gpt-oss-20b", 
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            temperature=0.3, 
            max_tokens=64000,
            cache=False
        )

dspy.configure(lm=lm)

class AssistantQA(dspy.Signature):
    """You are a helpful assistant. Answer a question using retrieved passages"""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class AssistantAgent(dspy.Module):
    def __init__(
        self
    ):
        self.lm = lm
        
        self._agent = ReAct(
            AssistantQA,
            tools=[vector_search_tool],
            max_iters=3,
        )

    def forward(self, question: str)->str:
        return self._agent(question=question)

def vector_search_tool(query: str, k: int = 3) -> str:
    """
    Returns top-k relevant documents from the vector database for the given query.
    """

    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return "\n\n".join(results["documents"][0])
    
class JudgeConsistency(dspy.Signature):
    """Judge whether the predicted answer matches the gold answer.

    # Instructions:
    - The score should be between 0.0 and 1.0 and based on the similarity of the predicted answer and the gold answer.
    - The justification should be a brief explanation of the score.
    - If the answer doesn't address the question properly, the score should be less than 0.5.
    - If the answer is completely correct, the score should be 1.0. Otherwise, the score should be less than 1.0.
    - Be very strict in your judgement as this is a medical question.
    """
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    justification: str = dspy.OutputField()

class JudgeReactStep(dspy.Signature):
    """Judge whether the next tool call (name + args) is appropriate, well-formed, and relevant.

    - Output a strict score in [0, 1].
    - Provide a brief justification and a yes/no style verdict in justification text.
    """
    question: str = dspy.InputField()
    tool_name: str = dspy.InputField()
    tool_args_json: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    verdict: str = dspy.OutputField()
    justification: str = dspy.OutputField()

def metric(gold: Example,
    pred: Prediction,
    trace = None,
    pred_name: Optional[str] = None,
    pred_trace = None):
    
    if pred_name and (pred_name == "react" or pred_name.endswith(".react")) and pred_trace:
        try:
                _, step_inputs, step_outputs = pred_trace[0]
        except Exception:
            step_inputs, step_outputs = {}, {}

        question_text = getattr(gold, "question", None) or step_inputs.get("question", "") or ""

        def _get(o, key, default=""):
            if isinstance(o, dict):
                return o.get(key, default)
            return getattr(o, key, default)

        tool_name = _get(step_outputs, "next_tool_name", "")
        tool_args = _get(step_outputs, "next_tool_args", {})

        args_is_dict = isinstance(tool_args, dict)
        has_query = args_is_dict and isinstance(tool_args.get("query"), str) and tool_args.get("query", "").strip() != ""
        k_val = tool_args.get("k") if args_is_dict else None
        k_ok = isinstance(k_val, int) and 1 <= k_val <= 10 or k_val is None
        used_tool = tool_name not in ("", "finish")
        early_finish = tool_name == "finish"

        heuristics_score = 0.0
        if used_tool:
            heuristics_score += 0.4
        if has_query:
            heuristics_score += 0.4
        if k_ok:
            heuristics_score += 0.1
        if not early_finish:
            heuristics_score += 0.1
        heuristics_score = max(0.0, min(1.0, heuristics_score))

        tool_args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
        with dspy.settings.context(lm=lm):
            react_judge = dspy.Predict(JudgeReactStep)
            judged = react_judge(
                question=question_text,
                tool_name=str(tool_name),
                tool_args_json=tool_args_json,
            )

        llm_score = getattr(judged, "score", 0.0) or 0.0
        llm_score = max(0.0, min(1.0, llm_score))
        llm_just = getattr(judged, "justification", "") or ""

        total = 0.5 * heuristics_score + 0.5 * llm_score

        suggestions = []
        if not used_tool:
            suggestions.append("Select a retrieval tool before finishing.")
        if early_finish:
            suggestions.append("Avoid selecting 'finish' until you have evidence from the retrieval tool.")
        if not args_is_dict:
            suggestions.append("Emit next_tool_args as a valid JSON object.")
        else:
            if not has_query:
                suggestions.append("Include a non-empty 'query' string in next_tool_args.")
            if k_val is not None and (not isinstance(k_val, int) or k_val < 1 or k_val > 10):
                suggestions.append("Choose a reasonable k (e.g., 3–5).")
        if not suggestions:
            suggestions.append("Good step. Keep queries concise and set k=5 by default.")

        feedback_text = (
            f"ReAct step — LLM score: {llm_score:.2f}, heuristics: {heuristics_score:.2f}. "
            + " ".join(suggestions)
            + (f" LLM justification: {llm_just}" if llm_just else "")
        ).strip()

        return ScoreWithFeedback(score=total, feedback=feedback_text)

    if gold is None or pred is None:
        return ScoreWithFeedback(score=0.0, feedback="Missing example or pred")

    predicted_answer = getattr(pred, "answer", None) or ""
    if not predicted_answer.strip():
        return ScoreWithFeedback(score=0.0, feedback="Empty prediction")

    with dspy.settings.context(lm=lm):
        judge = dspy.Predict(JudgeConsistency)
        judged = judge(
            question=gold.question,
            gold_answer=gold.answer,
            predicted_answer=predicted_answer,
        )

    score = getattr(judged, "score", None) or 0.0
    score = max(0.0, min(1.0, score))
    justification = getattr(judged, "justification", "") or ""
    feedback_text = f"Score: {score}. {justification}".strip()
    return ScoreWithFeedback(score=score, feedback=feedback_text)



if __name__ == "__main__":
    assistant = AssistantAgent()
    # print(assistant("What are the common symptoms of diabetes?"))


    # dataset = dspy.load_dataset("dataset.json", split="test")
    dataset = []
    with open(r"D:\projects\blockcure\SRC\DB\dataset.json", "r") as f:
        dataset_json = json.load(f)
        
        for i, item in enumerate(dataset):
            dataset.append(dspy.Example(question=item["diagnosis"], answer=item["treatment"]))
    print(f"Loaded {len(dataset)} examples for evaluation.")
    exit()
    evaluator = Evaluate(
        devset=dataset,
        num_threads=8,
        display_progress=True,
    )

    evaluation_result = evaluator(
        assistant,
        metric=metric,
    )
    print("Evaluation Result:", evaluation_result)