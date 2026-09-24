from langchain_core.documents import Document

from agent.agent import agent
from evaluation.dataset import test_cases
import time
from evaluation import metrics
import json
from rag.rag import retrieve_for_evaluation

results=[]

def get_called_tools(results):
    tools=[]
    for message in results["messages"]:
        if hasattr(message,"tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tools.append(tool_call["name"])
    return tools

def get_token_usage(results):
    input_tokens=0
    output_tokens = 0
    total_tokens = 0

    for message in results["messages"]:
        usage=getattr(message,"usage_metadata",None)
        if usage:
            input_tokens+=usage.get("input_tokens",0)
            output_tokens+=usage.get("output_tokens",0)
            total_tokens+=usage.get("total_tokens",0)

    return {"input_tokens":input_tokens,
            "output_tokens":output_tokens,
            "total_tokens":total_tokens}


for i,case in enumerate(test_cases):
    start=time.perf_counter()
    result=agent.invoke({"messages":[("user",case["question"])]},config={"configurable":{"thread_id":f"eval_{i}"}})
    latency=time.perf_counter()-start

    if case["type"] in ("rag","source"):
        retriever_documents=retrieve_for_evaluation(case["question"]),
    else:
        retriever_documents=None
    results.append({
        "type": case["type"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "agent_answer": result["messages"][-1].content,
        "expected_tool": [case["expected_tool"]],
        "tool_calls": get_called_tools(result),
        "latency": latency,
        "input_tokens": get_token_usage(result)["input_tokens"],
        "output_tokens": get_token_usage(result)["output_tokens"],
        "total_tokens": get_token_usage(result)["total_tokens"],
        "retriever_documents":retriever_documents
    })





print(f"results:",results)
print(f"平均延迟时间:",metrics.calculate_average_latency(results))
print(f"工具使用正确率：",metrics.calculate_tool_selection_accuracy(results))
print(f"回答正确率：",metrics.judge_answer(results))
print(f"token使用总数量：{metrics.calculate_token_usage(results)["average_total_tokens"]}\n输入token数量：{metrics.calculate_token_usage(results)["average_input_tokens"]}\n输出token数量：{metrics.calculate_token_usage(results)["average_output_tokens"]}")
print(f"文档检索相关度：{metrics.judge_retriever_accuracy(results)}")
print(f"失败率：",metrics.calculate_failure_rate(results))


#保存结果
def default_handler(doc):
    if isinstance(doc,Document):
        return{"page_content":doc.page_content,"metadata":doc.metadata}
    return str(doc)

with open("results.json","w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=4,default=default_handler)

summary={
    "average_latency":metrics.calculate_average_latency(results),
    "tool_selection_accuracy":metrics.calculate_tool_selection_accuracy(results),
    "answer_accuracy":metrics.judge_answer(results),
    "average_total_tokens":metrics.calculate_token_usage(results)["average_total_tokens"],
    "average_input_tokens":metrics.calculate_token_usage(results)["average_input_tokens"],
    "average_output_tokens":metrics.calculate_token_usage(results)["average_output_tokens"],
    "retriever_accuracy":metrics.judge_retriever_accuracy(results),
    "failure_rate":metrics.calculate_failure_rate(results)
}

with open("summary.json","w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=4)

