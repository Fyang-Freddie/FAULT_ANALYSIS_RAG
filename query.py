"""用户查询入口：LangGraph ReAct 风格检索、联网兜底和失效分析。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TypedDict

from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from rag_core import HybridRetriever, Settings, read_attachment


class AgentState(TypedDict, total=False):
    """图中各节点共享的最小状态，trace 只记录可展示的行动摘要。"""

    question: str
    attachment: str
    local_hits: list[dict]
    use_web: bool
    web_hits: list[dict]
    answer: str
    trace: list[str]


# 模型接口完全来自 .env，可连接 OpenAI 或任意兼容服务。
def create_llm() -> ChatOpenAI:
    api_key = os.getenv("API_KEY")
    model = os.getenv("API_MODEL")
    if not api_key or not model:
        raise ValueError(".env 中必须配置 API_KEY 和 API_MODEL")
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=os.getenv("API_BASE_URL") or None,
        temperature=0,
    )


def format_local_hits(hits: list[dict]) -> str:
    """为每条本地证据生成稳定编号，方便模型在结论中引用。"""
    blocks = []
    for number, hit in enumerate(hits, 1):
        doc = hit["document"]
        title = doc.metadata.get("report_title") or doc.metadata.get("file_name")
        blocks.append(f"[L{number}] {title}（匹配度 {hit['score']:.3f}）\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_web_hits(hits: list[dict]) -> str:
    """联网结果保留标题、摘要和 URL，避免生成无法追溯的网络事实。"""
    return "\n\n".join(
        f"[W{i}] {hit.get('title', '')}\n{hit.get('body', '')}\nURL: {hit.get('href', '')}"
        for i, hit in enumerate(hits, 1)
    )


def build_graph(settings: Settings):
    """构建受控 ReAct 图：观察本地证据、决定行动、联网补证、生成答案。"""
    retriever = HybridRetriever(settings)
    llm = create_llm()

    # Act 1：始终先查企业本地知识，符合“本地优先”的要求。
    def local_search(state: AgentState) -> AgentState:
        search_text = f"{state['question']}\n{state.get('attachment', '')[:2500]}"
        hits = retriever.search(search_text)
        top_score = hits[0]["score"] if hits else 0.0
        return {
            "local_hits": hits,
            "trace": [f"本地 Hybrid 检索完成，最高匹配度 {top_score:.3f}"],
        }

    # Reason：低于硬阈值必定联网；高于阈值时再让 Agent 判断证据是否有明显缺口。
    def decide_next_action(state: AgentState) -> AgentState:
        top_score = state["local_hits"][0]["score"] if state["local_hits"] else 0.0
        if top_score < settings.local_score_threshold:
            return {
                "use_web": True,
                "trace": state["trace"] + ["本地证据低于阈值，执行联网补充检索"],
            }

        prompt = f"""你是失效分析检索代理。只判断现有证据能否支撑回答，不要展开分析过程。
问题：{state['question']}
本地证据：
{format_local_hits(state['local_hits'])[:6000]}

若证据缺少问题涉及的机理、标准或处置依据，只回复 WEB；否则只回复 LOCAL。"""
        decision = str(llm.invoke(prompt).content).strip().upper()
        use_web = decision.startswith("WEB")
        action = "Agent 判断存在证据缺口，执行联网补充检索" if use_web else "Agent 判断本地证据足够"
        return {"use_web": use_web, "trace": state["trace"] + [action]}

    # Act 2：使用无需额外密钥的搜索源；网络结果只是补充证据，不覆盖本地事实。
    def web_search(state: AgentState) -> AgentState:
        query = f"{state['question']} 失效分析 原因 机理 建议"[:400]
        try:
            hits = list(DDGS().text(query, max_results=settings.web_max_results))
            message = f"联网获得 {len(hits)} 条结果"
        except Exception as error:
            # 网络不可用不应让整份分析丢失；答案会明确标记外部证据缺失。
            hits = []
            message = f"联网失败：{type(error).__name__}，将基于现有证据回答"
        return {"web_hits": hits, "trace": state["trace"] + [message]}

    # 最终推理只输出可核查的因果链和依据，不暴露模型隐藏思维链。
    def answer(state: AgentState) -> AgentState:
        user_evidence = f"[U1] 用户上传文件\n{state.get('attachment', '')}" if state.get("attachment") else "无"
        prompt = f"""你是严谨的设备失效分析专家。请基于证据回答，不得把推测写成事实。

用户问题：{state['question']}

用户输入：
{user_evidence}

本地历史报告：
{format_local_hits(state['local_hits'])}

联网补充资料：
{format_web_hits(state.get('web_hits', [])) or '未使用联网资料'}

请用中文给出详细、可审计的专业报告，结构必须是：
1. 信息充分性：已有信息、缺失信息、结论置信度。
2. 失效分析：现象与关键数据、失效模式、证据支持的原因链、根因/促成因素、排除或待验证因素。
3. 建议：立即措施、进一步检验、纠正措施、预防措施，并标出优先级与验证标准。
4. 引用：每个关键事实用 [U1]、[L1] 或 [W1] 编号引用；网络事实附对应 URL。

不要输出隐藏思维链或自言自语；请输出足以复核结论的“判断依据和因果链”。
若证据不足，明确写“待验证”，并说明需要什么检测数据。"""
        result = llm.invoke(prompt)
        return {"answer": str(result.content), "trace": state["trace"] + ["完成失效分析与建议"]}

    graph = StateGraph(AgentState)
    graph.add_node("local_search", local_search)
    graph.add_node("decide_next_action", decide_next_action)
    graph.add_node("web_search", web_search)
    graph.add_node("answer", answer)
    graph.add_edge(START, "local_search")
    graph.add_edge("local_search", "decide_next_action")
    graph.add_conditional_edges(
        "decide_next_action",
        lambda state: "web_search" if state["use_web"] else "answer",
        {"web_search": "web_search", "answer": "answer"},
    )
    graph.add_edge("web_search", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


# CLI 同时接受普通问题和临时 Word/PDF/Markdown 文件。
def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG 失效分析")
    parser.add_argument("question", help="需要分析的问题")
    parser.add_argument("--file", type=Path, help="可选的 .docx/.pdf/.md 用户输入")
    args = parser.parse_args()

    settings = Settings.from_env()
    attachment = read_attachment(args.file) if args.file else ""
    result = build_graph(settings).invoke({"question": args.question, "attachment": attachment})
    print("\n".join(f"- {item}" for item in result["trace"]))
    print("\n" + result["answer"])


if __name__ == "__main__":
    main()
