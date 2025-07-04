import time
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain.tools import tool, Tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from openai import RateLimitError

def retry_call(func, *args, max_retries=10, delay=2, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            print(f"[retry_call] 触发速率限制，尝试第 {attempt}/{max_retries} 次重试，等待 {delay} 秒...")
            time.sleep(delay)
    raise Exception(f"[retry_call] 超过最大重试次数({max_retries})，调用失败。")

class GraphState(TypedDict):
    user_input: str
    subjects: str
    refined_goal: str
    structure: str
    content: str
    references: str

llm = ChatOpenAI(
    model="moonshot-v1-8k",
    api_key="sk-BCw9Q6mEEjPGoppI7QWOsuMQZKRTZRrYaTRiSge1HxjRX9fR", 
    base_url="https://api.moonshot.cn/v1",
)

stm = ConversationBufferMemory(memory_key="chat_history")
ltm = ConversationSummaryMemory(llm=llm, memory_key="summary")

@tool
def sl(q: str) -> str:
    """模拟从外部文献数据库搜索相关研究内容"""
    time.sleep(5)
    return f"[模拟] 文献检索结果：关于 '{q}' 的相关研究内容..."

@tool
def css(p: str) -> str:
    """模拟样本计算工具，根据给定参数返回推荐样本量"""
    time.sleep(5)
    return f"[模拟] 基于参数({p}) 计算推荐样本数量为 60"

@tool
def ek(t: str) -> str:
    """从输入文本中提取关键词"""
    time.sleep(5)
    return f"[模拟] 提取关键词：{', '.join(t[:30])}..."

tools = [
    Tool.from_function(sl, name="sl", description="文献检索工具"),
    Tool.from_function(css, name="css", description="样本量计算工具"),
    Tool.from_function(ek, name="ek", description="关键词提取工具")
]

td = {t.name: t for t in tools}

def ei(tc: dict) -> dict:
    r = {}
    for n, p in tc.items():
        r[n] = retry_call(td[n].invoke, p)
    return r

ps = ChatPromptTemplate.from_messages([
    ("system", "你是一个科研领域分析专家。请根据用户输入识别涉及的研究学科，并推理各自的角色和价值。输出格式：学科列表 + 每个学科的说明。"),
    ("human", "{user_input}")
])
sc = ps | llm

def sn(s: GraphState) -> GraphState:
    r = retry_call(sc.invoke, {"user_input": s["user_input"]})
    stm.save_context({"input": s["user_input"]}, {"output": r.content})
    return {**s, "subjects": r.content}

pg1 = ChatPromptTemplate.from_messages([
    ("system", "你是科研目标专家，请先列出可以细化的多个子方向或关键视角："),
    ("human", "{user_input}")
])
ge = LLMChain(llm=llm, prompt=pg1)

pg2 = ChatPromptTemplate.from_messages([
    ("system", "基于以下探索路径，整合为明确的研究目标与关键问题：{thoughts}"),
    ("human", "主题：{user_input}")
])
gs = LLMChain(llm=llm, prompt=pg2)

def grn(s: GraphState) -> GraphState:
    tr = retry_call(ge.invoke, {"user_input": s["user_input"]})
    tt = tr["text"]
    time.sleep(1)
    sr = retry_call(gs.invoke, {"user_input": s["user_input"], "thoughts": tt})
    retry_call(ltm.save_context, {"user_input": s["user_input"]}, {"refined_goal": sr["text"]})
    return {**s, "refined_goal": sr["text"]}

pp = ChatPromptTemplate.from_messages([
    ("system", "你是一个科研策划专家。请基于研究目标和相关学科知识，构建研究计划的结构大纲，使用树状层次。"),
    ("human", "研究目标：{refined_goal}\n相关学科：{subjects}")
])
pc = pp | llm

def pn(s: GraphState) -> GraphState:
    r = retry_call(pc.invoke, {"refined_goal": s["refined_goal"], "subjects": s["subjects"]})
    return {**s, "structure": r.content}

pw = ChatPromptTemplate.from_messages([
    ("system", "你是一个科研写作助手。请根据研究结构逐部分撰写建议书内容，如需外部数据可通过工具调用。"),
    ("human", "研究结构：{structure}\n已知信息：{subjects}")
])
wc = pw | llm

def wn(s: GraphState) -> GraphState:
    r = retry_call(wc.invoke, {"structure": s["structure"], "subjects": s["subjects"]})
    return {**s, "content": r.content}

rn = RunnableLambda(lambda s: {
    **s,
    "references": retry_call(ei, {"sl": s["user_input"]})["sl"]
})

cn = RunnableLambda(lambda s: {
    **s,
    "content": s["content"] + "\n\n" +
               retry_call(ei, {"css": "置信度95%，误差5%"})["css"]
})

g = StateGraph(GraphState)

g.add_node("sn", sn)
g.add_node("grn", grn)
g.add_node("pn", pn)
g.add_node("wn", wn)
g.add_node("rn", rn)
g.add_node("cn", cn)

g.set_entry_point("sn")
g.add_edge("sn", "grn")
g.add_edge("grn", "pn")
g.add_edge("pn", "wn")
g.add_edge("wn", "rn")
g.add_edge("rn", "cn")
g.add_edge("cn", END)

app = g.compile()

if __name__ == "__main__":
    ui = "我想申请一个‘大语言模型中的偏见度量和减少策略’项目"
    start_time = time.time()
    fs = app.invoke({"user_input": ui})
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n=== 最终生成内容 ===\n")
    print(fs["content"])

    print("\n=== 检索文献 ===\n")
    print(fs["references"])

    print("\n=== 学科分析 ===\n")
    print(fs["subjects"])

    print("\n=== 内容结构 ===\n")
    print(fs["structure"])

    print("\n=== 研究目标澄清 ===\n")
    print(fs["refined_goal"])
    
    print(f"\n=== 总运行时间: {elapsed_time:.2f} 秒 ===\n")