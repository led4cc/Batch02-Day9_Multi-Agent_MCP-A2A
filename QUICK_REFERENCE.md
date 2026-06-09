# Quick Reference - A2A Multi-Agent Codelab

## Lệnh Cơ Bản

### Setup
```bash
# Cài đặt dependencies
uv sync

# Copy environment file
cp .env.example .env
# Sau đó sửa .env, thêm OPENROUTER_API_KEY
# Nếu gặp OpenRouter 402, giảm OPENROUTER_MAX_TOKENS xuống 512 hoặc nạp thêm credit
```

### Chạy Stages (Standalone)
```bash
# Stage 1: Direct LLM
uv run python stages/stage_1_direct_llm/main.py

# Stage 2: RAG + Tools
uv run python stages/stage_2_rag_tools/main.py

# Stage 3: ReAct Agent
uv run python stages/stage_3_single_agent/main.py

# Stage 4: Multi-Agent (in-process)
uv run python stages/stage_4_milti_agent/main.py
```

### Chạy Stage 5 (Distributed)
```bash
# Start tất cả services
./start_all.sh

# Test hệ thống (terminal khác)
uv run python test_client.py

# Stop tất cả
# Ctrl+C trong terminal chạy start_all.sh
```

### Chạy Từng Service Riêng
```bash
# Registry
uv run python -m registry

# Customer Agent
uv run python -m customer_agent

# Law Agent
uv run python -m law_agent

# Tax Agent
uv run python -m tax_agent

# Compliance Agent
uv run python -m compliance_agent
```

---

## Cấu Trúc Code

### Stage 1: Direct LLM
```python
from common.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

llm = get_llm()
messages = [
    SystemMessage(content="You are an expert..."),
    HumanMessage(content="Question here"),
]
response = await llm.ainvoke(messages)
```

### Stage 2: Tools
```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description."""
    return "result"

llm_with_tools = llm.bind_tools([my_tool])
response = await llm_with_tools.ainvoke(messages)

# Execute tools
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = my_tool.invoke(tool_call["args"])
```

### Stage 3: ReAct Agent
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[my_tool])
result = await agent.ainvoke({"messages": [HumanMessage(content="...")]})
```

### Stage 4: Multi-Agent
```python
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from typing import TypedDict

class State(TypedDict):
    question: str
    result: str

def agent_node(state: State) -> dict:
    # Process state
    return {"result": "..."}

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile()
result = await app.ainvoke({"question": "..."})
```

### Stage 5: A2A Client
```python
from a2a.client import A2AClient
from a2a.types import AgentCard, Message, Part, TextPart

# Get agent card
card_resp = await http_client.get(f"{agent_url}/.well-known/agent.json")
agent_card = AgentCard.model_validate(card_resp.json())

# Create client
client = A2AClient(httpx_client=http_client, agent_card=agent_card)

# Send message
message = Message(
    role=Role.user,
    parts=[Part(root=TextPart(text="Question"))],
)
response = await client.send_message(request)
```

---

## Debugging

### Check Environment
```bash
# Xem biến môi trường
cat .env

# Test API key
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_KEY_HERE"
```

### Check Ports
```bash
# Xem port nào đang được dùng
lsof -i :10000
lsof -i :10100
lsof -i :10101
lsof -i :10102
lsof -i :10103

# Kill process
kill -9 <PID>
```

### View Logs
```bash
# Logs được in ra stdout
# Tìm trace_id để theo dõi request flow
grep "trace_id" <log_output>
```

### Common Errors

**"Could not reach Customer Agent"**
- Chưa start services: chạy `./start_all.sh`
- Port bị chiếm: check với `lsof -i :10100`

**"API key invalid"**
- Check `.env` file có đúng key không
- Key phải bắt đầu bằng `sk-or-v1-...`

**"Module not found"**
- Chưa cài dependencies: `uv sync`
- Sai Python version: cần 3.11+

**"Timeout"**
- LLM response chậm là bình thường (30-60s)
- Nếu quá 5 phút thì có vấn đề

---

## Concepts Chính

### LLM Basics
- **System Message**: Định nghĩa role của AI
- **Human Message**: Câu hỏi từ user
- **AI Message**: Response từ LLM
- **Tool Message**: Kết quả từ tool execution

### RAG (Retrieval-Augmented Generation)
- Tra cứu knowledge base trước khi trả lời
- Giúp LLM có thông tin real-time
- Giảm hallucination

### Tools / Function Calling
- LLM quyết định gọi function nào
- Function execute và trả về kết quả
- LLM synthesize kết quả thành câu trả lời

### ReAct Pattern
- **Re**asoning + **Act**ing
- Loop: Think → Act → Observe
- Tự động cho đến khi có answer

### Multi-Agent
- Nhiều agents chuyên môn hóa
- Mỗi agent có tools riêng
- Có thể chạy song song (parallel)

### A2A Protocol
- Chuẩn giao tiếp giữa agents
- Agent Card: metadata về agent
- Message: standardized format
- Registry: service discovery

---

## Architecture Patterns

### Single Agent
```
User → Agent (with tools) → Response
```

### Multi-Agent (In-Process)
```
User → Orchestrator Agent
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
  Agent1 Agent2 Agent3
    ↓      ↓      ↓
    └──────┼──────┘
           ↓
       Aggregator → Response
```

### Distributed A2A
```
User → Customer Agent (HTTP)
           ↓
       Registry (discover)
           ↓
       Law Agent (HTTP)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Tax Agent    Compliance Agent
(HTTP)           (HTTP)
    ↓             ↓
    └──────┬──────┘
           ↓
       Response
```

---

## Ports Reference

| Service | Port | URL |
|---|---|---|
| Registry | 10000 | http://localhost:10000 |
| Customer Agent | 10100 | http://localhost:10100 |
| Law Agent | 10101 | http://localhost:10101 |
| Tax Agent | 10102 | http://localhost:10102 |
| Compliance Agent | 10103 | http://localhost:10103 |

### Agent Card Endpoints
- Registry: http://localhost:10000/.well-known/agent.json
- Customer: http://localhost:10100/.well-known/agent.json
- Law: http://localhost:10101/.well-known/agent.json
- Tax: http://localhost:10102/.well-known/agent.json
- Compliance: http://localhost:10103/.well-known/agent.json

---

## Useful Links

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **A2A Protocol**: https://github.com/google/A2A
- **OpenRouter**: https://openrouter.ai/docs
- **Python Type Hints**: https://docs.python.org/3/library/typing.html

---

## Bài Tập Files

- `exercises/exercise_2_tools.py` - Thêm tools và knowledge base
- `exercises/exercise_4_multiagent.py` - Thêm privacy agent
- `exercises/SOLUTIONS.md` - Đáp án (xem sau khi làm xong!)

---

## Tips

1. **Đọc error messages cẩn thận** - thường có hint rõ ràng
2. **Check logs** - mỗi service in ra stdout
3. **Dùng print() để debug** - thêm vào code để xem state
4. **Test từng bước** - đừng viết nhiều code rồi mới chạy
5. **Hỏi bạn bè** - pair programming hiệu quả hơn
6. **Google error** - nhiều người gặp vấn đề tương tự
7. **Đọc code có sẵn** - stages/* là examples tốt

---

**Good luck! 🚀**
