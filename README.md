# Legal Multi-Agent System with A2A Protocol

A distributed legal advisory system where specialised AI agents collaborate using Google's [Agent-to-Agent (A2A) protocol](https://github.com/google/A2A). Built with **LangGraph**, **LangChain**, and the **a2a-sdk**, the project serves as both a working demo and a hands-on learning path — progressing from a simple LLM API call (Stage 1) to a fully distributed multi-agent network (Stage 5).

## Architecture

```
                     ┌─────────────────────┐
                     │  Registry Service   │  :10000
                     │  /register          │
                     │  /discover/{task}   │
                     └─────────┬───────────┘
                               │  (agents self-register on startup)
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
   Tax Agent :10102   Law Agent :10101    Compliance Agent :10103
          │                    │                     │
          └─────────► delegates in parallel ◄────────┘
                               │
                        Customer Agent :10100
                               │
                             User
```

**Customer Agent** receives a user question and delegates to the **Law Agent**, which analyses the legal aspects, then dispatches to **Tax Agent** and **Compliance Agent** in parallel via LangGraph's `Send` API. Results are aggregated into a comprehensive legal analysis.

All agent discovery is dynamic — agents register their capabilities with the **Registry** on startup and discover each other at runtime. No hardcoded URLs.

### Agent Details

| Agent | Port | LangGraph Pattern | Role |
|---|---|---|---|
| Customer Agent | 10100 | `create_react_agent` | Entry point — routes user questions to Law Agent |
| Law Agent | 10101 | Custom `StateGraph` | Orchestrator — analyses law, delegates in parallel |
| Tax Agent | 10102 | `create_react_agent` | Specialist — tax law, IRS, penalties, FBAR/FATCA |
| Compliance Agent | 10103 | `create_react_agent` | Specialist — SEC, SOX, FCPA, GDPR, AML |
| Registry | 10000 | FastAPI (not an agent) | Service discovery and agent registration |

### Request Flow

```
User question
  → Customer Agent: LLM detects legal domain, calls delegate tool
    → Registry: discover("legal_question") → Law Agent endpoint
    → Law Agent:
        [analyze_law]      LLM contract/tort analysis
        [check_routing]    LLM decides: needs_tax? needs_compliance?
        [call_tax]         ──→ Registry discover → Tax Agent (A2A)     ┐
        [call_compliance]  ──→ Registry discover → Compliance (A2A)    ├ parallel
        [aggregate]        Combines all analyses into final response   ┘
  → Customer Agent returns response to user
```

### Key Design Patterns

- **Dynamic discovery** — agents find each other through the Registry, not hardcoded URLs
- **Parallel delegation** — LangGraph `Send` API dispatches tax and compliance branches concurrently
- **Trace propagation** — `trace_id` and `context_id` flow through every A2A hop for debugging
- **Depth guards** — `MAX_DELEGATION_DEPTH = 3` prevents infinite delegation loops
- **Annotated reducers** — `Annotated[str, _last_wins]` handles parallel writes to shared state fields

## Tech Stack

| Layer | Choice |
|---|---|
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM provider | Any model via [OpenRouter](https://openrouter.ai) (OpenAI-compatible API) |
| A2A transport | [a2a-sdk](https://pypi.org/project/a2a-sdk/) |
| Registry | FastAPI + in-memory store |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## 📚 Codelab for Students

**Thời gian:** 2 giờ | **Ngôn ngữ:** Tiếng Việt

Codelab hướng dẫn từng bước xây dựng multi-agent system, từ cơ bản đến nâng cao:

- **[CODELAB.md](CODELAB.md)** - Hướng dẫn chi tiết cho sinh viên
- **[INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md)** - Hướng dẫn cho giảng viên
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Tài liệu tham khảo nhanh
- **[exercises/](exercises/)** - Bài tập thực hành với skeleton code
- **[exercises/SOLUTIONS.md](exercises/SOLUTIONS.md)** - Đáp án chi tiết

### Lộ Trình Học

```
Stage 1: Direct LLM (20 phút)
    ↓
Stage 2: RAG + Tools (30 phút)
    ↓
Stage 3: ReAct Agent (25 phút)
    ↓
Stage 4: Multi-Agent (30 phút)
    ↓
Stage 5: Distributed A2A (30 phút)
    ↓
Tổng kết & Q&A (15 phút)
```

**Bắt đầu:** Đọc [CODELAB.md](CODELAB.md)

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai) API key

### Setup

```bash
# Clone and install
git clone <repo-url>
cd legal_multiagent
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your OpenRouter API key
```

### Run the Full System (Stage 5)

```bash
# Start all 5 services (registry + 4 agents)
./start_all.sh

# In another terminal, send a test question
uv run python test_client.py
```

### Run Individual Stage Demos

No servers needed — each demo runs as a standalone script:

```bash
uv run python stages/stage_1_direct_llm/main.py
uv run python stages/stage_2_rag_tools/main.py
uv run python stages/stage_3_single_agent/main.py
uv run python stages/stage_4_multi_agent/main.py
```

## LLM Evolution Stages

The `stages/` folder contains progressive demos that build from simple to complex, matching the roadmap in `docs/10_llm_roadmap.svg`:

| Stage | Name | What It Demonstrates |
|---|---|---|
| **1** | Direct LLM Calling | Stateless prompt → response. No tools, no memory. |
| **2** | LLM + RAG / Tools | Tool calling with a keyword-match knowledge base and damage calculator. Manual single-pass orchestration. |
| **3** | Single Agent (ReAct) | Autonomous Think → Act → Observe loop via `create_react_agent`. Agent decides which tools to call and when. |
| **4** | Multi-Agent (In-Process) | Multiple specialised agents with parallel execution via `StateGraph` + `Send` API. Same topology as Stage 5 but in a single process. |
| **5** | Distributed A2A (This Project) | Full distributed system — each agent is an independent HTTP service communicating via A2A protocol with dynamic discovery. |

Each stage's folder includes an `architecture.svg` diagram and a self-contained `main.py`.

## Project Structure

```
legal_multiagent/
├── start_all.sh               # Launches all services in correct order
├── test_client.py             # E2E test client
├── pyproject.toml             # Dependencies (uv-managed)
├── .env.example               # Required environment variables
│
├── common/                    # Shared utilities
│   ├── llm.py                 # get_llm() → ChatOpenAI via OpenRouter
│   ├── a2a_client.py          # delegate() — A2A message sending
│   └── registry_client.py     # discover() / register() — Registry API
│
├── registry/                  # Service discovery (port 10000)
├── customer_agent/            # Entry point agent (port 10100)
├── law_agent/                 # Legal orchestrator (port 10101)
├── tax_agent/                 # Tax specialist (port 10102)
├── compliance_agent/          # Compliance specialist (port 10103)
│
├── stages/                    # Progressive learning demos (1-4)
│   ├── stage_1_direct_llm/
│   ├── stage_2_rag_tools/
│   ├── stage_3_single_agent/
│   └── stage_4_multi_agent/
│
└── docs/                      # Architecture diagrams (SVG)
```

Each agent module follows the same structure:
- **`graph.py`** — LangGraph graph definition (all agent logic)
- **`agent_executor.py`** — Bridge between A2A SDK and LangGraph
- **`__main__.py`** — Server bootstrap, agent card, registration

## Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | (required) |
| `OPENROUTER_MODEL` | Model identifier | `anthropic/claude-sonnet-4-5` |
| `OPENROUTER_MAX_TOKENS` | Maximum output tokens per LLM response | `1000` |
| `REGISTRY_URL` | Registry service URL | `http://localhost:10000` |

The model is swappable to any OpenRouter-supported model (e.g., `openai/gpt-4o`, `google/gemini-2.0-flash`).

If OpenRouter returns `402` with a message like `requested up to 64000 tokens`, the API key is working but the request is too expensive for the account's current credits. Lower `OPENROUTER_MAX_TOKENS` (for example, `512`) or add credits in OpenRouter.

## Documentation Diagrams

The `docs/` folder contains SVG architecture diagrams:

| Diagram | Topic |
|---|---|
| `01_why_multiagent` | Why multi-agent over monolithic LLMs |
| `02_a2a_vs_traditional` | A2A protocol vs traditional multi-agent |
| `03_a2a_protocol` | A2A protocol technical details |
| `04_system_architecture` | Full system architecture |
| `05_law_agent_graph` | Law Agent StateGraph deep dive |
| `06_request_flow` | End-to-end request flow with trace propagation |
| `07_a2a_intro` | Introduction to A2A protocol |
| `08_a2a_core_concepts` | A2A core concepts (Agent Cards, Tasks, Parts) |
| `09_a2a_interaction_flow` | A2A interaction flow patterns |
| `10_llm_roadmap` | LLM evolution roadmap (Stages 1–5) |
