# AutoGen Agent Framework — Deep Research Report

**Research Date:** May 11, 2026  
**Subject:** AutoGen (Microsoft Research / AG2) — Architecture, Adoption, Use Cases, and Competitive Landscape

---

## Table of Contents

1. [What Is AutoGen?](#1-what-is-autogen)
2. [How It Works — Architecture Deep Dive](#2-how-it-works--architecture-deep-dive)
3. [The AutoGen Ecosystem](#3-the-autogen-ecosystem)
4. [Who Uses AutoGen?](#4-who-uses-autogen)
5. [Industries and Use Cases](#5-industries-and-use-cases)
6. [Why People Choose AutoGen](#6-why-people-choose-autogen)
7. [Why People Don't Choose AutoGen](#7-why-people-dont-choose-autogen)
8. [AutoGen vs Competing Frameworks](#8-autogen-vs-competing-frameworks)
9. [Community and Market Position](#9-community-and-market-position)
10. [Pricing](#10-pricing)
11. [Summary and Verdict](#11-summary-and-verdict)
- [Sources](#sources)

---

## 1. What Is AutoGen?

AutoGen is a Python framework for building multi-agent AI systems using a conversational coordination model — agents talk to each other, in natural language, to collaboratively solve tasks. It was pioneered by **Microsoft Research** in 2023 and quickly became the most academically influential agent framework in the field, with its founding paper accumulating over 1,300 citations. AutoGen's core proposition was straightforward and radical in equal measure: instead of designing explicit workflows, you let agents negotiate the solution through conversation. An assistant agent proposes code; a user proxy agent executes it; a critic agent reviews the result; a manager agent decides what to do next. The structure emerges from the dialogue, not from developer-defined graphs or configuration files.

The framework was created by **Chi Wang, Qingyun Wu**, and colleagues at Microsoft Research, with the founding paper published in August 2023 (arXiv: 2308.08155). Its release landed at precisely the right moment — GPT-4 had made LLM-driven reasoning practical, and the research community was racing to explore what multi-agent collaboration could look like. AutoGen became the reference implementation for that exploration, used in academic labs, research organizations, and early enterprise experiments worldwide.

**The governance situation as of 2026 is complex and must be stated clearly upfront:**

In **September 2024**, Chi Wang, Qingyun Wu, and the original AutoGen team departed Microsoft Research. In **November 2024**, they forked the codebase and established **AG2** (`ag2ai/ag2`, available at ag2.ai) under a new community governance model, describing it as the open-source AgentOS and the direct continuation of AutoGen's original vision. In **January 2025**, Microsoft released **AutoGen v0.4** — a near-complete architectural rewrite of the `microsoft/autogen` repository. In **October 2025**, Microsoft announced the **Microsoft Agent Framework**, which merges AutoGen and Semantic Kernel into a single enterprise-grade product. Both Microsoft AutoGen and Semantic Kernel formally entered **maintenance mode** at that point — receiving security patches and bug fixes, but no new features. Microsoft now directs new projects toward Microsoft Agent Framework.

**This report covers AutoGen as a framework lineage**: the architecture, design philosophy, strengths, and weaknesses apply to both the Microsoft-maintained v0.4 codebase and the community-maintained AG2 fork, which preserves backward compatibility with v0.2's API while developing toward its own v1.0. Where the two diverge meaningfully, both are addressed.

The core mental model is **conversation as the coordination primitive**. Where LangGraph uses a state graph, Haystack uses a component pipeline, and Pydantic AI uses typed function calls, AutoGen uses multi-agent dialogue. This makes AutoGen uniquely natural for tasks where the solution path is unknown — research, code generation with iteration, creative tasks — and uniquely awkward for tasks where precision and determinism are required.

AutoGen (both repositories) is **MIT licensed**, fully open source. The original `microsoft/autogen` repo is in maintenance mode. AG2 (`ag2ai/ag2`) is under active development.

**Headline metrics (as of May 2026):** AG2 repository: 50,000+ GitHub stars, 20,000+ Discord community members; original AutoGen paper: ~1,300 citations on Google Scholar — one of the most cited AI agent papers published; `microsoft/autogen`: in maintenance mode since October 2025; Azure AI Foundry Agent Service (AutoGen's enterprise successor path): 10,000+ organizations using managed deployments.

> *"AutoGen enables the next generation of LLM applications by offering a unified multi-agent conversation framework... agents in AutoGen are conversable — they can generate, receive, and reply to messages."*  
> — AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation, Wu et al., 2023

In a single sentence: AutoGen is the most academically influential multi-agent framework ever published — a conversation-first approach to agent coordination that pioneered the category and now exists in two lineages, one in Microsoft maintenance mode and one as the community-governed AG2, with Microsoft's production path having moved on to the Microsoft Agent Framework.

---

## 2. How It Works — Architecture Deep Dive

### The Two Architectures: v0.2 and v0.4 / AG2

AutoGen has two meaningfully different architectures in active use, and understanding both is necessary to understand the framework's present state.

**AutoGen 0.2** (the original architecture, preserved in AG2) remains the version most practitioners know and the one responsible for AutoGen's cultural footprint. It is simpler, less formally typed, and more directly accessible. **AutoGen 0.4** (the Microsoft rewrite, now in maintenance) introduced a layered async architecture. AG2's ongoing development incorporates v0.4's event-driven core while maintaining v0.2 API compatibility for existing users.

### Core Primitives (v0.2 / AG2 compatible)

**ConversableAgent** is the base class from which all AutoGen agents inherit. Every agent is conversable: it can send messages, receive messages, and decide how to respond. The response decision — whether to use an LLM, execute code, ask a human, or apply a custom function — is configurable per agent. This single base class supporting multiple response modes is the design decision that makes AutoGen's flexibility possible and its predictability challenging.

**AssistantAgent** is a `ConversableAgent` configured to use an LLM for response generation. It receives messages, reasons about them using the configured model, and produces text responses that may include code blocks, plans, or natural language. By default, it does not execute code — it produces it. Multiple AssistantAgents in a conversation can represent different expertise, different prompting strategies, or different model providers.

**UserProxyAgent** is a `ConversableAgent` configured to represent a human or to execute code. In automated pipelines, it executes code blocks produced by AssistantAgents in a local or Docker sandbox, returns the output (including errors), and continues the conversation. This is AutoGen's most distinctive primitive: a code-execution agent that closes the loop between "LLM writes code" and "code runs and returns results." The iteration loop — write code, run it, observe output, debug, try again — happens through conversation with no developer scaffolding required.

**GroupChat** is the multi-agent coordination mechanism. Multiple agents are added to a `GroupChat` object, and a `GroupChatManager` (itself an LLM-backed agent) decides which agent speaks next at each turn based on the conversation context. This is emergent routing: the manager reads the current conversation and selects the most appropriate next speaker, rather than following a developer-defined transition table. It is impressively powerful when it works and impressively difficult to control when it does not — the wrong agent selection is the most common production failure mode in AutoGen applications.

### The v0.4 / AG2 Architecture Layers

The AutoGen 0.4 rewrite introduced a formal three-layer architecture that AG2 is building toward in its v1.0:

The **Core layer** (`autogen-core`) implements an actor-model runtime with asynchronous message passing. Agents are actors that handle typed messages and produce typed outputs. The runtime supports both direct messaging (like RPC, for deterministic point-to-point communication) and broadcast to topics (like pub-sub, for event-driven fan-out). This layer enables distributed agent deployments where agents run in separate processes or on separate machines.

The **AgentChat layer** (`autogen-agentchat`) sits above Core and provides the high-level conversational API familiar from v0.2 — agents, teams, tasks, message histories. It is task-driven rather than message-driven: you give a team a task, it runs until completion or a stopping condition, and returns a result. This is the layer most application developers interact with.

The **Extensions layer** provides pluggable components for specific capabilities: code executors (local, Docker, Jupyter), model clients (OpenAI, Anthropic, Gemini, Ollama), and tool integrations.

### The Agent Loop

When a task is initiated, AutoGen enters a conversation loop: the GroupChatManager (or team orchestrator in v0.4) selects an agent, sends the current conversation context to that agent, receives its response (which may include tool calls, code blocks, or natural language), appends the response to the conversation history, and repeats. The loop terminates when an agent produces a TERMINATE signal, a maximum number of turns is reached, or a custom stopping condition is met. Every agent at every turn receives the full accumulated conversation history — which is both the strength (full context) and the cost driver (quadratically growing token consumption in long conversations).

### Code Execution

AutoGen's code execution capability is its most distinctive feature. The `UserProxyAgent` can be configured to automatically execute Python (or shell) code found in LLM responses, return the output to the conversation, and continue the loop. This enables a natural iterative development pattern: the AssistantAgent writes code, the UserProxyAgent runs it, errors are returned as conversation messages, the AssistantAgent revises, and so on. Execution can run locally, inside a Docker container (for isolation), or inside a Jupyter kernel (for data science workflows). This code execution loop is what made AutoGen the framework of choice for code generation research and agentic software development experiments.

### Minimal Code Example

```python
from autogen import AssistantAgent, UserProxyAgent

llm_config = {"model": "gpt-4o", "api_key": "..."}

# LLM-backed assistant that writes code and plans
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

# Proxy that executes code and relays results
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",      # fully automated
    code_execution_config={"work_dir": "coding"},
    max_consecutive_auto_reply=10,
)

# Start the conversation — agents iterate until TERMINATE
user_proxy.initiate_chat(
    assistant,
    message="Write a Python script that fetches the top 5 Hacker News stories.",
)
```

Two agents, a task, and a loop. The assistant writes the code; the proxy runs it; errors return as messages; the assistant fixes them. No graph, no pipeline, no explicit error handling — just conversation.

### Multi-Agent Patterns

Beyond two-agent loops, AutoGen supports several multi-agent patterns. **GroupChat** with a manager allows N agents to collaborate in a shared conversation. **Sequential chats** chain multiple two-agent conversations, passing outputs from one as inputs to the next. **Nested chats** allow one agent to internally spin up a sub-conversation to solve a subtask before responding to the outer conversation. In the v0.4 / AG2 architecture, **Teams** formalize these patterns: `RoundRobinGroupChat` (each agent speaks in turn), `SelectorGroupChat` (LLM picks the next speaker), and `Swarm` (agents hand off based on explicit handoff declarations).

### Memory and State

Within a conversation, state is the conversation history itself — the accumulated list of messages. There is no separate state schema or typed state object (a key difference from LangGraph). Across conversations, AutoGen 0.2 has no built-in persistence. The v0.4 / AG2 architecture adds a `ChatHistory` abstraction and externalized state, but cross-session memory remains an application-layer responsibility.

---

## 3. The AutoGen Ecosystem

### AutoGen Studio

**AutoGen Studio** is a no-code GUI for prototyping and debugging multi-agent AutoGen applications. It provides a browser-based interface for defining agents, composing them into teams, running multi-agent conversations interactively, and inspecting message-by-message execution traces. AutoGen Studio was rebuilt on the v0.4 API and is available as a standalone Python package (`pip install autogenstudio`). It is not a production deployment platform — it is a development and experimentation tool, analogous to what LangSmith's Playground provides for LangGraph, but with fewer production management features. For non-technical stakeholders who want to experiment with AutoGen without writing code, Studio is the entry point.

### .NET / C# SDK

AutoGen provides a **.NET SDK** (via the `AutoGen.NET` package), making it one of very few agent frameworks with first-class support for C# and .NET languages. This is a meaningful differentiator for enterprise engineering teams standardized on Microsoft's technology stack — C# applications can define agents, wire multi-agent conversations, and invoke tools natively without a language boundary. The .NET SDK supports the same agent primitives (AssistantAgent, UserProxyAgent, GroupChat) and integrates with Azure OpenAI natively.

### AG2 Open Ecosystem

Under the AG2 fork, the community has developed native integrations with major cloud and AI infrastructure providers. AG2 supports the **Agent-to-Agent (A2A)** protocol for framework-agnostic inter-agent communication, the **Model Context Protocol (MCP)** for tool server integration, and the **AG-UI** protocol for building streaming, event-driven user interfaces on top of agent conversations. Cloud integrations include AWS, IBM Watsonx, Databricks, and Google Cloud. The A2A integration enables AG2 agents to communicate natively with agents built in LangGraph, CrewAI, Semantic Kernel, Pydantic AI, and other A2A-compliant frameworks — making AG2 a viable orchestration layer in heterogeneous multi-framework deployments.

### Azure AI Foundry Agent Service

For teams that want managed production deployment of AutoGen-lineage agents, **Azure AI Foundry Agent Service** is Microsoft's answer. It provides hosted agent execution, state management, tool integrations (Azure Cognitive Search, Microsoft Graph, SharePoint, Elastic, Redis), and enterprise governance features (RBAC, audit logging, compliance) on Azure infrastructure. Over 10,000 organizations use Azure AI Foundry, though this figure encompasses the full Foundry offering rather than AutoGen-specific deployments. KPMG, BMW, and Fujitsu are among named Foundry enterprise customers. For new Microsoft-ecosystem projects, Azure AI Foundry is the recommended path rather than self-managed AutoGen.

### Observability

AutoGen integrates with **AgentOps** for production monitoring — tracking agent runs, token usage, tool invocations, and conversation traces. AG2 adds OpenTelemetry-compatible tracing as of its v0.4-compatible releases. Neither matches the depth of LangSmith (for LangGraph) or Logfire (for Pydantic AI) as a purpose-built agent observability platform, but the integrations cover basic production monitoring needs.

### Multi-LLM Provider Support

AG2 supports OpenAI, Anthropic Claude, Google Gemini, Alibaba DashScope, Ollama, and any OpenAI-compatible API. Model configuration is agent-level, not framework-level — different agents in the same GroupChat can use different model providers, enabling cost-optimized architectures where cheap models handle routing or summarization and expensive models handle generation.

---

## 4. Who Uses AutoGen?

| **Company / Organization** | **Use Case** |
|---|---|
| **Microsoft Research** | Original creator; internal multi-agent research, evaluation frameworks (AutoGenBench), and prototyping for MSR projects across code generation, data analysis, and scientific reasoning tasks |
| **KPMG** | Production agent deployments on Azure AI Foundry (AutoGen-lineage) for audit automation, document analysis, and enterprise knowledge management |
| **BMW** | Azure AI Foundry Agent Service for manufacturing operations automation and enterprise knowledge workflows |
| **Fujitsu** | Production workloads on Azure AI Foundry Agent Service for enterprise IT and operational AI use cases |
| **RevolutionAI** | Production AutoGen deployment for multi-agent workflows in their AI product suite |
| **Alice Labs** | Production AutoGen deployments for clients in financial services, media, and public sector — code generation and data analysis workflows |
| **Academic institutions (global)** | AutoGen is the most widely cited agent framework in academic research; used in hundreds of labs for multi-agent research, LLM evaluation, and agentic system design experiments |
| **Enterprise ISVs (Azure ecosystem)** | Independent software vendors building on Azure AI Foundry leverage AutoGen-lineage orchestration for customer-facing AI features in accounting, biotech, retail, healthcare, and supply chain verticals |

*Note: AutoGen's roots in Microsoft Research mean its most documented adoption is in research settings and through Microsoft's enterprise channel (Azure AI Foundry) rather than via direct published case studies typical of commercial frameworks. Named enterprise case studies are less abundant than for CrewAI or LangGraph.*

---

## 5. Industries and Use Cases

### Software Development and Code Generation

Code generation with iterative self-correction is AutoGen's clearest competitive strength and most documented use case. The UserProxyAgent's code execution loop — write code, run it, observe errors, revise, repeat — maps perfectly to the software development workflow. Research teams at Microsoft and academic institutions have used AutoGen to build multi-agent coding systems: one agent writes code, another reviews it, a third tests it, a fourth proposes optimizations. This pattern has been applied to competitive programming, data analysis pipeline generation, automated debugging, and test generation. Alice Labs reports production code generation agent deployments in client software development workflows.

### Data Analysis and Research Automation

AutoGen's conversational architecture is well-suited to data analysis tasks where the analysis plan needs to adapt based on intermediate results. An analyst agent writes Python code to explore a dataset; an executor agent runs it; results are shared with the conversation; the analyst revises the approach. This iterative loop does not require a pre-defined workflow structure — the analysis plan emerges from what the data reveals. Academic research teams use AutoGen extensively for this pattern, and financial services practitioners cite it for quantitative analysis and model evaluation.

### Financial Services and Quantitative Analysis

The Alice Labs financial services deployments, and broader reports of AutoGen use in banking and fintech, follow the code-based analysis pattern: agents that write and execute Python against financial datasets, generate reports from structured data, and iterate on analysis until outputs meet quality criteria. The GroupChat pattern is also used for multi-perspective financial document review — one agent reviews for risk, another for compliance, a third for opportunity — with a manager agent synthesizing the views. The lack of deterministic routing is a concern for high-stakes financial workflows, which is one reason Microsoft is steering this segment toward Microsoft Agent Framework.

### Healthcare and Biotech Research

The AutoGen academic paper and subsequent research demonstrated multi-agent biomedical use cases: literature review automation (agents search, summarize, cross-reference, and synthesize research papers), drug interaction analysis pipelines, and clinical trial data extraction. Biotech and healthcare R&D organizations are among the enterprise verticals cited in Azure AI Foundry adoption. The research-exploration nature of biotech work — where the right analytical approach often isn't known in advance — maps naturally onto AutoGen's emergent conversation model.

### Government and Public Sector

Government agencies appear in the Azure AI Foundry enterprise customer base as part of the AutoGen-lineage deployment pathway. Document analysis, regulatory compliance checking, and knowledge management for large government document repositories are the common patterns. The public sector's preference for Microsoft Azure infrastructure (driven by existing contracts, FEDRAMP compliance, and familiarity) makes the Azure AI Foundry path the natural commercial route.

### Academic and Research Computing

AutoGen's ~1,300 citation count is not incidental — it reflects genuine adoption as the reference implementation for multi-agent AI research. University labs use AutoGen for: agent behavior evaluation, multi-agent debate experiments, LLM benchmarking, and research into agent coordination patterns. The `AutoGenBench` evaluation framework — built on top of AutoGen by Microsoft Research — formalizes this academic use case as a structured benchmarking harness. This research community is AutoGen's most loyal constituency and the one that AG2 explicitly serves with its community governance model.

---

## 6. Why People Choose AutoGen

### The Natural Fit for Code Generation and Iteration

No other framework handles the write-run-debug-revise loop as naturally as AutoGen. The UserProxyAgent's automated code execution, combined with an AssistantAgent that can interpret error messages and revise its approach, produces a self-correcting coding system with fewer lines of agent scaffolding than any alternative. For applications where the core value is generating, testing, and iterating on code — data pipelines, analysis scripts, test suites, deployment automation — the AutoGen pattern is the least-friction path.

### Conversation as the Workflow Model for Open-Ended Tasks

When the solution path to a problem is genuinely unknown — when you need agents to explore, debate, propose, and critique before converging on an answer — AutoGen's conversational model is more natural than a pre-defined graph. Requiring developers to define a graph implies they know the workflow structure in advance; AutoGen allows the workflow to emerge from the agents' dialogue. This is not a workaround — it is the right model for research, brainstorming, and open-ended exploration tasks where rigid workflows would constrain the agents' problem-solving.

### The Largest Academic and Research Community

AutoGen's ~1,300 research citations and its origins in Microsoft Research give it a constituency that no other agent framework has matched: an active academic community that produces benchmarks, evaluations, novel agent patterns, and published evidence of what works. Teams operating at the intersection of production AI and research — R&D organizations, AI labs, university spin-outs — benefit from this community in the form of peer-reviewed evaluation evidence, novel architectural patterns, and a talent pool trained on AutoGen's concepts.

### Multi-Provider Agent Configurations

Each agent in an AutoGen GroupChat can use a different LLM. A fast, cheap model (GPT-4o-mini, Claude Haiku) can handle the GroupChatManager's speaker selection, while expensive frontier models (GPT-4o, Claude Opus) handle substantive generation. This per-agent model configuration is not unique to AutoGen but is natively supported and practically useful — the architecture encourages thinking about each agent's cost-capability tradeoff independently.

### .NET / C# Support Is Unique

AutoGen is one of the only production-capable agent frameworks with a full-featured .NET SDK. For organizations standardized on C# — particularly in finance, enterprise software, and legacy Microsoft technology stacks — this eliminates a language boundary that forces Python for agent development. The .NET SDK supports the same agent primitives and Azure OpenAI integration as the Python version, enabling C# teams to build multi-agent systems without adopting a new language runtime.

### No-Code Studio for Experimentation

AutoGen Studio lowers the barrier for non-engineer stakeholders to prototype and evaluate multi-agent configurations without writing code. Research leads, product managers, and technical program managers can experiment with agent compositions, observe conversation flows, and evaluate output quality before committing engineering resources to a production implementation. This experimentation surface is undervalued by pure engineering teams but meaningful for organizations where AI system design involves non-technical stakeholders.

---

## 7. Why People Don't Choose AutoGen

### Microsoft AutoGen Is in Maintenance Mode

The `microsoft/autogen` repository entered maintenance mode in October 2025. No new features will be added. Microsoft has explicitly redirected new projects to the Microsoft Agent Framework. Teams starting a new project in 2026 who want to stay aligned with Microsoft's supported product path should not build on AutoGen directly. The v0.4 architecture, while technically sound, represents a transitional state between the original AutoGen design and Microsoft's Agent Framework successor — not a platform with a long-term roadmap of its own.

### AG2's Governance and Longevity Are Unproven

AG2 is the most viable continuation of the AutoGen design philosophy, but it is a community-governed fork run by volunteers and a small core team led by the original creators (who are no longer at Microsoft — Chi Wang moved to Google DeepMind). The long-term sustainability of a volunteer-governed open-source project competing with well-funded commercial frameworks (LangGraph via LangChain, CrewAI via CrewAI Inc.) is genuinely uncertain. Teams making a multi-year framework commitment should weigh AG2's community governance model honestly against alternatives with clearer commercial backing.

### Non-Determinism Makes Production Debugging Painful

AutoGen's emergent coordination model — the GroupChatManager selects the next speaker based on LLM reasoning — means that the same task run twice may follow different conversation paths and reach different conclusions. When an AutoGen system misbehaves in production, there is no deterministic conversation trace to replay and diagnose. Debugging requires prompt engineering on the manager's instructions rather than changes to explicit routing code. For teams that need to reproduce bugs, audit agent decisions, or pass compliance reviews of AI decision-making, this non-determinism is a structural problem, not a configuration issue.

### Token Costs Scale Quadratically with Conversation Length

Every agent turn in a GroupChat involves sending the full accumulated conversation history to the LLM. A six-agent GroupChat running for ten rounds sends 60 LLM calls, each with a growing context window. Benchmarks show AutoGen consuming roughly 3× more tokens than more efficient frameworks for equivalent tasks. At high volume — customer-facing applications, high-frequency data processing, real-time use cases — this cost profile is prohibitive. AutoGen was designed for research-scale experimentation, not cost-optimized production at scale.

### No Built-In Durable Persistence

AutoGen has no native checkpoint-based state persistence. A multi-agent conversation that fails mid-execution — due to an API timeout, a code execution error, or a process crash — must restart from the beginning. For long-running research tasks or complex analysis workflows that may run for minutes or hours, this means any failure restarts the entire computation. LangGraph's durable execution with configurable checkpoints is the alternative for teams where long-running reliability is a requirement.

### Complex Orchestration Is Harder Than Graph-Based Frameworks

Expressing conditional branching ("if the code has no errors, proceed to deployment; otherwise, return to the developer agent"), parallel execution ("run the security review and the performance review simultaneously"), or explicit state transitions is harder in AutoGen's conversational model than in LangGraph's graph model. You can approximate these patterns through GroupChatManager prompt engineering or custom speaker selection functions, but the result is harder to read, test, and maintain than an explicit graph. Teams building applications with complex, well-understood workflow structures will find LangGraph more tractable.

### Smaller Practical Ecosystem Than LangChain or LangGraph

Despite the massive GitHub star count (which reflects academic and research interest), AutoGen's practical ecosystem of production templates, pre-built integrations, and community-tested patterns is smaller than LangChain's. The framework's academic orientation means most community content addresses research use cases rather than production engineering problems. Teams looking for "how do I connect AutoGen to Salesforce CRM" or "what's the right pattern for human-in-the-loop AutoGen approval" will find fewer answers than for LangGraph equivalents.

---

## 8. AutoGen vs Competing Frameworks

| **Framework** | **Core Metaphor** | **Best For** | **Time-to-Demo** | **Production Maturity** |
|---|---|---|---|---|
| **AutoGen / AG2** | Conversational multi-agent collaboration | Code generation, research, open-ended exploration | Low (10–20 min) | Medium — maintenance mode (Microsoft); active (AG2) |
| **LangGraph** | State graph, nodes and edges | Complex stateful workflows, deterministic routing, human-in-the-loop | Medium-high (45–90 min) | High (since 2023) |
| **CrewAI** | Role-based agent crews | Rapid prototyping, role-delegation, content generation | Low (10–20 min) | Medium-high |
| **Pydantic AI** | Type-safe agents, dependency injection | Python-native teams, multi-provider, production testability | Low (15–25 min) | Medium-high (v1.0 Sept 2025) |
| **Microsoft Agent Framework** | Graph workflows + enterprise orchestration | Azure teams, Semantic Kernel successor users, enterprise governance | Medium (30–60 min) | High (GA Q1 2026) |
| **Haystack** | Component pipeline graph | Retrieval-heavy, document-centric enterprise AI | Medium (30–60 min) | High (since 2020) |
| **OpenAI Agents SDK** | Agents, handoffs, guardrails | OpenAI-committed teams, voice agents, speed | Very low (10–20 min) | Medium-high (March 2025) |
| **LlamaIndex** | Data pipeline + retrieval-first agents | Document-heavy RAG, enterprise data ingestion, knowledge agents | Low-medium (20–40 min) | High for RAG; medium for orchestration |
| **Mastra** | TypeScript-first composable agents | JS/TS-primary teams, Node.js environments | Low (15–20 min) | Medium (v1.0 Jan 2026) |

### AutoGen vs. LangGraph

AutoGen and LangGraph are the most frequently compared frameworks and represent opposing philosophies on the most fundamental question in agent design: should the workflow structure be explicit or emergent? LangGraph says explicit — define your graph, your state schema, your edge conditions in code. AutoGen says emergent — let the agents negotiate the workflow through conversation. LangGraph is predictable, debuggable, and expensive to design upfront. AutoGen is flexible, fast to prototype, and expensive to control precisely.

**Choose AutoGen / AG2 when:** the task structure is genuinely unknown in advance; the primary value is iterative reasoning or code generation where agents need to adapt based on results; or rapid exploration of multi-agent patterns is more important than production determinism.

**Choose LangGraph when:** the workflow structure is known; durable execution across failures is required; deterministic routing is a compliance or audit requirement; or LangSmith's debugging and time-travel replay would materially reduce development time.

The differentiating dimension is **emergent flexibility vs. deterministic control**. Both frameworks are in active use in production; the choice depends on whether unpredictability in your agent system is a feature or a bug.

### AutoGen vs. CrewAI

These two share ergonomic similarities — both are relatively easy to get started with, both support multi-agent patterns — but their coordination models are different. CrewAI uses role-based crews where each agent has a declared role, goal, and backstory, and tasks are assigned to the crew to complete collaboratively. AutoGen uses conversational exchange with no mandatory role declarations, relying on prompts and GroupChatManager logic to coordinate. CrewAI's role model is more readable to non-engineers; AutoGen's conversational model is more flexible for technical tasks.

**Choose AutoGen / AG2 when:** code execution is part of the workflow; agents need to adapt their roles dynamically based on conversation context; or .NET support is required.

**Choose CrewAI when:** the workflow maps naturally to a team of humans with distinct roles; non-engineers need to read or configure agent behavior; or fast time-to-demo with role clarity is the priority.

The differentiating dimension is **conversational flexibility vs. role-based clarity**. CrewAI produces demos faster; AutoGen handles code execution more naturally.

### AutoGen vs. Microsoft Agent Framework

This comparison is unique: Microsoft Agent Framework is explicitly positioned as AutoGen's successor. AutoGen's design philosophy (conversational multi-agent) is preserved in Microsoft Agent Framework's high-level agent patterns, but wrapped in Semantic Kernel's production infrastructure — session state management, middleware, telemetry, Azure Durable Functions checkpointing. If you are currently on AutoGen and your deployment is Azure, migrating to Microsoft Agent Framework is Microsoft's recommended path with a published migration guide.

**Choose AutoGen / AG2 when:** you are not on Azure; you are committed to community-governed open source; or you need AG2's active development roadmap rather than Microsoft's maintenance-mode framework.

**Choose Microsoft Agent Framework when:** your infrastructure is Azure; you need enterprise compliance features (SOC 2, FEDRAMP, audit logging); Semantic Kernel's middleware model fits your architecture; or long-term Microsoft support SLAs are a procurement requirement.

The differentiating dimension is **community open source vs. enterprise Microsoft support**. This is the most consequential choice for existing AutoGen users.

### AutoGen vs. Pydantic AI

These two frameworks serve fundamentally different audiences with different engineering values. AutoGen's users tend to be research-oriented or exploration-oriented teams where flexibility and conversation-driven emergence are features. Pydantic AI's users tend to be production-engineering teams where type safety, determinism, and testability are features. The two rarely compete directly for the same use case.

**Choose AutoGen / AG2 when:** multi-agent conversation patterns, code execution, and open-ended task exploration are the primary requirements.

**Choose Pydantic AI when:** structured output validation, multi-provider flexibility, dependency injection for testing, and production code quality are the primary requirements.

The differentiating dimension is **research flexibility vs. engineering rigor**. Both are Python-native; they disagree on what "production-ready" means.

### AutoGen vs. OpenAI Agents SDK

Both frameworks make getting to a first working multi-agent prototype fast, but they come from different lineages and make different tradeoffs. The OpenAI Agents SDK is explicitly provider-committed — it provides the tightest possible integration with OpenAI's platform (hosted tools, native tracing, Realtime voice agents) but ties you to OpenAI models. AutoGen / AG2 is provider-agnostic and was designed for research and exploration, with code execution and multi-party conversation as first-class features rather than an afterthought. The OpenAI Agents SDK is the better production choice for OpenAI-committed teams wanting minimal surface area; AutoGen / AG2 is the better research tool for teams exploring emergent multi-agent conversation patterns across any model provider.

**Choose AutoGen / AG2 when:** multi-provider flexibility is required, code-executing agents are central to the workflow, or you need expressive multi-party group chat patterns not easily expressed in handoff routing.

**Choose the OpenAI Agents SDK when:** your stack is fully committed to OpenAI, you need voice agent support, or you want the smallest possible framework surface area with built-in guardrails.

The differentiating dimension is **provider flexibility and conversational emergence vs. OpenAI platform depth**.

### AutoGen vs. Haystack

AutoGen and Haystack rarely compete directly — they address different layers of the AI application stack. AutoGen operates at the agent coordination layer: how multiple agents converse, debate, and produce outputs through iterative dialogue. Haystack operates at the retrieval and pipeline layer: how documents are ingested, indexed, retrieved, and fed into generation. For applications that need both, the natural pattern is Haystack pipelines exposed as tools called by AutoGen agents.

**Choose AutoGen / AG2 when:** the primary challenge is multi-agent reasoning, conversation-driven exploration, or code-generating workflows where retrieval is incidental.

**Choose Haystack when:** retrieval quality, document intelligence, and pipeline explainability are the primary engineering concerns — there is no retrieval-heavy application where AutoGen is a better choice than a dedicated retrieval framework.

The differentiating dimension is **conversation-driven orchestration vs. retrieval pipeline depth**. The two frameworks compose naturally rather than compete.

### AutoGen vs. LlamaIndex

AutoGen and LlamaIndex are architectural peers at different layers of the stack — AutoGen at orchestration, LlamaIndex at data. LlamaIndex's agentic workflow capabilities (Events, Steps, Context) have expanded significantly, but the framework remains retrieval-centric at heart. AutoGen remains conversation-centric. For applications where the primary value is reasoning over large document corpora, LlamaIndex's retrieval infrastructure is far superior. For applications where the primary value is multi-party conversational agent behavior, AutoGen's dialogue model is more expressive.

**Choose AutoGen / AG2 when:** conversational multi-agent patterns, code execution, and emergent task negotiation are the core requirements and document retrieval is a secondary concern.

**Choose LlamaIndex when:** document ingestion quality, retrieval accuracy, and data pipeline sophistication are the primary differentiators — LlamaIndex's data layer is materially stronger than what AutoGen provides.

The differentiating dimension is **conversational orchestration vs. data retrieval infrastructure**. Production systems requiring both typically use LlamaIndex for the data layer with AutoGen or a more production-hardened orchestrator above it.

### AutoGen vs. Mastra

AutoGen and Mastra occupy entirely different language ecosystems and rarely compete for the same team. AutoGen is a Python framework targeting research and AI-exploration workflows; Mastra is a TypeScript framework targeting production Node.js applications. If your team is polyglot and evaluating which language to build your agent system in, Mastra's batteries-included TypeScript stack (memory, durable workflows, RAG, evals, observability) is a stronger production foundation than AutoGen, which is in maintenance mode on the Microsoft side and focused on the research community via AG2.

**Choose AutoGen / AG2 when:** your team is Python-native, the use case is research or exploration-oriented, and conversational multi-agent patterns are the primary value.

**Choose Mastra when:** your team is TypeScript-native, you need a production-ready agent framework with first-class memory, workflows, and observability, and you want active development and long-term maintainability.

The differentiating dimension is **language ecosystem and production readiness**. These frameworks do not compete in the same market segment.

---

## 9. Community and Market Position

### Key Metrics (as of May 2026)

- **GitHub stars (`ag2ai/ag2`):** 50,000+ (inherited from the AutoGen fork; one of the highest star counts of any agent framework)
- **GitHub stars (`microsoft/autogen`):** ~40,000+ on the Microsoft-maintained repo (maintenance mode)
- **Discord community:** 20,000+ members (AG2 Discord, inherited from AutoGen community)
- **Original paper citations:** ~1,300 (Google Scholar) — most cited agent framework paper in academic literature
- **Azure AI Foundry users:** 10,000+ organizations (across the full Foundry platform, including AutoGen-lineage workloads)
- **AutoGen Studio:** Available via `pip install autogenstudio`; no-code GUI for prototyping
- **Microsoft AutoGen status:** Maintenance mode as of October 2025

### Company Background and Funding

AutoGen was created at **Microsoft Research** — one of the most well-funded corporate research organizations in the world. The framework was not a startup product; it was academic research code that became a de facto standard through community adoption. The original team (Chi Wang, Qingyun Wu, and colleagues) operated within Microsoft Research's academic culture, prioritizing research novelty over production engineering — a tradeoff that explains many of AutoGen's architectural characteristics.

**AG2** is now governed by the original creators outside of Microsoft. Chi Wang moved to **Google DeepMind** after departing Microsoft. AG2 is maintained as a community-governed open-source project under the AG2AI organization, with explicit commitment to open governance and contributions from multiple organizations (AWS, IBM, Databricks, Google Cloud partnerships are listed on the AG2 ecosystem page). AG2 is not VC-funded — its sustainability depends on community contributions and the reputations of its founding contributors.

The contrast with competitors is stark: LangChain (which backs LangGraph) has raised significant VC funding; CrewAI Inc. has raised capital for its commercial platform; Pydantic received $17M from Sequoia. AG2 has no disclosed external funding and relies on community governance. This is either a strength (no investor pressure to commercialize in ways that compromise the open-source framework) or a risk (no capital to sustain a dedicated engineering team).

### Industry Recognition

AutoGen's recognition is primarily academic rather than commercial: the ~1,300 research citations are the most significant measure of influence in the field. The framework is consistently cited in AI agent architecture papers as the reference implementation for conversational multi-agent systems. IBM's published explainer on AutoGen and the wide coverage in practitioner-oriented publications (DataCamp, Analytics Vidhya, Real Python) reflect a large learning community that has studied AutoGen even if not all are using it in production.

The Microsoft Research backing, despite the maintenance mode status, lends institutional credibility that purely community-originated frameworks lack. And the AG2 fork's partnerships with major cloud providers (AWS, IBM, Databricks, Google Cloud) suggest that the community continuation has attracted enterprise-level interest even without a commercial entity behind it.

### Community Sentiment

The community sentiment around AutoGen is bifurcated along use-case lines. Research and experimentation users consistently praise the framework for its flexibility, the intuitive conversation model, and the code execution loop — it is widely described as the easiest framework for "getting something surprising to happen quickly." Production users consistently criticize the non-determinism, token costs, and debugging difficulty — production GitHub issues include memory leaks, concurrency bugs in GroupChat, and the auto speaker selection picking wrong agents. The sentiment trend in 2025–2026 has shifted toward viewing AutoGen as a prototype and research tool rather than a production platform of choice, with LangGraph gaining adoption for production workloads that AutoGen was previously used for.

The AG2 fork has generated genuine community enthusiasm among practitioners who value the continuation of the original design philosophy under community governance. Discord and GitHub discussions reflect an active and technically engaged community, though one smaller and less commercially oriented than LangGraph's.

---

## 10. Pricing

AutoGen (both `microsoft/autogen` and `ag2ai/ag2`) is **fully free and MIT licensed** with no framework fees, no usage-based charges, and no commercial platform subscription required. AG2 has no enterprise product or commercial tier — it is purely open source. All monetary costs associated with AutoGen deployments come from external sources: **LLM API provider fees**, **cloud infrastructure**, and, for managed enterprise deployments, **Azure AI Foundry Agent Service** pricing.

| **Tier** | **Price** | **What's Included** | **LLM Costs** | **Deployment** | **Support** |
|---|---|---|---|---|---|
| **AutoGen OSS (microsoft/autogen)** | Free (MIT) | Framework, Studio, .NET SDK — maintenance mode | Pay-per-token (provider) | Self-managed | Community (GitHub Issues) |
| **AG2 OSS (ag2ai/ag2)** | Free (MIT) | Active framework, A2A/MCP/AG-UI support, multi-provider | Pay-per-token (provider) | Self-managed | Community (Discord, GitHub) |
| **Azure AI Foundry Agent Service** | Consumption-based (Azure) | Managed hosting, state management, enterprise integrations | Included in Azure AI consumption | Azure-managed | Azure enterprise SLAs |
| **Microsoft Agent Framework** | Free OSS (MIT) | AutoGen + Semantic Kernel successor, enterprise features | Pay-per-token (provider) | Self or Azure-managed | Microsoft (enterprise contracts) |

*Azure AI Foundry pricing is consumption-based and depends on model usage, storage, and API calls. Pricing requires review of Azure's current rate card at azure.microsoft.com/pricing. Microsoft Agent Framework is free and open source; enterprise support is available through Microsoft enterprise agreements.*

### Open Source (Free)

Both AutoGen and AG2 are free at the framework level. The complete framework, including AutoGen Studio, the .NET SDK, all agent patterns, and all integrations, is available at zero cost. Infrastructure to run AutoGen (a Python environment, network access to LLM APIs) is the only hard requirement. This makes AutoGen accessible to any developer or organization regardless of budget — which is a significant factor in its massive adoption in academic and research settings.

### LLM API Costs at Scale

AutoGen's token consumption model is the primary cost driver and its most significant production cost concern. The GroupChat pattern's full-history-per-turn design means costs scale faster than linearly with conversation length and agent count. A practical cost model for a 4-agent GroupChat running 10 turns with GPT-4o:

- ~20,000 input tokens per turn (growing conversation history) × 10 turns × $2.50/million = ~$0.50 per conversation
- With 1,000 conversations/day: ~$500/day ($15,000/month) in model costs alone

This is substantially higher than equivalent LangGraph workflows that avoid redundant context passing through graph-based state management. Cost optimization in AutoGen typically requires: using cheaper models for GroupChatManager speaker selection; limiting GroupChat turns aggressively; summarizing conversation history at intervals; or routing simpler subtasks to two-agent loops rather than full GroupChats.

### Azure AI Foundry (Enterprise Path)

For teams wanting managed infrastructure, **Azure AI Foundry Agent Service** provides hosted agent execution with enterprise governance. Pricing is consumption-based on Azure: model inference tokens (at Azure OpenAI rates), agent execution compute, storage for conversation state, and API calls. At enterprise scale, Azure offers committed-use discounts and private pricing. Teams already within Azure's enterprise agreements can leverage existing credit and procurement relationships. This is not an AutoGen-specific product — it is Microsoft's full enterprise AI platform — but it is the recommended production path for AutoGen users on Azure.

### Real-World Cost Scenarios

**Solo developer / side project:** $0 framework cost. LLM API costs for AutoGen experimentation depend heavily on conversation length — budget $20–$100/month for moderate development usage with GPT-4o-mini as the primary model.

**Small startup (3–5 people):** Self-managed AG2 on cloud infrastructure. LLM costs at production volume (5,000 agent conversations/month, 4-agent GroupChats, 8 turns average, GPT-4o): approximately $500–$2,000/month in inference. Infrastructure (a small VM or serverless compute): $50–$200/month. Total: $550–$2,200/month.

**Mid-size team in production (20–50 people):** AG2 or AutoGen on self-managed infrastructure with per-agent model routing (expensive models for reasoning, cheap models for coordination). At 50,000 conversations/month with intelligent model routing: $2,000–$8,000/month in LLM costs. Consider Azure AI Foundry for managed state and governance: additional $500–$2,000/month. Total: $2,500–$10,000/month.

**Large enterprise (100+ people):** Azure AI Foundry Agent Service with committed enterprise pricing. At this scale, model costs dominate and are negotiated directly with Microsoft. Infrastructure and platform: $5,000–$30,000/month. Total annual cost: $100,000–$500,000+, primarily driven by LLM consumption volume and the degree of Azure-managed infrastructure.

### Pricing Caveats

Azure AI Foundry pricing changes frequently as Microsoft adjusts Azure OpenAI and Foundry rates. LLM API costs across providers (OpenAI, Anthropic, Google) are in active flux as model tiers expand. AutoGen's token-intensive GroupChat model means that production cost estimates should be validated against actual conversation data before budgeting. The self-hosted path (AG2 + self-managed LLM provider) is fully viable and eliminates platform fees at the cost of infrastructure management.

---

## 11. Summary and Verdict

**Positioning statement:** AutoGen pioneered conversational multi-agent AI and remains the most academically influential framework in the category, but its Microsoft-backed version is in maintenance mode, its community fork (AG2) has uncertain long-term governance, and its conversation-first design makes it genuinely expensive and difficult to control in high-volume production — it is the right framework for research, code generation, and exploratory multi-agent experimentation, and the wrong framework for cost-sensitive, deterministic, or high-reliability production workloads.

### When to Choose AutoGen / AG2

- Your primary use case is code generation with iterative self-correction — the write-run-debug loop is what you're building, and no other framework handles it as naturally
- The problem structure is genuinely unknown in advance and you need agents to explore, debate, and converge on an approach rather than follow a pre-defined workflow
- You are building for a research or academic context where the community's ~1,300-citation body of published benchmarks and evaluations is directly relevant to your work
- Your stack is .NET / C# and you need a multi-agent framework with native C# support
- You are using AG2 and value community governance and the original AutoGen design philosophy over commercial backing
- You need to prototype quickly and are willing to address production concerns (costs, determinism, persistence) in a subsequent engineering phase

### When Not to Choose AutoGen / AG2

- You are starting a new project on Azure and want Microsoft's supported enterprise path — use Microsoft Agent Framework instead
- Deterministic, auditable routing is a compliance or governance requirement — conversation-driven speaker selection cannot satisfy this
- Your production volume makes per-turn token costs a real concern — the GroupChat full-history pattern is expensive at scale
- You need durable workflow persistence: conversations that fail mid-execution must resume from a checkpoint rather than restart
- Long-term framework stability is important to your organization — Microsoft AutoGen is in maintenance mode, and AG2's community governance model carries more uncertainty than commercially backed alternatives

### Closing Perspective

AutoGen's place in the agent framework ecosystem is unusual: it is simultaneously the most influential framework (by academic citations), one of the least production-ready (by the standards of determinism and cost), and in an ambiguous governance position (maintained by Microsoft in maintenance mode; forked and continued by the original creators as AG2). This combination makes it the framework that almost everyone has heard of, many have experimented with, and fewer have taken all the way to high-reliability production.

The honest assessment for teams evaluating AutoGen in 2026 is this: if you are exploring what multi-agent systems can do, AutoGen's conversation model remains the fastest path to interesting behavior. If you are building something that needs to work reliably, cost-predictably, and auditively for months or years, the conversation model's flexibility becomes a liability. LangGraph gives you the production infrastructure AutoGen lacks; Microsoft Agent Framework gives you the Microsoft-supported successor for Azure teams; Pydantic AI gives you the engineering discipline AutoGen deliberately avoids. AutoGen's legacy is foundational — the frameworks that followed were all responding to what it demonstrated. Its future as a first-choice production platform is genuinely uncertain.

---

## Sources

- [GitHub — microsoft/autogen](https://github.com/microsoft/autogen)
- [GitHub — ag2ai/ag2: AG2 (formerly AutoGen)](https://github.com/ag2ai/ag2)
- [AutoGen — Microsoft Research](https://www.microsoft.com/en-us/research/project/autogen/)
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation — arXiv:2308.08155](https://arxiv.org/abs/2308.08155)
- [AutoGen Studio: A No-Code Developer Tool for Building and Debugging Multi-Agent Systems — arXiv:2408.15247](https://arxiv.org/abs/2408.15247)
- [AutoGen Official Documentation — microsoft.github.io](https://microsoft.github.io/autogen/stable//index.html)
- [AutoGen v0.4: Simplifying Agentic AI for Developers Everywhere — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/01/autogen-v0-4/)
- [AG2: Build Systems, Not Prompts — ag2.ai](https://www.ag2.ai/)
- [AG2 Open Ecosystem — ag2.ai](https://www.ag2.ai/ecosystem)
- [AG2 Release Roadmap — docs.ag2.ai](https://docs.ag2.ai/latest/docs/user-guide/release-roadmap/)
- [AutoGen to Microsoft Agent Framework Migration Guide — Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/)
- [Microsoft Agent Framework Overview — Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Semantic Kernel + AutoGen = Open-Source Microsoft Agent Framework — Visual Studio Magazine](https://visualstudiomagazine.com/articles/2025/10/01/semantic-kernel-autogen--open-source-microsoft-agent-framework.aspx)
- [Microsoft Agent Framework: Production-Ready Convergence of AutoGen and Semantic Kernel — Cloud Summit EU](https://cloudsummit.eu/blog/microsoft-agent-framework-production-ready-convergence-autogen-semantic-kernel)
- [What is AutoGen? — IBM](https://www.ibm.com/think/topics/autogen)
- [AutoGen Studio User Guide — AutoGen Documentation](https://microsoft.github.io/autogen/dev//user-guide/autogenstudio-user-guide/index.html)
- [Multi-Agent Conversation Framework — AutoGen 0.2 Docs](https://microsoft.github.io/autogen/0.2/docs/Use-Cases/agent_chat/)
- [CrewAI vs LangGraph vs AutoGen 2026: Benchmarks and the Right Choice — Pooya Golchian](https://pooya.blog/blog/crewai-vs-langgraph-autogen-comparison-2026/)
- [CrewAI vs LangGraph vs AutoGen — DataCamp](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)
- [Is AutoGen Worth the Hype? Limitations and Real-World Use Cases — Toolify](https://www.toolify.ai/ai-news/is-autogen-worth-the-hype-limitations-and-realworld-use-cases-revealed-1482386)
- [Best Multi-Agent Frameworks in 2026 — Gurusup](https://gurusup.com/blog/best-multi-agent-frameworks-2026)
- [AG2 vs CrewAI: The Complete Comparison — DEV Community](https://dev.to/agentsindex/ag2-vs-crewai-the-complete-comparison-including-the-autogen-rebrand-explained-248l)
- [AutoGen Pricing Overview — Procurement Sciences](https://www.procurementsciences.com/blog/autogen-pricing)
- [AutoGen vs Microsoft Agent Framework: Comparison — iCert Global](https://www.icertglobal.com/community/autogen-vs-microsoft-agent-framework-2026-guide)
- [Definitive Guide to Agentic Frameworks 2026 — Softmax Data](https://softmaxdata.com/blog/definitive-guide-to-agentic-frameworks-in-2026-langgraph-crewai-ag2-openai-and-more/)
- [AI Agent Frameworks 2026: Production-Tested Ranking — Alice Labs](https://alicelabs.ai/en/insights/best-ai-agent-frameworks-2026)
- [AutoGen (framework) — AI Wiki](https://aiwiki.ai/wiki/autogen)
- [A Friendly Introduction to the AutoGen Framework (v0.4 API) — Victor Dibia Newsletter](https://newsletter.victordibia.com/p/a-friendly-introduction-to-the-autogen)
- [AutoGen Multi-Agent Framework Implementation Patterns — Galileo AI](https://galileo.ai/blog/autogen-multi-agent)
- [Comparing Open-Source AI Agent Frameworks — Langfuse](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)
