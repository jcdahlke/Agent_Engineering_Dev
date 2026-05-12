# Microsoft Agent Framework — Deep Research Report

**Research Date:** May 8, 2026  
**Subject:** Microsoft Agent Framework — Architecture, Adoption, Use Cases, and Competitive Landscape

---

## Table of Contents

1. [What Is Microsoft Agent Framework?](#1-what-is-microsoft-agent-framework)
2. [How It Works — Architecture Deep Dive](#2-how-it-works--architecture-deep-dive)
3. [The Microsoft Agent Framework Ecosystem](#3-the-microsoft-agent-framework-ecosystem)
4. [Who Uses Microsoft Agent Framework?](#4-who-uses-microsoft-agent-framework)
5. [Industries and Use Cases](#5-industries-and-use-cases)
6. [Why People Choose Microsoft Agent Framework](#6-why-people-choose-microsoft-agent-framework)
7. [Why People Don't Choose Microsoft Agent Framework](#7-why-people-dont-choose-microsoft-agent-framework)
8. [Microsoft Agent Framework vs Competing Frameworks](#8-microsoft-agent-framework-vs-competing-frameworks)
9. [Community and Market Position](#9-community-and-market-position)
10. [Pricing](#10-pricing)
11. [Summary and Verdict](#11-summary-and-verdict)
- [Sources](#sources)

---

## 1. What Is Microsoft Agent Framework?

Microsoft Agent Framework is an open-source SDK and runtime for building, orchestrating, and deploying production-grade AI agents and multi-agent workflows in Python and .NET. It is Microsoft's unified answer to a problem the company created for itself: over several years, its two flagship AI developer frameworks — **AutoGen** (focused on multi-agent orchestration) and **Semantic Kernel** (focused on enterprise-grade AI integration) — had diverged into incompatible ecosystems that split the Microsoft developer community. Agent Framework merges them into a single coherent platform.

Microsoft announced the project in public preview on October 1, 2025, describing it as "an open-source SDK and runtime that unifies the enterprise-ready foundations of Semantic Kernel with the innovative orchestration of AutoGen." After roughly six months of hardening with enterprise customers and partners, the framework reached **General Availability at version 1.0 on April 3, 2026**. As of early May 2026, it is at version 1.3.0. Both AutoGen and Semantic Kernel are now in maintenance mode — receiving security patches but no new features — with Microsoft officially directing all new development toward Agent Framework.

The core mental model is a **dual-track architecture**: Agent Orchestration for open-ended, LLM-driven reasoning tasks and Workflow Orchestration for deterministic, business-logic-driven processes. Unlike frameworks that force you to pick one paradigm, Agent Framework treats these as composable building blocks — a workflow can call an agent, and an agent can trigger a workflow sub-graph.

The framework is released under the **MIT License** and is fully open source, hosted at `github.com/microsoft/agent-framework`. The companion enterprise runtime, **Foundry Agent Service**, is a commercial managed platform built on Azure.

**Headline metrics (as of May 2026):** The `microsoft/agent-framework` repository is actively developed with frequent releases (v1.3.0 released May 8, 2026). The predecessor `microsoft/autogen` repository accumulated over 43,000 GitHub stars before being placed into maintenance mode. Enterprise adoption is confirmed across pharmaceutical, financial, automotive, and professional services sectors. The `agent-framework` PyPI package crossed 1 million monthly downloads within its first three months of GA.

> *"Microsoft Agent Framework is where we are consolidating our investments in agentic AI. It doesn't replace Semantic Kernel and AutoGen — it builds on them."*  
> — Microsoft Foundry Blog, April 2026

In a single sentence: Microsoft Agent Framework is Microsoft's production-hardened, enterprise-focused multi-agent SDK that inherits AutoGen's orchestration patterns and Semantic Kernel's enterprise plumbing, designed for teams building at scale on Azure.

---

## 2. How It Works — Architecture Deep Dive

### Core Primitives

The framework is built on five foundational concepts:

**Agent** is the primary unit of work. An agent wraps a model client, a system prompt, a set of tools, and a context provider into a single callable object. Agents are stateless by default; state is managed separately via sessions. There are several built-in agent types — `ChatCompletionAgent` for standard LLM turns, `StructuredOutputAgent` for typed JSON responses, and `ToolCallAgent` for tool-heavy workflows. Custom agents extend a base `Agent` class.

**AgentSession** manages multi-turn conversational state. Rather than threading conversation history directly through agent calls, sessions act as scoped containers that maintain message history, tool call records, and metadata. This separation keeps agents reusable across different conversations. Sessions can be serialized and checkpointed to durable storage, which is what enables long-running workflows that survive process restarts.

**Middleware** is a pipeline of interceptors that wraps every agent invocation. Middleware handlers receive the request before the model is called and the response after, allowing content safety filters, logging, compliance policies, retry logic, and usage tracking to be applied globally without modifying individual agent prompts. This is one of the clearest inheritances from Semantic Kernel.

**Context Providers** feed information into the agent's reasoning at invocation time — RAG retrievals, vector search results, structured data from APIs, or summaries from previous sessions. Providers are pluggable and composable, allowing the same agent definition to behave differently depending on what context is injected.

**Workflows** represent deterministic multi-agent pipelines expressed as a typed graph. A workflow consists of **Executors** (agents, functions, or nested sub-workflows) connected by **Edges** (typed, optionally conditional message routes). Workflows support checkpointing and request/response pauses — a workflow can pause mid-execution, await a human approval event, and resume without losing state.

### Data Flow and Execution

A single-agent invocation flows: input message → middleware pre-processing → context provider enrichment → model call (with tool loop if tools are invoked) → middleware post-processing → output message. Sessions accumulate this history across turns.

In multi-agent workflows, messages pass between executors along defined edges. The framework supports five named orchestration patterns: **Sequential** (A → B → C in order), **Concurrent** (A and B run in parallel, results merge), **Group Chat** (agents deliberate collaboratively on a shared thread), **Handoff** (control transfers based on specialization, like a router pattern), and **Magentic** (a manager agent builds a dynamic task ledger and dispatches to specialized sub-agents — Microsoft's implementation of the Magentic-One research pattern for open-ended complex tasks).

### Decision-Making and Routing

In workflow mode, routing is **explicit and deterministic**: edge conditions are defined in code using typed predicates. The LLM does not decide the flow — the developer does. In agent orchestration mode (group chat, Magentic), routing is **emergent**: the manager agent decides which sub-agent handles the next step based on LLM reasoning. Both modes can coexist in the same application.

### Minimal Code Example

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

async def main():
    # Connect to a model via Azure AI Foundry
    client = FoundryChatClient(
        endpoint="https://<your-project>.services.ai.azure.com",
        credential=DefaultAzureCredential(),
        model="gpt-4o",
    )
    # Define a single agent with a system prompt
    agent = client.as_agent(instructions="You are a helpful research assistant.")

    # Run a single-turn invocation
    response = await agent.run("Summarize the latest news about agentic AI.")
    print(response.content)

asyncio.run(main())
```

The `FoundryChatClient` wraps model connectivity and telemetry; `as_agent()` produces a fully configured `ChatCompletionAgent` with the session managed implicitly for single-turn use.

### Error Handling and Resilience

Agent Framework includes built-in retry policies configurable at the middleware layer, with exponential backoff on model API failures. Workflows support checkpoint-based recovery: if a workflow executor fails after a checkpoint, the workflow can resume from the last saved state rather than starting over. The session serialization mechanism also serves as an audit log for compliance-sensitive deployments.

### Memory and Context

Short-term memory is the session history — messages and tool calls accumulated within a single AgentSession. Long-term memory is the responsibility of external context providers, typically backed by Azure AI Search, Azure Cosmos DB, or any vector store accessible via an MCP server. There is no built-in long-term memory implementation; the framework provides the injection hooks, not the storage.

### MCP Integration

The framework has native first-class support for the **Model Context Protocol (MCP)**. Agents can connect to MCP servers at startup or dynamically at runtime, discover the available tool manifest, and make tools available for LLM tool-calling without any manual integration code. This enables agent systems to stay current with expanding tool registries without redeployment.

---

## 3. The Microsoft Agent Framework Ecosystem

### Parent Platform: Microsoft Foundry

**Microsoft Foundry** (formerly Azure AI Foundry) is the commercial platform layer that Agent Framework is designed to deploy into. Foundry provides managed infrastructure, model access, observability dashboards, and governance tooling. The open-source SDK and the Foundry platform are designed to be used together, though the SDK can run independently against any supported model provider.

### Managed Runtime: Foundry Agent Service

**Foundry Agent Service** is the hosted execution environment for Agent Framework applications. It runs agents in customer-dedicated containers managed by Microsoft, with built-in auto-scaling, OpenTelemetry-powered observability, security sandboxing, and governance hooks. Foundry Agent Service also supports "hosted agents" from other frameworks — including LangGraph — so teams with mixed stacks can centralize deployment on a single platform.

### Visual IDE: Foundry Portal and Aspire Integration

The **Azure AI Foundry portal** provides a visual interface for inspecting agent runs, viewing trace data, configuring tool connections, and monitoring token consumption. For local development, Agent Framework integrates with **.NET Aspire**, Microsoft's developer orchestration tool, for running distributed agent applications locally with service discovery, dashboarding, and health checks — effectively providing a local development environment that mirrors the cloud architecture.

### Model Connectors

Agent Framework ships first-party service connectors for: Azure OpenAI, OpenAI, Anthropic Claude, Amazon Bedrock, Google Gemini, and Ollama (for local models). This means teams are not locked to Azure-hosted models and can switch or mix model providers at the connector level.

### Observability

The framework integrates with **OpenTelemetry** natively, emitting structured traces for every agent invocation, tool call, and workflow transition. These traces flow into Azure Monitor, Application Insights, or any OTLP-compatible backend. First-party evaluation dashboards in the Foundry portal surface metrics like task completion rate, tool call frequency, latency, and cost per agent run.

### Tool Ecosystem and MCP Marketplace

**Foundry Tools** is Microsoft's managed MCP server marketplace, providing pre-built, governed tool integrations for Microsoft Graph, SharePoint, Bing Search, Azure Blob Storage, and external SaaS connectors. Teams can use these tools directly without building or hosting their own MCP servers.

### Azure Durable Functions Integration

For workflows requiring very long execution windows (days or weeks), Agent Framework supports deployment as **Azure Durable Functions**, leveraging Durable Functions' existing checkpointing and orchestration infrastructure for reliability at scale.

---

## 4. Who Uses Microsoft Agent Framework?

| **Company** | **Use Case** |
|---|---|
| **Novo Nordisk** | Multi-agent framework for helping data scientists and researchers derive insights from complex pharmaceutical and technical data in production |
| **KPMG** | Deployed "Clara AI," a multi-agent audit system connecting agents to financial data with governance and observability features required for regulated audit workflows |
| **BMW** | Multi-agent system analyzing terabytes of vehicle telemetry; agents deliver actionable engineering insights rather than raw data outputs |
| **Commerzbank** | Piloting avatar-driven customer support agents; selected Agent Framework for its MCP support and reduced development effort vs. prior tooling |
| **Fujitsu** | Embedding Agent Framework into enterprise integration services to enable human-AI collaborative workflows at scale |
| **Citrix** | Deploying agent-based workflows for IT service management and customer-facing support automation |
| **TCS (Tata Consultancy Services)** | Integrating Agent Framework into enterprise AI offerings delivered to TCS clients across industries |
| **TeamViewer** | Building agent-assisted remote support and IT automation workflows for enterprise customers |
| **Elastic** | Implementing Agent Framework for developer-facing and data pipeline automation scenarios |
| **Accenture** | Leveraging Agent Framework within Foundry-based enterprise AI solutions for Fortune 500 clients |
| **Microsoft (internal)** | Used internally across multiple Microsoft product teams for Copilot feature development, developer tools, and internal automation workflows |

---

## 5. Industries and Use Cases

### Financial Services and Audit

Financial services firms are using Agent Framework for regulated, multi-step workflows that require auditability at every decision point. KPMG's Clara AI demonstrates the pattern: multiple specialized agents (data retrieval, anomaly detection, narrative generation, compliance review) are wired together in a workflow graph where every step is logged, every tool call is traced, and human reviewers can be inserted as checkpoint gates. The framework's built-in session persistence and OpenTelemetry tracing make it well-suited for audit trails, which are mandatory in regulated environments. Commerzbank's customer support pilot shows a parallel pattern — avatar-driven agents in financial customer service where interaction history and compliance guardrails are non-negotiable.

### Pharmaceutical and Life Sciences

Novo Nordisk's production deployment represents a growing use pattern: data science teams using multi-agent systems to translate complex technical datasets into researcher-accessible insights. The workflow involves one agent for data retrieval from scientific databases, another for statistical summarization, and a third for narrative generation tailored to the audience (clinical vs. engineering). The framework's context provider pattern handles the heterogeneous data sources cleanly, and session-based state allows multi-turn conversations where researchers follow up on summaries.

### Automotive and Manufacturing

BMW's telemetry analysis application illustrates the high-volume, high-stakes analytical pattern: agents processing terabytes of sensor data from production vehicles to identify durability issues, failure patterns, and maintenance signals. The framework's Concurrent orchestration pattern allows multiple analytical agents to process different telemetry streams in parallel before a synthesis agent assembles actionable reports for engineers. Observability and durability are cited as the primary reasons for choosing Agent Framework over alternatives.

### Enterprise IT and Developer Tools

Citrix and TeamViewer represent a natural fit — IT automation workflows where agents must interact with multiple enterprise systems (ticketing, monitoring, user directories, deployment pipelines) and route tasks to the right handler. The MCP integration is particularly valuable here, as it allows agents to discover and invoke IT tools dynamically without hardcoded integrations. TCS and Fujitsu are embedding Agent Framework as an infrastructure layer beneath client-facing AI products, treating it as a standard enterprise integration runtime.

### Professional Services and Consulting

KPMG and Accenture represent the "framework as delivery vehicle" pattern: large consulting firms adopting Agent Framework as their standard agentic AI platform for client engagements. For these firms, enterprise readiness features — compliance hooks, role-based access, audit logging, multi-cloud model support — matter more than developer ergonomics. The Foundry Agent Service managed runtime reduces infrastructure burden for client deployments.

### Customer Support and Engagement

The customer support pattern appears across multiple adopters. The typical architecture pairs a routing agent (classifying the inquiry type) with specialized agents (order management, billing, technical support) connected via the Handoff orchestration pattern. Session persistence allows agents to maintain context across multi-turn support conversations, and the middleware pipeline handles content safety filtering without modifying agent logic.

### Software Development and DevOps

Internally at Microsoft and among engineering-heavy adopters, Agent Framework powers Copilot-style developer tools: agents that query codebases, execute tests, analyze CI/CD output, and draft pull request summaries. The Magentic orchestration pattern is particularly valuable for open-ended development tasks where the sequence of sub-tasks is not known in advance.

---

## 6. Why People Choose Microsoft Agent Framework

### Enterprise Readiness Out of the Box

Agent Framework is the only major open-source agent framework that ships production-grade enterprise features — middleware pipelines, session persistence, checkpoint-based recovery, OpenTelemetry tracing, and compliance hooks — as first-class primitives, not afterthoughts. Frameworks like LangGraph require significant additional work to achieve the same level of observability and durability. Teams building in regulated industries (financial services, healthcare, government) get most of what they need from the framework itself.

### The Dual-Track Architecture Covers More Ground

Most frameworks force a choice: either build deterministic, graph-based workflows (LangGraph's model) or embrace emergent, LLM-driven collaboration (AutoGen's model). Agent Framework supports both patterns natively and lets them compose. A workflow can contain a group chat agent as one of its executors; a Magentic orchestrator can invoke deterministic sub-workflows. This is a genuine architectural advantage for production applications that need both reliable, repeatable business processes and flexible agentic reasoning in the same system.

### Best-in-Class MCP Integration

Microsoft built first-class, native MCP support before most competitors shipped experimental integrations. This matters practically: agents can discover tools at runtime from any MCP server without manual integration code, and the Foundry Tools marketplace provides governed, production-quality MCP server implementations for common enterprise data sources. Teams who have invested in the MCP ecosystem get the most leverage from Agent Framework.

### Multi-Language Support (.NET and Python)

Agent Framework is one of the few frameworks with full feature parity between Python and .NET. Enterprise organizations with significant .NET codebases — which describes most large financial and government institutions — can adopt Agent Framework without rewriting infrastructure in Python. LangGraph, CrewAI, and Google ADK are Python-only. This is not a minor consideration for large enterprises where the majority of backend services are in C#.

### Azure Integration Depth

For organizations already on Azure, Agent Framework's integration with Azure AI Foundry, Azure OpenAI, Azure Durable Functions, Application Insights, and Azure AI Search removes weeks of plumbing work. Foundry Agent Service provides managed deployment, scaling, and monitoring without requiring teams to operate their own infrastructure. This is a genuine productivity advantage over self-managed LangGraph deployments.

### Explicit Microsoft Backing and Long-Term Support

The v1.0 GA release includes a commitment to stable APIs and long-term support — a significant consideration for enterprise procurement decisions. Organizations that avoided AutoGen due to its rapid API churn now have a framework with an enterprise-grade stability commitment. The migration guides provided for both AutoGen and Semantic Kernel users reflect a serious investment in the existing Microsoft developer community.

### Magentic-One for Open-Ended Task Automation

Agent Framework is the only framework with a first-party, production-supported implementation of the **Magentic-One** pattern — Microsoft Research's approach to open-ended, multi-agent task completion where a manager agent maintains a dynamic task ledger and assembles specialized sub-agents on the fly. For teams working on complex research, analysis, or software automation tasks where the sequence of steps cannot be predetermined, this is the most advanced orchestration pattern available in any major framework.

---

## 7. Why People Don't Choose Microsoft Agent Framework

### Real and Intentional Azure Lock-In

The framework works outside Azure — it supports OpenAI, Anthropic, Bedrock, and Gemini as model providers — but the most valuable features assume Azure infrastructure. Foundry Agent Service managed deployment, the Foundry Tools MCP marketplace, native Durable Functions checkpointing, Azure Monitor observability, and the Foundry portal visual IDE all require Azure. Teams on GCP or AWS who want the enterprise features must build the equivalent plumbing themselves or accept a multi-cloud cost and complexity penalty. The architecture was not designed to be cloud-neutral; it was designed to be excellent on Azure.

### Complexity Overhead for Simple Use Cases

What takes 30–50 lines in LangGraph or CrewAI can require substantially more scaffolding in Agent Framework when Azure service dependencies are wired in. For small teams, solo developers, or projects that will never approach enterprise scale, the framework's architecture introduces ceremony that provides no return. If your use case is a simple two-agent pipeline with a few tools, Agent Framework is the wrong abstraction level.

### Youngest Production Track Record

Agent Framework reached GA in April 2026. LangGraph has been in production deployments since 2023; CrewAI since early 2024. The community of practitioners who have encountered and documented production failure modes, scaling limits, and edge case behaviors is proportionally smaller. There are fewer Stack Overflow answers, fewer blog posts about debugging obscure failures, and fewer third-party integrations than with more established frameworks. This matters significantly when your agent system breaks at 2am and you need community knowledge.

### Python Ecosystem Lag in Some Areas

Despite full parity on core features, the .NET SDK benefits from deeper integration with the broader Microsoft developer stack. Python users working outside the Azure ecosystem report that some advanced Foundry integrations require Azure SDK knowledge that adds friction for teams coming from the broader Python AI ecosystem (e.g., LangChain, LlamaIndex). The Python SDK's import structure changed between the release candidate and 1.0, and migration between minor versions has required code adjustments.

### AutoGen Migration Friction

For the large community of developers who built on AutoGen 0.2 or 0.4, migration to Agent Framework is non-trivial. The API surface is significantly different — Agent Framework is not a superset of AutoGen, it is a redesign. The official migration guide exists but describes substantial refactoring work, particularly for applications that relied on AutoGen's `ConversableAgent` patterns directly. Teams with significant AutoGen codebases face a real migration cost before realizing the benefits.

### Debugging Multi-Agent Workflows Remains Hard

Despite the improved observability story, debugging complex multi-agent failures in production is still painful. When a Magentic orchestrator delegates to the wrong sub-agent or a handoff loop cycles unexpectedly, the OpenTelemetry traces provide data but not diagnosis. The framework does not yet provide a time-travel debugger equivalent to LangGraph's (available via LangSmith), and the Foundry portal's trace viewer is functional but not yet as ergonomic as LangSmith for deep inspection of agent reasoning chains.

### No Native Graph Visualization

Agent Framework's workflow graph is defined in code but has no built-in visualization of the workflow topology. Understanding a complex workflow requires reading code or manually building a diagram. LangGraph produces visual representations of the graph natively in LangSmith. For teams onboarding new engineers or presenting agentic architectures to stakeholders, this is a genuine usability gap.

---

## 8. Microsoft Agent Framework vs Competing Frameworks

| **Framework** | **Core Metaphor** | **Best For** | **Time-to-Demo** | **Production Maturity** |
|---|---|---|---|---|
| **Microsoft Agent Framework** | Dual-track: graph workflows + agent orchestration | Enterprise Azure deployments, .NET/Python, regulated industries | Medium (30–60 min) | High (GA April 2026) |
| **LangGraph** | Nodes and edges on a state graph | Production stateful workflows, human-in-the-loop, complex routing | Medium-high (45–90 min) | High (since 2023) |
| **CrewAI** | Role-based agent crews with task delegation | Rapid prototyping, team-based workflows, non-engineer use | Low (15–20 min) | Medium-high (since 2024) |
| **AutoGen** | Conversational multi-agent dialogue | Group chat, consensus-building, multi-party reasoning (maintenance mode) | Low-medium (20–40 min) | Medium (maintenance mode) |
| **Google ADK** | Workflow + LLM agents, GCP-native | GCP-deployed agents, Gemini integration, multi-agent collaboration | Medium (30–60 min) | Medium (growing rapidly) |
| **Mastra** | TypeScript-first composable agents | JS/TS teams, frontend-heavy stacks, Node.js environments | Low (15–30 min) | Medium |
| **OpenAI Agents SDK** | Minimal primitives: agents, handoffs, guardrails, tools | Simple OpenAI-native agents, fast prototyping | Very low (10–15 min) | Medium |
| **Pydantic AI** | Type-safe agents, dependency injection | Python-native teams, multi-provider, production testability | Low (15–25 min) | Medium-high (v1.0 Sept 2025) |
| **Haystack** | Component pipeline graph | Retrieval-heavy, document-centric enterprise AI | Medium (30–60 min) | High (since 2020) |
| **LlamaIndex** | Data pipeline + retrieval-first agents | Document-heavy RAG, enterprise data ingestion | Low-medium (20–40 min) | High for RAG; medium for orchestration |

### Microsoft Agent Framework vs. LangGraph

LangGraph models agents as nodes in a directed graph with shared state, emphasizing explicit control over every transition. It is the production-hardened choice for Python teams who need stateful, long-running workflows and are willing to invest in LangSmith for observability. LangGraph's time-travel debugging and graph visualization are meaningfully better than Agent Framework's current offering.

**Choose Microsoft Agent Framework when:** your infrastructure is Azure, you need .NET support, or your application mixes deterministic business workflows with emergent agent reasoning in a way that benefits from Agent Framework's dual-track architecture.

**Choose LangGraph when:** you are Python-only, you prioritize debugging and visualization tooling, or you need the deepest possible control over graph state transitions and the broadest community knowledge base. LangGraph is currently more battle-tested at production scale.

The key differentiating dimension is **debugging experience and visualization vs. enterprise platform depth**. LangGraph wins on developer experience; Agent Framework wins on enterprise plumbing.

### Microsoft Agent Framework vs. CrewAI

CrewAI's role-based crew metaphor is the easiest on-ramp in the agent framework space. A developer can define agents with human-readable roles and tasks in minutes. It is not a framework for production enterprise systems with compliance requirements — it is a framework for getting to a working prototype quickly and iterating fast.

**Choose Microsoft Agent Framework when:** you need session persistence, compliance hooks, middleware pipelines, .NET support, or Foundry deployment. Essentially, when you are building something that needs to survive contact with enterprise requirements.

**Choose CrewAI when:** speed to prototype is the primary concern, the team is non-technical, or the workflow maps cleanly to role delegation (research agent, writing agent, review agent). CrewAI's simplicity is a feature if your use case fits the mold.

The differentiating dimension is **enterprise readiness vs. prototyping speed**. Agent Framework is the right call once a proof of concept needs to become a product; CrewAI is the right call for building the proof of concept.

### Microsoft Agent Framework vs. AutoGen

AutoGen is the direct predecessor and is now in maintenance mode. Its core strength — expressive, multi-party conversational agent patterns — is preserved and extended in Agent Framework. For AutoGen users, the question is not whether to use Agent Framework, but when to migrate. New projects should not start on AutoGen.

**Choose Microsoft Agent Framework when:** starting any new project that would previously have used AutoGen.

**Choose AutoGen when:** you have a stable, working AutoGen application that is not growing and does not need new features. The migration cost may not be worth it for a system that works.

The differentiating dimension is **current development vs. maintenance mode**. AutoGen receives no new features. This is not a competition — it is a succession.

### Microsoft Agent Framework vs. Google ADK

Google ADK is the most direct architectural competitor. It also supports both workflow and LLM agent patterns, and it has tight cloud platform integration — but for GCP rather than Azure. ADK also has native support for the A2A (Agent-to-Agent) protocol, which Agent Framework is slower to adopt. Gemini integration is first-class in ADK in the same way Azure OpenAI integration is first-class in Agent Framework.

**Choose Microsoft Agent Framework when:** your cloud infrastructure is Azure or your organization uses Microsoft 365, Azure DevOps, or other Microsoft enterprise products.

**Choose Google ADK when:** you are deployed on GCP, need tight Gemini integration, or need the most mature A2A protocol support. ADK is growing rapidly in 2026.

### Microsoft Agent Framework vs. OpenAI Agents SDK

These two are the "platform-first" frameworks — both designed to provide the best experience within a specific vendor ecosystem. The comparison almost always reduces to which vendor you are committed to. Agent Framework provides significantly deeper enterprise plumbing: .NET + Python dual runtime, Azure Durable Functions checkpointing, Foundry deployment, Semantic Kernel middleware, compliance-grade session management, and a published AutoGen migration path. The OpenAI Agents SDK provides a dramatically simpler developer experience and first-class voice agent support, but is Python/TypeScript-only and accepts OpenAI model lock-in as the cost of platform integration.

**Choose Microsoft Agent Framework when:** your infrastructure is Azure, .NET support is required, enterprise compliance machinery is non-negotiable, or you need the dual-track workflow + agent orchestration architecture for mixed deterministic/agentic workloads.

**Choose the OpenAI Agents SDK when:** your stack is cloud-neutral or AWS-based, your team is Python or TypeScript (not .NET), you want the fastest initial development experience, or voice agent support is a first-class requirement.

The differentiating dimension is **enterprise Azure depth vs. simplicity and OpenAI platform integration**.

### Microsoft Agent Framework vs. Pydantic AI

Microsoft Agent Framework and Pydantic AI both target production Python teams but at different levels of the stack. Pydantic AI focuses on agent code quality: type-safe structured outputs, multi-provider model support, dependency injection for testability, and minimal framework overhead. Agent Framework focuses on enterprise orchestration infrastructure: session persistence, middleware pipelines, .NET support, Azure Durable Functions checkpointing, and Foundry deployment. Teams committed to Azure will often use both together — Pydantic AI-style agent patterns inside Semantic Kernel components within the Agent Framework orchestration layer.

**Choose Microsoft Agent Framework when:** Azure infrastructure, .NET support, enterprise compliance, or Semantic Kernel integration are hard requirements.

**Choose Pydantic AI when:** cloud neutrality is important, you need multi-provider model flexibility, your team values testability and minimal framework overhead over enterprise orchestration plumbing, or you are not on Azure.

The differentiating dimension is **enterprise Azure orchestration depth vs. cloud-neutral agent code quality**.

### Microsoft Agent Framework vs. LlamaIndex

Microsoft Agent Framework and LlamaIndex both operate at the enterprise-serious tier but address different parts of the stack. Agent Framework's core strengths are workflow orchestration, enterprise middleware, and Azure integration. LlamaIndex's core strengths are document ingestion, retrieval accuracy, and data pipeline sophistication. For Azure enterprises building applications where both complex orchestration and deep document retrieval matter, combining the two is the natural architecture — LlamaIndex for the data layer, Agent Framework for the orchestration layer. Choosing exclusively requires identifying which concern is primary.

**Choose Microsoft Agent Framework when:** orchestration complexity, enterprise compliance, .NET support, or Azure-native deployment are the primary requirements and retrieval needs can be met with Azure AI Search.

**Choose LlamaIndex when:** document parsing quality, retrieval accuracy, and data pipeline control are the primary differentiators — LlamaIndex's retrieval depth exceeds what Azure AI Search integration provides for complex document intelligence use cases. Also choose LlamaIndex when cloud neutrality is required.

The differentiating dimension is **enterprise Azure orchestration plumbing vs. retrieval pipeline depth and cloud neutrality**.

### Microsoft Agent Framework vs. Haystack

Both frameworks take production reliability seriously and both have strong enterprise adoption, but they serve different primary concerns. Agent Framework is an orchestration platform — it manages how agents, workflows, and state interact in complex multi-step processes on Azure. Haystack is a retrieval platform — it manages how documents are ingested, searched, ranked, and synthesized into generation outputs. For European enterprises, Haystack's cloud-neutral, EU-hosted architecture is often a hard requirement that Agent Framework's Azure-first model cannot satisfy. For organizations already on Azure, combining Agent Framework orchestration with Azure AI Search or Hayhooks-served Haystack pipelines is a practical production architecture.

**Choose Microsoft Agent Framework when:** Azure deployment, enterprise workflow orchestration, and .NET support are the primary requirements, and retrieval can be handled by Azure AI Search or an external tool.

**Choose Haystack when:** retrieval quality, hybrid search depth, pipeline explainability, or EU data sovereignty requirements make a dedicated retrieval framework necessary — and cloud neutrality is a constraint.

The differentiating dimension is **enterprise Azure orchestration vs. cloud-neutral retrieval pipeline sophistication**.

### Microsoft Agent Framework vs. Mastra

Agent Framework and Mastra serve different infrastructure profiles and different language ecosystems. Agent Framework is Python + .NET and Azure-committed, with its enterprise plumbing (Durable Functions, Foundry, Semantic Kernel middleware) deeply tied to the Azure stack. Mastra is TypeScript-native and cloud-agnostic, running on Cloudflare Workers, Vercel, and any Node.js environment. For organizations on Azure with .NET requirements, Agent Framework is the natural choice. For organizations running TypeScript services on cloud-neutral or non-Azure infrastructure, Mastra is the appropriate framework — it cannot match Agent Framework's enterprise compliance features, but its production tooling (memory, durable workflows, observability) is sufficient for most workloads outside regulated industries.

**Choose Microsoft Agent Framework when:** your infrastructure is Azure, .NET support is required, or enterprise compliance features are non-negotiable.

**Choose Mastra when:** your team is TypeScript-first, your infrastructure is cloud-neutral or non-Azure, and enterprise compliance plumbing is not a hard requirement.

The differentiating dimension is **Azure enterprise compliance depth vs. TypeScript cloud-agnostic production stack**.

---

## 9. Community and Market Position

### Key Metrics (as of May 2026)

- **`microsoft/autogen` GitHub stars:** ~43,000+ (the predecessor framework; placed in maintenance mode but star count reflects cumulative community interest in Microsoft's multi-agent work)
- **`microsoft/agent-framework` GitHub stars:** ~10,000+ (Growing rapidly post-1.0 GA; the repo was created in October 2025 and reached the top 10 of the AI agent frameworks GitHub ranking within six months of GA)
- **`microsoft/semantic-kernel` GitHub stars:** ~28,000+ (now in maintenance mode; represents the enterprise developer community migrating to Agent Framework)
- **PyPI downloads:** The `agent-framework` package crossed 1 million monthly downloads within the first three months of GA (April 2026)
- **Languages supported:** Python and .NET (C#) with full feature parity
- **Version:** 1.3.0 as of May 8, 2026; v1.0 GA on April 3, 2026; public preview October 1, 2025

### Company and Funding

Microsoft Agent Framework is a Microsoft Research and Microsoft Foundry product, developed by a dedicated engineering team spanning the former AutoGen and Semantic Kernel teams. Microsoft is a $3+ trillion market cap company; the framework is one component of a much larger Azure AI strategy. There is no external funding — this is internally funded R&D. The principal researchers behind AutoGen (Chi Wang, Gagan Bansal, and collaborators at Microsoft Research) are identified as the intellectual forebears of the project.

### Industry Recognition

Agent Framework was featured prominently at **Microsoft Build 2025** and **Microsoft Build 2026** as the centerpiece of Microsoft's agentic AI developer story. The 1.0 GA release was covered by Visual Studio Magazine, Cloud Wars, and DevClass as a significant milestone in the enterprise agent framework landscape. Multiple Gartner and Forrester analyst notes on enterprise agentic AI have named Microsoft Agent Framework alongside LangGraph as the two frameworks most likely to see meaningful enterprise adoption in 2026.

### Community Sentiment

The practitioner community is broadly positive about the framework's ambition and enterprise features, with the most common criticisms being Azure lock-in and complexity for simple use cases. On Reddit and Discord, the prevailing view is: "best-in-class if you're on Azure; hard to justify if you're not." The AutoGen community's migration experience has been mixed — developers who appreciated AutoGen's conversational flexibility appreciate that the patterns are preserved, but those who found AutoGen's architecture approachable report that Agent Framework's additional abstractions add cognitive overhead before they add value.

### Market Context

Microsoft Agent Framework occupies the enterprise-Azure segment of the agent framework market, competing most directly with LangGraph at the production end and Google ADK at the platform-integration end. It does not compete with CrewAI (different audience), Mastra (different language ecosystem), or OpenAI Agents SDK (different abstraction level). The trajectory is clearly upward — Microsoft's distribution reach, enterprise relationships, and investment in Foundry create strong adoption tailwinds. The framework's primary risk is that Azure-native teams who would be natural adopters are also being courted by Azure-native no-code and low-code tools (Copilot Studio, Agent 365) that require no framework-level development at all.

---

## 10. Pricing

Microsoft Agent Framework itself is free, open-source (MIT License), and can be run on any infrastructure. You can build and run agents locally against the OpenAI API or Ollama without spending a dollar on Microsoft services. Where cost enters the picture is the managed platform layer: **Foundry Agent Service** for hosted deployment, **Foundry Tools** for managed MCP servers, and **Azure model endpoints** for LLM inference. The pricing model is consumption-based — you pay for what your agents actually use, not for seats or a platform subscription.

| **Tier** | **Price** | **What You Pay For** | **Included** | **Target User** | **Support** |
|---|---|---|---|---|---|
| **Open Source (Self-Hosted)** | Free | Infrastructure + LLM API costs only | Full SDK, all agent types, all orchestration patterns | Solo developers, startups, non-Azure teams | Community (GitHub Issues, Discord) |
| **Foundry Agent Service (Pay-as-you-go)** | $0 for orchestration + model consumption | Model tokens ($2.50–$30/M input, $15–$180/M output depending on model) | Managed containers, auto-scaling, OpenTelemetry observability, Foundry portal | Teams deploying on Azure | Standard Azure support tiers |
| **Foundry Tools (Add-on)** | Varies by tool / consumption | Per-invocation charges for managed MCP tool connections (Graph, SharePoint, Bing, etc.) | Pre-built enterprise tool integrations | Organizations using Microsoft data sources | Standard Azure support |
| **Agent 365** | $15/user/month (GA May 1, 2026) | Per-user per-month; also bundled in Microsoft 365 E7 ($99/user/month) | Enterprise agent experiences across M365 suite | Enterprise end-user agent deployment | Enterprise Microsoft support |
| **Enterprise Agreement / EA** | Custom (contact sales) | Volume pricing, reserved capacity, private deployments | Custom SLAs, dedicated compliance support, private regions | Large enterprises, government, regulated industries | Premier / Unified support |

### Open Source Tier

The open-source tier is genuinely unlimited. There are no feature restrictions, rate limits, or telemetry requirements in the SDK itself. Teams can build and run sophisticated multi-agent systems in their own infrastructure, against any supported model provider, at any scale, without ever interacting with Microsoft's commercial platform. The cost is purely infrastructure (compute to run the application) and LLM API fees. This is the right choice for organizations with GCP or AWS infrastructure, cost-sensitive teams, and developers who want to evaluate the framework without commitment.

### Foundry Agent Service (Pay-as-you-go)

The managed runtime tier costs nothing for agent orchestration itself — Microsoft does not charge for scheduling, routing, or executing agent logic. All costs come from the underlying model inference and optional tool invocations. Standard model pricing through Azure AI Foundry is approximately $2.50 per million input tokens and $15 per million output tokens for GPT-4o-class models; Pro-tier models (for deep analytical workloads) run approximately $30/$180 per million tokens. These are list prices; Azure enterprise agreements may include committed-use discounts. The key advantage of this tier is that you inherit Azure's observability, security, and scaling infrastructure without operating it yourself.

### Foundry Tools

Each Foundry-managed MCP tool integration carries its own consumption-based pricing that varies by tool type and invocation volume. These are separate line items from model token costs. For most enterprise use cases, Foundry Tools charges are secondary to model inference costs, but they can become significant in high-volume agentic workflows that frequently invoke knowledge retrieval tools.

### Agent 365

Agent 365 is not the Agent Framework SDK — it is Microsoft's enterprise end-user agent experience, built on top of the same underlying infrastructure. At $15/user/month (bundled in Microsoft 365 E7 at $99/user/month as of May 2026), it provides pre-built agent capabilities within the Microsoft 365 suite. It is relevant for buyers who want to deploy agent experiences to end users without building on the SDK; it is not relevant for developers building custom agent applications.

### Real-World Cost Scenarios

**Solo developer / side project:** $0/month for the SDK. If running against OpenAI's API directly, a light development workload (a few thousand agent turns/day) might cost $5–$20/month in model tokens. No Foundry costs unless opted in.

**Small startup (3–5 people):** Likely on Foundry pay-as-you-go for deployment simplicity. Model costs depend heavily on agent design and invocation frequency. A startup running moderate-volume agents (10,000–50,000 LLM calls/month, GPT-4o pricing) should expect $250–$1,500/month in model inference; Foundry Agent Service hosting adds minimal overhead at this scale.

**Mid-size team in production (20–50 people):** Active production deployment with Foundry Agent Service, Foundry Tools for a few enterprise data integrations, and Application Insights for monitoring. Total monthly costs likely $3,000–$15,000+, primarily driven by model inference volume and the number of expensive tool-calling patterns. An Azure enterprise agreement likely applies at this scale, providing meaningful discounts.

**Large enterprise (100+ people):** Enterprise Agreement pricing with reserved capacity, dedicated compliance support, and potentially private Azure regions. Annual costs vary widely by usage but commonly run $50,000–$500,000+/year for large-scale agentic deployments; the enterprise contract typically includes Foundry, Azure OpenAI, and associated Azure services under unified pricing.

### Pricing Caveats

Pricing figures above are based on publicly available Azure pricing pages and third-party analyses as of May 2026. Azure pricing changes frequently and varies by region, tier, and enterprise agreement terms. Agent 365 pricing was confirmed at $15/user/month for GA on May 1, 2026, but bundle pricing through Microsoft 365 E7 should be verified with a Microsoft account representative. Foundry Tools per-invocation pricing requires consulting the Azure portal or Microsoft sales for current rates on specific tool integrations.

### Self-Host Option

A fully self-hosted deployment — running the open-source SDK on your own Kubernetes cluster or Azure App Service, with your own OpenTelemetry collector and vector store — is fully supported and requires only infrastructure costs and LLM API fees. The primary trade-offs are: you must operate the deployment infrastructure yourself; you lose Foundry's managed observability dashboards and auto-scaling; and some advanced Foundry-native features (like the Foundry Tools MCP marketplace and Foundry portal trace viewer) are not available. For teams with strong DevOps capabilities and a preference for infrastructure ownership, self-hosted deployment on Azure Container Apps is a well-documented path.

---

## 11. Summary and Verdict

**Positioning statement:** Microsoft Agent Framework trades breadth of community adoption and prototyping speed for depth of enterprise plumbing and Azure integration — it is the most production-ready agent framework for Azure-committed organizations and the most over-engineered one for everyone else.

### When to Choose Microsoft Agent Framework

- Your infrastructure is Azure and you intend to run agents in Foundry Agent Service or Azure Durable Functions
- Your team writes in .NET (C#) — no other major agent framework provides production-grade .NET support
- Your use case is in a regulated industry where compliance hooks, audit trails, session persistence, and governance tooling are non-negotiable
- You need to mix deterministic business workflows with emergent LLM-driven orchestration in the same application
- You are migrating from AutoGen or Semantic Kernel and want to preserve institutional investment in those patterns
- You need native MCP integration and want to leverage the Foundry Tools managed tool marketplace

### When Not to Choose Microsoft Agent Framework

- Your infrastructure is GCP or AWS — you can use the SDK but will sacrifice most of its distinctive enterprise features
- Your team is Python-only, already invested in LangGraph's ecosystem, and values debugging experience and visualization above enterprise plumbing
- You need a fast proof-of-concept — the complexity overhead pays off in production, not in a hackathon
- You are a solo developer or small startup with no Azure footprint and no immediate enterprise compliance requirements
- Your project is TypeScript/JavaScript — Mastra is the right choice for JS-first teams

### Closing Perspective

Microsoft Agent Framework occupies a clear and defensible position in the 2026 agent framework landscape: it is the enterprise-Azure framework. It does not attempt to be the simplest, the fastest to prototype with, or the most model-agnostic. It attempts to be the framework you choose when you need your agent system to run reliably in production, in a regulated environment, on Azure, with enterprise-grade observability and governance. In that specific context, it has no peer.

The more interesting competitive question is not whether Agent Framework is better than LangGraph or CrewAI — it is whether enterprises will build agentic systems at the SDK level at all, or whether no-code platforms like Copilot Studio and Agent 365 will absorb the majority of enterprise agent demand before framework-level adoption can compound. Microsoft has hedged both bets simultaneously, which is its structural advantage: if enterprises want to build, they use Agent Framework; if they want to buy, they use Agent 365. No other vendor in the space controls both ends of that spectrum.

---

## Sources

- [Microsoft Agent Framework Overview — Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Introducing Microsoft Agent Framework: The Open-Source Engine for Agentic AI Apps — Microsoft Foundry Blog](https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/)
- [Microsoft Agent Framework Version 1.0 — Microsoft Agent Framework Blog](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/)
- [GitHub — microsoft/agent-framework](https://github.com/microsoft/agent-framework)
- [AutoGen v0.4: Reimagining the Foundation of Agentic AI — Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)
- [Microsoft Agent Framework Workflows — Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/workflows/)
- [AutoGen to Microsoft Agent Framework Migration Guide — Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/)
- [Microsoft Agent Framework: The Production-Ready Convergence of AutoGen and Semantic Kernel — European AI & Cloud Summit](https://cloudsummit.eu/blog/microsoft-agent-framework-production-ready-convergence-autogen-semantic-kernel)
- [Microsoft Ships Production-Ready Agent Framework 1.0 for .NET and Python — Visual Studio Magazine](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx)
- [The Future of Agentic AI: Inside Microsoft Agent Framework 1.0 — Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/the-future-of-agentic-ai-inside-microsoft-agent-framework-1-0/4510698)
- [Foundry Agent Service + Microsoft Agent Framework Explained — Microsoft Community Hub](https://techcommunity.microsoft.com/blog/microsoftmechanicsblog/foundry-agent-service--microsoft-agent-framework-explained/4511661)
- [Orchestrating Multi-Agent Intelligence: MCP-Driven Patterns in Agent Framework — Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/orchestrating-multi-agent-intelligence-mcp-driven-patterns-in-agent-framework/4462150)
- [Foundry Agent Service — Pricing | Microsoft Azure](https://azure.microsoft.com/en-us/pricing/details/foundry-agent-service/)
- [Microsoft Foundry — Pricing | Microsoft Azure](https://azure.microsoft.com/en-us/pricing/details/microsoft-foundry/)
- [Tracking Every Token: Granular Cost and Usage Metrics for Microsoft Foundry Agents — Microsoft Community Hub](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/tracking-every-token-granular-cost-and-usage-metrics-for-microsoft-foundry-agent/4503143)
- [The Hidden Cost of Microsoft Agents: Metered Pricing, Consumption Billing, and Budget Surprises — TeamsFox](https://teamsfox.com/hidden-cost-microsoft-agents-metered-pricing-consumption-billing)
- [Exploring Microsoft Agent Framework — AI Agents for Beginners (Microsoft)](https://microsoft.github.io/ai-agents-for-beginners/14-microsoft-agent-framework/)
- [Build a Real-World Example with Microsoft Agent Framework, Foundry, MCP and Aspire — Microsoft for Developers](https://developer.microsoft.com/blog/build-a-real-world-example-with-microsoft-agent-framework-microsoft-foundry-mcp-and-aspire)
- [AI Agent Frameworks Tier List 2026 — Paperclipped](https://www.paperclipped.de/en/blog/ai-agent-frameworks-tier-list-2026/)
- [LangGraph vs CrewAI vs AutoGen: Which Agent Framework Should You Actually Use in 2026? — Medium / Data Science Collective](https://medium.com/data-science-collective/langgraph-vs-crewai-vs-autogen-which-agent-framework-should-you-actually-use-in-2026-b8b2c84f1229)
- [AI Agent Frameworks in 2026: 8 SDKs, ACP, and the Trade-offs Nobody Talks About — Morph LLM](https://www.morphllm.com/ai-agent-framework)
- [Semantic Kernel + AutoGen = Open-Source Microsoft Agent Framework — Visual Studio Magazine](https://visualstudiomagazine.com/articles/2025/10/01/semantic-kernel-autogen--open-source-microsoft-agent-framework.aspx)
- [Microsoft Agent Framework — Building Blocks for AI Part 3 — .NET Blog](https://devblogs.microsoft.com/dotnet/microsoft-agent-framework-building-blocks-for-ai-part-3/)
- [What Is Microsoft Agent Framework? A 2026 Guide for Enterprise AI Teams — Nerova AI](https://nerova.ai/blog/what-is-microsoft-agent-framework-enterprise-ai-2026)
