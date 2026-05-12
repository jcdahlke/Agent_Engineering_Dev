# Strands Agents — Deep Research Report

**Research Date:** May 12, 2026  
**Subject:** Strands Agents — Architecture, Adoption, Use Cases, and Competitive Landscape

---

## Table of Contents

1. [What Is Strands?](#1-what-is-strands)
2. [How It Works — Architecture Deep Dive](#2-how-it-works--architecture-deep-dive)
3. [The Strands Ecosystem](#3-the-strands-ecosystem)
4. [Who Uses Strands?](#4-who-uses-strands)
5. [Industries and Use Cases](#5-industries-and-use-cases)
6. [Why People Choose Strands](#6-why-people-choose-strands)
7. [Why People Don't Choose Strands](#7-why-people-dont-choose-strands)
8. [Strands vs Competing Frameworks](#8-strands-vs-competing-frameworks)
9. [Community and Market Position](#9-community-and-market-position)
10. [Pricing](#10-pricing)
11. [Summary and Verdict](#11-summary-and-verdict)
12. [Sources](#sources)

---

## 1. What Is Strands?

Strands Agents is an open-source Python SDK that takes a **model-driven approach** to building AI agents, meaning the LLM itself drives the agent loop rather than hand-coded orchestration graphs or rigid workflow rules. Instead of requiring developers to define every possible state transition or role assignment in advance, Strands hands the wheel to the model and gets out of the way: you provide a system prompt, a set of tools, and a model, and the agent figures out what to call and when.

AWS announced Strands as a public preview in May 2025 and released version 1.0 in July 2025. The framework was led by Ryan Coleman (Product Manager) and Belle Guttman (Engineering Lead), who heads the Agentic AI Engineering teams responsible for the Strands SDK and agentic chat in Q Developer products. The origin story is pragmatic rather than academic: AWS internal teams building agents for Amazon Q Developer, AWS Glue, and Kiro found that modern LLMs' native tool-use and reasoning capabilities had made complex orchestration frameworks unnecessary. Where it previously took months to go from agent prototype to production in Q Developer, Strands cut that to days. AWS open-sourced the SDK immediately, licensed under **Apache 2.0**, with genuine community contributions from day one.

The core mental model Strands uses is the **agent loop**: a tight cycle in which the model reads the full conversation history, decides whether to call a tool or produce a final answer, executes any requested tool, and loops again with the result appended. This loop runs autonomously until the model signals completion. Version 1.0 extended this single-agent loop into four multi-agent coordination patterns — Agents-as-Tools, Handoffs, Swarms, and Graphs — without abandoning the simplicity of the original design.

As of early 2026, Strands has accumulated over 5 million total PyPI downloads, over 1.4 million weekly downloads, and more than 2,000 GitHub stars since its May 2025 preview launch. It ships with 40+ pre-built tools and supports model providers including Amazon Bedrock (default), Anthropic, OpenAI, Gemini, LiteLLM, Ollama, Cohere, Mistral, and Meta Llama, among others.

> *"Strands Agents is an open source SDK that takes a model-driven approach to building and running AI agents in just a few lines of code."* — AWS Open Source Blog, May 2025

In one sentence: Strands is AWS's open-source answer to the question of what happens when you trust the model to orchestrate itself — and it works remarkably well for teams already operating in the AWS ecosystem.

---

## 2. How It Works — Architecture Deep Dive

### Core Primitives

Strands is built on three primitives for single-agent use: the **Model**, the **Tool**, and the **Agent**. Everything else — conversation history, tool execution, loop control — is managed internally.

- **Model**: Any supported LLM provider configured via a model object. The default is Claude 3.7 Sonnet accessed via Amazon Bedrock, but the model can be swapped by passing a different `BedrockModel`, `AnthropicModel`, `OpenAIModel`, `LiteLLMModel`, or custom provider instance to the Agent constructor.
- **Tool**: A Python function decorated with `@tool` (or imported from the `strands_tools` package). The decorator extracts the function signature and docstring to generate a tool spec the model understands. Tools can be synchronous or asynchronous.
- **Agent**: The central class. It holds a reference to the model, a system prompt, and a list of tools. When invoked, it runs the agent loop until the model signals it is done.

### The Agent Loop

On each iteration of the agent loop, the model receives the complete conversation history — every prior message, tool call, and tool result — and decides one of two things: call a tool, or produce a final answer. If the model requests a tool call, Strands executes it and appends the result to the conversation, then runs the model again. This continues until the model outputs a final message. The loop is fully transparent: all iterations fire OpenTelemetry events, so traces are captured automatically.

Strands does not expose a state machine, a graph definition language, or a role configuration file at the single-agent level. The model's own chain-of-thought replaces those constructs. This is the central design bet: that modern LLMs (Claude 3.7 Sonnet, GPT-4o, Llama 4, etc.) are smart enough to self-route through multi-step tasks without explicit wiring.

### Minimal Working Example

```python
from strands import Agent
from strands_tools import calculator, web_search

# Define the agent with a model, prompt, and tools
agent = Agent(
    system_prompt="You are a financial research assistant.",
    tools=[calculator, web_search]
)

# Invoking the agent starts the loop; it runs until a final answer is produced
result = agent("What was Germany's GDP in 2024, and what is its square root?")
print(result)
```

The model will call `web_search` to retrieve the GDP figure, then call `calculator` to compute the square root, then synthesize both results into a final answer — all without any explicit routing logic.

### Multi-Agent Primitives (v1.0)

Version 1.0 introduced four primitives for orchestrating multiple agents, designed to layer on top of the single-agent model rather than replace it:

**Agents-as-Tools**: A specialized agent is wrapped with the `@tool` decorator and passed into an orchestrator agent's tool list. The orchestrator calls specialist agents the same way it calls any other tool — by including them in its tool list. This implements hierarchical delegation: the orchestrator stays in control of the overall task and consults specialists on demand.

**Handoffs**: An agent explicitly transfers control to another agent or to a human using the built-in `handoff_to_user` tool. The full conversation context is preserved during the handoff. This is the pattern for human-in-the-loop workflows where an agent needs to pause and ask for input before proceeding.

**Swarms**: A `Swarm` class wraps a list of specialized agents that coordinate through **shared memory**. Each agent can read and write to the shared memory store, allowing the team to build on each other's outputs. The swarm self-organizes; there is no designated coordinator. This is the right pattern for brainstorming, collaborative research, and parallel problem decomposition — and the wrong pattern for tasks that require a strict ordering of steps.

**Graphs**: A `GraphBuilder` lets you define explicit workflows with `add_node`, `add_edge`, and conditional routing functions. This brings LangGraph-style determinism to Strands when you need it — approval chains, quality gates, or multi-step processes with known branching logic.

The four patterns are composable: a swarm can be a node inside a graph, and any node in a graph can be an agent that uses other agents as tools.

### A2A Protocol

Version 1.0 also ships native support for the **Agent-to-Agent (A2A) protocol**, an open cross-framework standard. Any Strands agent can be wrapped with an `A2AServer` to become a network-accessible endpoint with an auto-generated agent card. Strands agents can also connect to external A2A-compatible agents built in other frameworks via `A2AClientToolProvider`, enabling cross-organization and cross-framework multi-agent systems.

### Error Handling, Retries, and Memory

Strands does not implement automatic retry logic at the framework level — retry behavior is delegated to individual tool implementations or the model provider SDK. For context management, Strands passes the full conversation history on every iteration, so short-term memory is the conversation itself. Long-term memory requires explicit use of a memory tool (the `strands_tools` package ships a `memory` tool) or integration with Amazon Bedrock AgentCore Memory. For durable session state, the `SessionManager` abstraction (introduced in v1.0) allows agents to persist and restore conversation history to file, S3, or custom backends, enabling agents to survive compute restarts and scaling events.

---

## 3. The Strands Ecosystem

### Amazon Bedrock AgentCore

The primary managed platform for Strands is **Amazon Bedrock AgentCore**, AWS's enterprise hosting solution for production agent workloads. AgentCore is modular — each component can be used independently — and covers seven distinct capabilities: Runtime (container execution), Memory (short-term and long-term), Identity (authentication and authorization), Observability (tracing and logging), Gateway (tool and API proxy), Browser (cloud browsing), and Code Interpreter (sandboxed code execution). Strands agents deploy to AgentCore Runtime in as few as three API calls, per a 2026 update that simplified the deployment path significantly.

### Deployment Targets

Beyond AgentCore, Strands agents can be deployed to AWS Lambda, AWS Fargate, Amazon EC2, Amazon EKS, Docker, or Kubernetes. Reference implementations covering Lambda and Fargate are maintained in the official GitHub samples repository. Because Strands is a standard Python package with no AWS-specific runtime requirement, it also runs on-premises or on competing clouds — though AWS integration features (IAM, KMS encryption, VPC isolation) only apply to AWS deployments.

### Observability

Strands ships **first-class OpenTelemetry tracing** out of the box: every agent loop iteration, tool call, and model request fires a trace event. These traces can be forwarded to AWS CloudWatch, Langfuse, Arize AX, or any OpenTelemetry-compatible backend. AWS published a dedicated blog post on the Strands + Arize AX integration for production observability. The Strands Evals framework (a separate package) provides structured evaluation at the session, trace, and tool levels, addressing the difficulty of measuring agent quality at each layer.

### MCP Integration

Strands has native support for the **Model Context Protocol (MCP)**, which means any MCP server's tools are automatically available to a Strands agent without additional integration work. This gives Strands access to a rapidly growing ecosystem of MCP-compatible tools across a wide range of services.

### Visual and Debug Tooling

There is no official visual IDE or Studio-style drag-and-drop interface for Strands as of mid-2026. The `strands.my` dashboard exists as a community project for exploring the agent ecosystem from GitHub and PyPI metrics. Debugging relies on standard Python logging (set `logging.getLogger("strands.multiagent")` to `DEBUG`) and the OpenTelemetry traces forwarded to your chosen observability backend.

### Cloud Provider Integrations

Deep integration is with AWS: Bedrock, S3 (for session storage), IAM (for agent identity), CloudWatch (for logs), Lambda/Fargate/ECS (for compute). Third-party cloud integrations are available through LiteLLM (which can proxy to Azure, GCP, and dozens of other providers) and community-contributed model providers.

---

## 4. Who Uses Strands?

| **Company / Team** | **Use Case** |
|---|---|
| **Amazon Q Developer** | Internal agentic AI assistant for software development; adopted Strands to cut agent deployment time from months to days and weeks |
| **AWS Glue** | Data integration workflows with AI-driven task routing and transformation agents |
| **VPC Reachability Analyzer** | Network diagnostics agents that reason over AWS infrastructure topology to identify connectivity issues |
| **Kiro (AWS IDE)** | AI-powered coding assistant with agent loop architecture for in-editor task automation |
| **Amazon Transform** | Application modernization at scale — agentic workflows that analyze and rewrite legacy codebases using Strands and AgentCore |
| **Accenture** | Enterprise AI agent deployments across industries; contributed code to the Strands SDK and participating in multi-year Anthropic partnership that references Strands as preferred framework |
| **PwC** | Finance and life sciences enterprise agent deployments under a collaboration with Anthropic; using Strands-based agents for mission-critical regulated workflows |
| **Anthropic** | Contributed the direct Anthropic API model provider integration to Strands; Claude models are the default Strands model via Bedrock |
| **Meta** | Contributed the Meta Llama API model provider integration; Llama 4 available as a Strands model |
| **Cohere** | Contributed Cohere model provider; enterprise NLP workloads on Strands |
| **Writer** | Contributed Writer model provider; enterprise content and document generation agents |
| **Langfuse** | Contributed observability integration; Strands agents export traces to Langfuse for monitoring |
| **mem0.ai** | Contributed memory integration; long-term memory layer for Strands agents |
| **Tavily** | Contributed web search tool integration; Strands-powered research agents |

---

## 5. Industries and Use Cases

### Software Development and DevOps

The most validated production use case for Strands is AI-assisted software development, demonstrated by Amazon Q Developer and Kiro. The agent loop excels at multi-step coding tasks — understanding requirements, searching documentation, writing code, running tests, and iterating — because these tasks require dynamic tool selection over many turns rather than a fixed workflow. AWS reports that teams shipping via Q Developer reduced agent time-to-production from months to days after adopting Strands internally.

### Cloud Infrastructure Management

VPC Reachability Analyzer is a canonical example of using Strands for infrastructure reasoning: the agent must interpret network topology data, call multiple AWS APIs, reason about firewall rules and routing tables, and synthesize a diagnostic report — all dynamically. Similarly, the "FinOps Intelligent Agents" pattern published by AWS shows Strands agents analyzing cloud cost data across accounts and surfacing optimization recommendations automatically.

### Application Modernization

AWS's **Amazon Transform** service, which automates legacy code migration (particularly Java upgrades), uses Strands to orchestrate the analysis, refactoring, and validation steps involved in moving large codebases to modern targets. The multi-agent patterns in Strands 1.0 — particularly Graphs for deterministic approval steps — are well-suited to this workflow, where some stages must run in order while others can parallelize.

### Financial Services

PwC's deployment of enterprise AI agents in finance under the Anthropic collaboration focuses on automating workflows in highly regulated environments: meeting action capture, communication drafting, follow-through tracking, and financial metric calculation. Strands's native AWS IAM integration and VPC isolation make it compliant with enterprise security requirements in financial services without additional configuration.

### Life Sciences

PwC and Anthropic's life sciences deployment uses Strands agents for mission-critical, regulated workflows where auditability matters. The built-in OpenTelemetry tracing provides the full execution trace needed for compliance review — every tool call, every model decision, every output is captured.

### Customer Service and Operations

The multi-agent swarm and handoff patterns in Strands 1.0 map naturally to customer service: a triage agent handles initial classification, specialist agents address domain-specific questions, and `handoff_to_user` transfers control when human judgment is needed. An unnamed air carrier was cited in AWS marketing as using agentic workflows for common customer transactions like rebooking flights and rerouting bags, though the specific framework was not confirmed as Strands.

### Research and Knowledge Work

The Swarm primitive — multiple agents sharing a memory store and building on each other's outputs — is designed for collaborative research tasks. A typical pattern pairs a Researcher agent, an Analyst agent, and a Writer agent, where the team divides and conquers a topic without a central coordinator dictating each step. The shared memory approach allows each agent to read prior contributions and extend rather than duplicate them.

### Robotics and Physical AI

AWS published a 2025 blog post on using Strands with Bedrock AgentCore, Claude 4.5, NVIDIA GR00T, and Hugging Face LeRobot to build agents for physical AI systems. This is an emerging use case, but the combination of async execution, streaming, and real-time bidirectional audio in Strands makes it technically capable of handling edge-cloud coordination for robotic agents.

---

## 6. Why People Choose Strands

### Minimal Boilerplate to a Working Agent

Strands is genuinely competitive on time-to-first-working-agent. A functional agent with tools runs in fewer than 10 lines of Python. Compare this to LangGraph, which requires defining nodes, edges, a state schema, and a graph compile step before running anything. For teams evaluating frameworks on a deadline, Strands wins the prototype sprint almost every time. The `@tool` decorator pattern is intuitive — any function with a docstring becomes a tool the model can call.

### Model-Agnostic by Default

Despite being an AWS product, Strands is genuinely model-agnostic. The same agent code runs against Claude, GPT-4o, Gemini, Llama, and a dozen other providers. Switching models requires changing one argument in the Agent constructor, not rewriting orchestration logic. This matters for teams that want to benchmark models or migrate away from a provider without framework-level rewrites. The LiteLLM backend provides a catch-all for providers that don't have first-party Strands integrations.

### AWS Ecosystem Integration Without Lock-In

For teams already running on AWS, Strands is deeply integrated: IAM for identity, KMS for encryption, VPC for isolation, S3 for session storage, CloudWatch for logging, and Bedrock for model access. These integrations are zero-configuration when running inside AWS. But because the SDK itself has no AWS runtime requirement, teams that later want to move to another cloud or run on-prem don't face a framework migration — they just reconfigure the deployment target and model provider.

### Four Multi-Agent Patterns That Compose Naturally

The Agents-as-Tools, Handoffs, Swarms, and Graphs primitives cover the four primary multi-agent design patterns without requiring a separate framework for each. More importantly, they compose: a Swarm can sit inside a Graph, a Graph node can call an agent-as-tool, and any of them can expose themselves via A2A. Most competing frameworks force a choice between one pattern and everything else.

### A2A and MCP as First-Class Citizens

Strands ships native support for both A2A (cross-framework agent interop) and MCP (tool ecosystem access). These protocols are growing in adoption across the industry, and Strands's first-class support means Strands agents can participate in multi-framework deployments without a custom adapter layer. This is particularly relevant for enterprises deploying agents across multiple vendor platforms.

### Production Signals from Internal AWS Use

Strands was not released as a research project — it was hardened in production at Amazon before it was open-sourced. Amazon Q Developer, AWS Glue, VPC Reachability Analyzer, and Kiro are not toy workloads. The session management, async support, and observability features in v1.0 were added because AWS internal teams needed them for real production deployments, not as speculative roadmap items. That backstory matters for enterprise engineering teams evaluating production readiness.

### Built-In Observability

First-class OpenTelemetry integration means no additional instrumentation work to get traces. Every agent run generates a complete trace of model calls, tool invocations, and their results. Combined with the Strands Evals framework, teams can instrument both individual tool quality and end-to-end session quality from day one — rather than retrofitting observability after the fact.

---

## 7. Why People Don't Choose Strands

### AWS-Centric Gravitational Pull

While Strands is technically model-agnostic and cloud-agnostic, the defaults all point to AWS: the default model is Claude 3.7 Sonnet via Amazon Bedrock, the easiest deployment path runs to AgentCore or Lambda, and the richest integrations (IAM, KMS, S3 session storage, CloudWatch) are all AWS services. Teams running on Azure or GCP can use Strands, but they will spend time working around defaults that assume Bedrock. The perception — and partially the reality — is that Strands is an AWS framework that happens to support other clouds, not a genuinely neutral framework.

### Insufficient Control for Deterministic Workflows

The model-driven loop is a liability when you need guarantees. If a regulated financial workflow requires a specific sequence of steps — check balance, verify identity, confirm authorization, execute transaction — handing that control to the model introduces unpredictability. Strands's Graph primitive mitigates this, but teams with strict determinism requirements will find LangGraph's explicit state machine more natural and less likely to surprise them in production. The community consistently notes that "the model drives itself" is a feature for exploratory tasks and a risk for compliance-critical ones.

### Young Ecosystem and Limited Battle-Testing (Beyond AWS)

Strands launched in May 2025. As of mid-2026, it has about a year of public history. Most of the documented production deployments are internal AWS services, with limited third-party case studies at scale. LangGraph and CrewAI have two to three years of community tutorials, Stack Overflow answers, edge-case documentation, and third-party plugins. Strands's documentation is solid for core use cases, but teams that hit an unusual error are more likely to find themselves without a community answer. This gap is closing, but it is real.

### Context Window Pressure at Scale

Because the agent loop feeds the full conversation history — every tool call and result — to the model on each iteration, complex multi-step tasks can hit context limits faster than frameworks that use explicit state management and selective context passing. LangGraph lets you store large intermediate results in a state store and pass only what's needed to the model. Strands's approach works well for medium-complexity tasks but can become expensive and unreliable for tasks with dozens of tool calls or very large tool outputs.

### Swarm Unpredictability

The Swarm primitive is the highest-risk pattern in Strands's toolkit. Multiple agents coordinating through shared memory with no central director can produce excellent results — and can also cycle, duplicate work, or contradict each other in ways that are difficult to debug. There is no built-in mechanism to detect or break cycles in a swarm, and the `DEBUG` logging output for swarm coordination is verbose but not always diagnostic. Teams that need predictable multi-agent behavior should use Graphs, not Swarms.

### API Throttling on AWS ConverseStream

Users experimenting at scale with Strands on Bedrock's ConverseStream API have run into throttling limits where queries hang for extended periods, indicating capacity constraints. This is an AWS infrastructure limitation rather than a Strands SDK issue, but the default Strands configuration routes all traffic through ConverseStream, so the throttling surfaces as a Strands problem from the developer's perspective. Teams with high-throughput workloads need to implement retry logic or provision provisioned throughput on Bedrock.

### No Visual IDE or Low-Code Interface

Strands is a code-first framework with no official visual debugging or no-code authoring interface. There is no equivalent of LangSmith's trace replay UI or CrewAI's Studio. For engineering teams this is fine, but for organizations where business analysts or non-engineers need to author or inspect agent workflows, Strands offers no path. Tools like Dify or n8n serve that audience; Strands does not.

### Evaluation Complexity

Evaluating Strands agents at scale is genuinely hard. Session-level success can mask tool-level failures — an agent might reach the right answer through incorrect intermediate steps, or achieve high tool accuracy while failing to synthesize a coherent final answer. The Strands Evals framework helps, but AWS's own blog post on evaluation notes that it requires instrumentation at three separate levels: session, trace, and tool. For teams without dedicated eval infrastructure, this is a significant ongoing maintenance burden.

---

## 8. Strands vs Competing Frameworks

| **Framework** | **Core Metaphor** | **Best For** | **Time-to-Demo** | **Production Maturity** |
|---|---|---|---|---|
| **Strands** | Model-driven loop | AWS-ecosystem agents, rapid prototyping, model-agnostic multi-agent | Minutes | High (v1.0, July 2025; AWS internal use) |
| **LangGraph** | Stateful graph / state machine | High-stakes deterministic workflows, human-in-the-loop, complex branching | Hours | Very High (2+ years, enterprise-validated) |
| **CrewAI** | Role-based crew | Role-oriented multi-agent, business user prototyping | Minutes | Moderate (strong OSS, weaker production controls) |
| **AutoGen** | Conversation / message passing | Historical context, research; now in maintenance mode | Minutes | Maintenance mode (merged with Semantic Kernel, Oct 2025) |
| **OpenAI Agents SDK** | Explicit handoffs | OpenAI-stack teams, simple multi-agent handoffs | Minutes | Moderate (launched March 2025, rapid improvement) |
| **Google ADK** | Hierarchical agent tree | Gemini-native apps, Vertex AI deployments, cross-framework A2A | Hours | Early (launched April 2025, growing fast) |
| **Mastra** | TypeScript-native graph | JS/TS-first teams, Next.js/Node.js integration | Minutes | Early (growing in TypeScript ecosystem) |

### Strands vs LangGraph

LangGraph treats agent execution as a directed graph with an explicit state schema: nodes transform state, edges define transitions, and you compile the graph before running it. This gives you total control over what happens at every step and makes it straightforward to insert checkpoints, human approvals, and conditional routing at precise moments. Strands gives you none of that explicit wiring — the model decides what to do next.

Choose Strands when: your task is exploratory or open-ended, you want rapid iteration without graph compilation overhead, and you're willing to accept that the model may route through steps in unexpected orders.

Choose LangGraph when: your workflow has regulatory compliance requirements, strict step ordering, or complex branching that must be auditable — or when you need LangSmith's observability and replay tooling. LangGraph remains the standard for high-stakes agent production workloads as of mid-2026 and is in active, heavy development.

The key differentiating dimension is control vs. speed. Strands wins the prototype; LangGraph wins the audit.

### Strands vs CrewAI

CrewAI frames agents as specialized workers in a managed crew, assigning them roles, backstories, and goals. The abstraction is intuitive and accessible to non-engineers, and CrewAI has a larger tutorial ecosystem and community than Strands. But CrewAI's role abstraction is fixed: you define the crew structure upfront, and agents can't dynamically recruit new specialists mid-task. Strands's Agents-as-Tools pattern is more dynamic — the orchestrator can call any tool-agent on demand based on what the current context requires.

Choose Strands when: you need dynamic specialist delegation, want to run on AWS with tight cloud integration, or need model-agnostic flexibility. Choose CrewAI when: the role metaphor maps cleanly to your use case, you want to move quickly with a larger community, or your team prefers configuration-driven agent definitions over code-first tool decoration. CrewAI's production maturity at scale is lower than LangGraph's — teams often prototype in CrewAI and migrate to LangGraph or Strands when they need checkpointing and durability.

The key differentiating dimension is code-first extensibility (Strands) vs. configuration-first accessibility (CrewAI).

### Strands vs AutoGen

Microsoft Research's AutoGen — a multi-agent conversation framework — officially merged with Semantic Kernel in October 2025 and is now in maintenance mode. The unified Microsoft Agent Framework has a targeted GA date of end of Q1 2026. AutoGen remains a useful reference for historical context and is still widely cited in academic comparisons, but it is not a framework to choose for new production deployments.

Choose Strands over AutoGen for any new project. AutoGen's conversation-driven architecture remains influential as a design pattern, but its active development has stopped.

### Strands vs OpenAI Agents SDK

OpenAI released its Agents SDK in March 2025 as a production-grade replacement for the experimental Swarm framework. The design philosophy is similar to Strands — minimalist, tool-heavy, model-as-orchestrator — but with one critical difference: the OpenAI Agents SDK is built for OpenAI models only. It does not support Bedrock, Anthropic API, Ollama, or any other provider. For teams locked into GPT-4o or GPT-4.1, the OpenAI SDK may feel more native; for everyone else, Strands's model-agnostic design is a decisive advantage.

Choose Strands when: you're not locked into the OpenAI ecosystem, or you want to run the same agent logic against multiple model providers. Choose OpenAI Agents SDK when: your organization is all-in on OpenAI, and you want tight integration with OpenAI's tracing, evals, and fine-tuning infrastructure.

### Strands vs Google ADK

Google released the Agent Development Kit (ADK) in April 2025, one week before Strands. Both are cloud-provider-backed, open-source, model-capable frameworks targeting similar enterprise audiences. The primary differences: ADK uses a hierarchical agent tree (a root agent delegates to sub-agents in a fixed hierarchy), is optimized for Gemini models and Vertex AI, and also supports A2A natively. Strands's dynamic multi-agent patterns are more flexible than ADK's fixed hierarchy, but ADK's deep Vertex AI integration makes it the clear choice for GCP-native teams.

Choose Strands when: you're running on AWS or want cloud-neutral flexibility. Choose Google ADK when: you're running on GCP, using Gemini models, or need Vertex AI's ML tooling alongside your agents. Both frameworks are in active, well-resourced development.

---

## 9. Community and Market Position

**GitHub Stars:** 2,000+ as of July 2025 (preview launched May 2025); growing through early 2026.

**PyPI Downloads:** Over 5 million total as of early 2026; approximately 1.4 million downloads per week and 5.2 million per month as of May 2026, reflecting rapid adoption within the AWS ecosystem.

**Community Contributions:** Of the 150+ pull requests merged between Strands 0.1.0 and 1.0, 22% came from community contributors — a meaningful open-source engagement rate for a framework less than a year old. Contributors include engineers from Accenture, Anthropic, Meta, PwC, Langfuse, mem0.ai, Ragas.io, Tavily, Cohere, Mistral, Writer, Baseten, and others.

**Funding and Company Background:** Strands is backed by Amazon Web Services, one of the largest technology companies in the world. There is no separate funding round — Strands is a first-party AWS open-source project, which means it has essentially unlimited runway and integration with the full AWS product surface. The risk is strategic rather than financial: if AWS decided to deprioritize the project, it could be sunset, though the Apache 2.0 license means the code would remain available.

**Industry Recognition:** Strands was covered at AWS re:Invent 2025 and featured in multiple AWS Summit sessions in 2026. It was cited in the Gartner 2026 AI agent framework landscape alongside LangGraph and CrewAI as a framework achieving significant enterprise adoption. The Strands + AgentCore combination appeared in multiple third-party vendor comparison guides as the "AWS-native path" for enterprise agent deployment.

**Community Sentiment:** Practitioners consistently praise Strands for its simplicity and the quality of its AWS integrations, particularly the zero-configuration IAM and Bedrock connectivity. The most consistent complaints are about the default-to-AWS configuration feeling like lock-in even when it technically isn't, the lack of a visual debugging interface, and the relative immaturity of the community knowledge base compared to LangGraph. Reddit and Discord discussions frequently describe Strands as "the fastest way to get something running on Bedrock" and "the right choice if you're already all-in on AWS" — which is fairly accurate.

**Market Context:** Strands entered the market at the same moment as Google ADK (April-May 2025), with both cloud providers recognizing that the framework space had become a credible platform-extension battleground. The framework landscape in mid-2026 has roughly stratified: LangGraph for complex stateful workflows, CrewAI for accessible multi-agent prototyping, Strands for AWS-ecosystem production deployments, and OpenAI Agents SDK for OpenAI-stack teams. Strands is growing in the fastest-growing segment (AWS enterprise) and is unlikely to plateau soon given its upstream ecosystem advantages.

---

## 10. Pricing

The Strands Agents SDK itself is completely free and open source (Apache 2.0). There is no charge to download, use, or deploy Strands. The costs associated with running Strands agents come from two sources: LLM API usage (charged by your model provider) and, if you choose it, Amazon Bedrock AgentCore for managed hosting and production infrastructure.

### AgentCore Pricing

Amazon Bedrock AgentCore is fully consumption-based with no upfront commitments and no minimum fees. Each module is billed independently, and you can use them à la carte or together.

| **Module** | **Price** | **Unit** | **Notes** |
|---|---|---|---|
| **Runtime / Browser / Code Interpreter** | $0.0895 per vCPU-hour + $0.00945 per GB-hour | Active compute time | Charged only for active CPU/memory use, not idle time |
| **Gateway** | $0.005 per 1,000 tool API calls; $0.025 per 1,000 searches; $0.02 per 100 tools indexed/month | Invocations / searches / tools | Tool proxy and search layer |
| **Memory** | $0.25 per 1,000 short-term events; $0.75 per 1,000 long-term stored; $0.50 per 1,000 retrievals | Memory operations | Custom strategies reduce long-term storage to $0.25/1,000 |
| **Identity** | $0.010 per 1,000 token or API key requests | Auth requests | Free when accessed via Runtime or Gateway |
| **LLM Inference (Bedrock)** | Varies by model | Tokens | Separate from AgentCore; Claude 3.7 Sonnet ~$3/$15 per million input/output tokens |

Note: These prices are sourced from the official AWS pricing page (`aws.amazon.com/bedrock/agentcore/pricing/`) as of early 2026 and should be verified before procurement decisions, as AWS pricing is subject to change.

**Runtime Tier** is designed for teams that need managed execution environments for their Strands agents without managing containers or Kubernetes. The active-compute billing model is favorable for agentic workloads, which typically spend 30-70% of their time in I/O wait (waiting on model responses or external APIs) and would be over-billed by traditional always-on compute pricing.

**Gateway** is for teams exposing agent capabilities as APIs or connecting agents to large numbers of external tools. The per-invocation pricing makes it predictable for low-traffic workloads but can accumulate quickly for agents making thousands of tool calls per hour.

**Memory** pricing is the most variable component. Short-term memory (within a session) is cheap; long-term memory (across sessions) is more expensive and depends heavily on how many memories are written and retrieved. Teams building agents with persistent user context should model this cost carefully before deploying at scale.

### Real-World Cost Scenarios

**Solo developer / side project:** Running a Strands agent locally against the Anthropic API (or Bedrock with minimal traffic) costs roughly $0-$20/month — primarily LLM token costs for testing, with no AgentCore charges. Self-hosted on a small EC2 instance adds $10-30/month of compute.

**Small startup (3-5 people):** A team running a production Strands agent on AgentCore Runtime with moderate traffic (a few hundred sessions/day) and basic Memory usage would pay approximately $100-$400/month: $50-150 in Runtime compute, $20-80 in Memory operations, $20-100 in LLM tokens via Bedrock, and nominal Gateway charges. Total: roughly $100-400/month depending on session length and model.

**Mid-size team in production (20-50 people):** A team with several agents handling thousands of sessions per day, using Memory for user context and Gateway for tool routing, should budget $1,000-$3,000/month for AgentCore services plus $500-$2,000/month for LLM inference. Total: $1,500-$5,000/month. Annual commitment discounts from AWS may reduce this 10-20%.

**Large enterprise (100+ people):** At enterprise scale with dozens of agents, high-volume tool calls, and extensive long-term memory usage, AgentCore costs can reach $5,000-$20,000+/month before LLM inference. Enterprises at this scale would typically negotiate AWS Enterprise Discount Program (EDP) rates, which can provide 20-40% discounts on Bedrock and AgentCore usage. Custom pricing via an AWS account team is standard for this tier.

### Pricing Caveats

These figures are approximate and constructed from the published AWS unit pricing. AWS pricing changes frequently, and LLM inference costs in particular have been declining steadily throughout 2025-2026. Verify all figures at `aws.amazon.com/bedrock/agentcore/pricing/` and with your AWS account team before budget planning.

### Self-Host Option

Running Strands agents entirely outside AgentCore is straightforward and free of platform fees. A self-hosted Strands deployment on EC2, EKS, or Fargate incurs only compute and LLM API costs. You sacrifice the zero-configuration Runtime, managed Memory, built-in Identity, and the operational simplicity of AgentCore — but for teams with existing Kubernetes infrastructure or strong DevOps capability, self-hosting is a fully viable and cost-effective path. The open-source Apache 2.0 license means there are no enterprise feature gates behind a paywall on the SDK itself.

---

## 11. Summary and Verdict

**One-sentence positioning statement:** Strands trades workflow explicitness for radical simplicity, betting that modern LLMs are smart enough to self-orchestrate — and that bet pays off for AWS-native teams but introduces risk for anyone who needs the model's decisions to be auditable or deterministic.

**When to choose Strands:**

- Your team is running on AWS and wants zero-configuration Bedrock, IAM, and S3 integration without writing adapter code
- You need a working agent prototype in under an hour and can iterate from there
- Your use case requires dynamic, open-ended task completion where the model choosing its own path is acceptable
- You need to run the same agent logic against multiple model providers (Claude, GPT-4o, Llama, Gemini) without rewriting orchestration
- You want native A2A support for cross-framework or cross-organization agent interop
- Your team has strong AWS/Python engineering capability and doesn't need a visual or low-code interface

**When not to choose Strands:**

- Your workflow has regulatory compliance requirements where every decision step must be deterministic and auditable — use LangGraph
- Your team is running on GCP or Azure and the AWS-centric defaults will create more friction than value
- You need a mature community with years of tutorials, forum answers, and battle-tested edge-case guidance
- Your users are non-engineers who need to author or inspect agent workflows through a visual interface
- You are building on OpenAI exclusively and want tight integration with OpenAI's eval and tracing infrastructure — use the OpenAI Agents SDK

**Closing context:** Strands occupies a clear and defensible position in the 2026 agent framework landscape: it is the canonical path for AWS-native agent development, with a production backstory that most competing frameworks can't match (Amazon Q Developer and AWS Glue are not small-scale pilots). The framework's model-driven philosophy puts it in the same philosophical camp as OpenAI's Agents SDK and Google's ADK — all three are betting against LangGraph-style explicit graphs and toward letting capable models self-direct. AWS's cloud scale and enterprise relationships give Strands an adoption engine that independent frameworks like CrewAI and Mastra cannot replicate. The primary ceiling is the AWS gravitational pull: Strands will struggle to be the default choice for GCP or Azure shops, regardless of its technical merits. For AWS-committed engineering teams building production agents in 2026, Strands is a serious first option, not a curiosity.

---

## Sources

- [Introducing Strands Agents, an Open Source AI Agents SDK — AWS Open Source Blog](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)
- [Introducing Strands Agents 1.0: Production-Ready Multi-Agent Orchestration Made Simple — AWS Open Source Blog](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-1-0-production-ready-multi-agent-orchestration-made-simple/)
- [Strands Agents and the Model-Driven Approach — AWS Open Source Blog](https://aws.amazon.com/blogs/opensource/strands-agents-and-the-model-driven-approach/)
- [Strands Agents SDK: A Technical Deep Dive into Agent Architectures and Observability — AWS AI Blog](https://aws.amazon.com/blogs/machine-learning/strands-agents-sdk-a-technical-deep-dive-into-agent-architectures-and-observability/)
- [Strands Agents — Official Documentation and Homepage](https://strandsagents.com/)
- [Strands Agents SDK Python — GitHub Repository](https://github.com/strands-agents/sdk-python)
- [Strands Agents — AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html)
- [Amazon Bedrock AgentCore Pricing — AWS](https://aws.amazon.com/bedrock/agentcore/pricing/)
- [Amazon Bedrock AgentCore Overview — AWS](https://aws.amazon.com/bedrock/agentcore/)
- [AWS Unveils Bedrock AgentCore — VentureBeat](https://venturebeat.com/ai/aws-unveils-bedrock-agentcore-a-new-platform-for-building-enterprise-ai-agents-with-open-source-frameworks-and-tools)
- [Evaluating AI Agents for Production: A Practical Guide to Strands Evals — AWS AI Blog](https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-for-production-a-practical-guide-to-strands-evals/)
- [Observing and Evaluating AI Agentic Workflows with Strands and Arize AX — AWS AI Blog](https://aws.amazon.com/blogs/machine-learning/observing-and-evaluating-ai-agentic-workflows-with-strands-agents-sdk-and-arize-ax/)
- [Agentic Application Modernization at Scale with Strands and Amazon Transform — AWS DevOps Blog](https://aws.amazon.com/blogs/devops/use-generative-ai-agents-for-application-modernization-at-scale-with-strands-amazon-transform-custom-and-amazon-bedrock-agentcore/)
- [Using Strands Agents to Create a Multi-Agent Solution with Meta's Llama 4 and Amazon Bedrock — AWS AI Blog](https://aws.amazon.com/blogs/machine-learning/using-strands-agents-to-create-a-multi-agent-solution-with-metas-llama-4-and-amazon-bedrock/)
- [Comparing 4 Agentic Frameworks: LangGraph, CrewAI, AutoGen, and Strands Agents — Medium](https://medium.com/@a.posoldova/comparing-4-agentic-frameworks-langgraph-crewai-autogen-and-strands-agents-b2d482691311)
- [2026 AI Agent Framework Showdown: Claude Agent SDK vs Strands vs LangGraph vs OpenAI Agents SDK — QubitTool](https://qubittool.com/blog/ai-agent-framework-comparison-2026)
- [Google ADK vs AWS Strands: What's Best AI Agent Platform for Enterprise? — TechAhead](https://www.techaheadcorp.com/blog/google-adk-vs-aws-strands-which-ai-agent-platform-wins/)
- [Comparing Agentic AI Frameworks — AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/comparing-agentic-ai-frameworks.html)
- [AWS Intros Strands Agents SDK — TechTarget](https://www.techtarget.com/searchEnterpriseAI/news/366624093/AWS-intros-Strands-Agents-SDK)
- [Amazon Open Sources Strands Agents SDK for Building AI Agents — InfoQ](https://www.infoq.com/news/2025/06/amazon-strands-agents-sdk/)
- [strands-agents PyPI Download Statistics — ClickPy](https://clickpy.clickhouse.com/dashboard/strands-agents)
- [First Impressions with Strands Agents SDK — DEV Community](https://dev.to/aws/first-impressions-with-strands-agents-sdk-4hha)
- [What Is Strands Agents? — Mission Cloud](https://www.missioncloud.com/blog/what-is-strands-agents)
- [AgentCore (Bedrock) Pricing and When Self-Hosting Wins — Scalevise](https://scalevise.com/resources/agentcore-bedrock-pricing-self-hosting/)
