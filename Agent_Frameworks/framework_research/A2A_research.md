# A2A (Agent-to-Agent) Protocol: Deep Research Report

**Date:** May 2026  
**Topic:** Agent-to-Agent (A2A) Protocol — What It Is, How It Works, and Why It Matters for Agent Engineering

---

## Table of Contents

1. [Introduction & Origins](#1-introduction--origins)
2. [What Is A2A?](#2-what-is-a2a)
3. [Core Technical Architecture](#3-core-technical-architecture)
4. [Key Concepts: Agent Cards, Tasks, and Artifacts](#4-key-concepts-agent-cards-tasks-and-artifacts)
5. [Communication Modes](#5-communication-modes)
6. [Security & Authentication](#6-security--authentication)
7. [Relationship to Agent Engineering](#7-relationship-to-agent-engineering)
8. [A2A vs. MCP vs. ACP: Protocol Landscape](#8-a2a-vs-mcp-vs-acp-protocol-landscape)
9. [Framework Integration](#9-framework-integration)
10. [Real-World Use Cases](#10-real-world-use-cases)
11. [Ecosystem & Adoption](#11-ecosystem--adoption)
12. [Governance & Open Source](#12-governance--open-source)
13. [Future Trajectory](#13-future-trajectory)
14. [Summary](#14-summary)
15. [Sources](#15-sources)

---

## 1. Introduction & Origins

The A2A (Agent-to-Agent) protocol, formally named **Agent2Agent**, was announced by Google in **April 2025** alongside their Agent Development Kit (ADK). It was created to solve a fundamental problem in the emerging multi-agent AI world: agents from different vendors, built on different frameworks, could not talk to each other in any standardized way.

At launch, Google brought over **50 enterprise partners** on board — including Salesforce, Accenture, SAP, and Deloitte — signaling immediate industry alignment. By **June 2025**, Google donated the protocol to the **Linux Foundation**, formally making it a vendor-neutral open standard. By **April 2026**, just one year after launch, over **150 organizations** supported A2A, and it had achieved production use at major enterprises worldwide.

The timing of A2A's emergence is important context: it arrived during the "agentic AI explosion" of 2025, when nearly every major AI vendor was releasing agent frameworks (OpenAI's Agents SDK in March 2025, Google's ADK in April 2025, Anthropic's Agent SDK alongside Claude 4 models). Each framework produced capable agents in isolation — but agents from different ecosystems could not collaborate. A2A was the answer to that fragmentation.

---

## 2. What Is A2A?

**A2A is an open protocol that enables AI agents built by different vendors, on different frameworks, to discover each other, communicate, delegate tasks, and exchange results — all in a standardized, secure way.**

In simple terms: if MCP (Model Context Protocol) is how an agent talks to *tools*, A2A is how an agent talks to *other agents*. These two protocols are complementary, not competing.

A2A addresses three foundational problems:

1. **Discovery** — How does one agent know what another agent can do?
2. **Task exchange** — How do agents hand off structured work to each other?
3. **Transport** — How do agents communicate reliably over a network?

A2A's design philosophy is deliberately pragmatic. Rather than inventing new infrastructure, it builds on standards that enterprise IT already understands: **HTTP, JSON-RPC 2.0, and Server-Sent Events (SSE)**. This dramatically reduces adoption friction.

---

## 3. Core Technical Architecture

A2A defines a layered architecture:

**Layer 1 — Identity & Discovery:** Agent Cards (JSON metadata files) advertise an agent's identity, capabilities, endpoint URL, and authentication requirements. Any client can fetch an agent's card to understand what it can do before ever sending a request.

**Layer 2 — Abstract Behaviors:** Defines the fundamental operations and lifecycle states that all A2A agents must support, independent of transport mechanism.

**Layer 3 — Protocol Bindings:** Concrete mappings of abstract operations to specific wire formats. Currently supports:
- **JSON-RPC 2.0 over HTTP(S)** — the primary and most widely implemented binding
- **gRPC** — added in v0.3 for high-performance, low-latency scenarios
- **Server-Sent Events (SSE)** — for streaming real-time updates
- **Webhooks** — for push notifications on long-running or disconnected tasks

The client/server model in A2A defines two roles:

- **Client Agent** — the agent that initiates requests and coordinates tasks on behalf of the user or an orchestrator
- **Remote/Service Agent** — the agent that advertises specific capabilities and handles incoming requests

These roles are not fixed — any agent can act as either client or server depending on context, enabling true peer-to-peer multi-agent collaboration.

---

## 4. Key Concepts: Agent Cards, Tasks, and Artifacts

### Agent Cards

An Agent Card is a JSON metadata document published by every A2A server. It functions as the agent's "business card" or résumé. It contains:

- **Identity** — name, description, version
- **Service endpoint URL** — where to send requests
- **Authentication requirements** — what credentials are needed and how to obtain them
- **Skills** — a list of specific capabilities the agent advertises, with descriptions and example inputs
- **Supported modalities** — whether the agent can handle text, images, files, structured data, etc.

Agent Cards enable agent *discovery* — a key enabler for dynamic, self-organizing multi-agent systems where agents find collaborators at runtime rather than via hardcoded integrations.

### Tasks

A **Task** is the fundamental unit of work in A2A. Tasks are:

- **Stateful** — they progress through a defined lifecycle: `submitted → working → completed / failed / canceled`
- **Identified** — each task has a unique ID for tracking and referencing
- **Asynchronous-friendly** — long-running tasks (hours or days) are first-class citizens, not afterthoughts

Tasks contain a message payload, which is composed of *Parts* — typed content units that can be text, files, binary data, or structured JSON. This flexible part-based model means a single task message can carry mixed content (e.g., a text instruction plus a file attachment).

### Artifacts

An **Artifact** is the output that a remote agent generates in response to a task. Like task messages, artifacts are composed of Parts and can represent:

- Documents and text responses
- Generated images or files
- Structured data (JSON objects)
- Code or executable outputs

Artifacts are the formalized "return value" of an A2A task, designed to be easily consumed by downstream agents or users.

---

## 5. Communication Modes

A2A supports three distinct communication patterns to handle different scenarios:

**Synchronous Request/Response** — The client sends a task and waits for the completed result. Best for short, fast tasks where latency is acceptable.

**Streaming via SSE (Server-Sent Events)** — The client opens an HTTP connection and receives real-time incremental updates as the agent works. Best for long-running tasks where the user or orchestrator wants live progress. The agent can stream partial results, status changes, and intermediate artifacts before the final result is ready.

**Asynchronous Push Notifications via Webhooks** — For very long-running tasks or disconnected scenarios (tasks that run for hours or days), the client provides a webhook URL. The server calls that webhook when significant state changes occur (e.g., task completed, error occurred). This model enables true "fire and forget" task delegation at enterprise scale.

---

## 6. Security & Authentication

A2A was designed with enterprise security requirements as a first-class concern, not an afterthought. Key security features:

**Authentication Schemes:** A2A supports all standard enterprise auth mechanisms: OAuth 2.0, OpenID Connect (OIDC), API keys, and mutual TLS (mTLS). Authentication requirements are declared upfront in the Agent Card so clients know exactly what credentials they need before making their first request. Credentials are obtained *out-of-band*, meaning A2A itself doesn't handle credential exchange — it works with existing enterprise identity infrastructure.

**Transport Security:** All communication happens over HTTPS. JSON-RPC payloads are encrypted in transit by default.

**Push Notification Security:** When using webhooks, A2A supports request signing using JWT with ECDSA or RSA key pairs. The agent signs webhook payloads with its private key; the receiving client verifies with the public key. This ensures that webhook calls genuinely originate from the claimed agent and aren't spoofed.

**Agent Card Signing:** Version 0.3 introduced the ability to cryptographically sign Agent Cards, allowing clients to verify that an agent's advertised capabilities haven't been tampered with.

---

## 7. Relationship to Agent Engineering

A2A is directly foundational to the discipline of **Agent Engineering** — the practice of designing, building, operating, and orchestrating systems of autonomous AI agents. Here's how A2A connects to the core concerns of agent engineers:

### Interoperability by Default

Before A2A, building a multi-agent system meant either using a single framework (vendor lock-in) or writing custom integration code between frameworks. A2A eliminates this by providing a universal handshake. An agent engineer can now compose systems from the best available agents — a LangGraph agent for stateful workflows, a CrewAI agent for role-based collaboration, a custom Google ADK agent for Gemini-native tasks — and wire them together via A2A without framework-specific adapters.

### Enabling Multi-Agent Architectures

A2A enables agent engineers to implement the standard multi-agent design patterns:

- **Orchestrator-Worker** — An orchestrator agent breaks down a complex task and delegates sub-tasks to specialist agents via A2A
- **Pipeline/Chaining** — Agent A completes work and passes artifacts to Agent B, which continues the pipeline
- **Peer-to-Peer Negotiation** — Agents discover each other via Agent Cards and negotiate task delegation dynamically
- **Hierarchical Delegation** — Deep chains where Agent A → Agent B → Agent C, each level A2A-connected

### Long-Running Task Management

One of the hardest problems in agent engineering is managing tasks that take minutes, hours, or days, with humans potentially in the loop at various points. A2A's streaming and push-notification models provide native primitives for this, reducing the complexity agent engineers must handle themselves.

### Observable, Auditable Workflows

A2A's stateful task model — with explicit task IDs and lifecycle states — makes multi-agent workflows observable and auditable. This is critical for enterprise deployment, where compliance, debugging, and rollback capabilities are non-negotiable.

### Decoupled Deployment

Because A2A operates over HTTP, agents can be deployed anywhere — different cloud regions, different organizations, different vendors — and still collaborate. This decoupling is essential for building resilient, scalable agent systems.

---

## 8. A2A vs. MCP vs. ACP: Protocol Landscape

As of 2026, three protocols dominate the agent interoperability conversation:

### MCP (Model Context Protocol) — Anthropic

- **Purpose:** Agent-to-tool communication. Standardizes how an AI model/agent connects to external tools, APIs, data sources, and resources.
- **Analogy:** USB-C for tools — one universal connector.
- **Use it when:** Your agent needs to access databases, call APIs, read files, execute code, or use any external capability.
- **Status:** Most mature protocol; 97 million+ monthly SDK downloads as of early 2026.

### A2A (Agent-to-Agent Protocol) — Google / Linux Foundation

- **Purpose:** Agent-to-agent communication. Standardizes how agents discover each other, delegate tasks, and collaborate across vendors.
- **Analogy:** The communication layer between workers on a team.
- **Use it when:** You have multiple agents that need to coordinate, delegate sub-tasks, or work as peers across frameworks or organizations.
- **Status:** 150+ organizations, production use at major enterprises, v1.0 reached early 2026.

### ACP (Agent Communication Protocol) — IBM/AGNTCY

- **Purpose:** REST-native performative messaging for local multi-agent systems, with a focus on observability and agent registries.
- **Architecture:** Client-server (vs. A2A's peer-to-peer)
- **Use it when:** You need structured, observable intra-system communication with registry-based discovery, particularly in IBM or AGNTCY ecosystems.
- **Status:** Less widely adopted than A2A but gaining traction in specific enterprise contexts.

### How They Work Together

A well-architected enterprise multi-agent system in 2026 typically uses all three:

- **MCP** for all tool access (databases, APIs, file systems, third-party services)
- **A2A** for agent-to-agent coordination and cross-vendor/cross-organization delegation
- **ACP** (optionally) for structured observability and registry-based discovery in IBM/enterprise contexts

These protocols address different layers of the stack and are genuinely complementary, not competing.

---

## 9. Framework Integration

A2A has achieved broad integration across the major agent engineering frameworks:

**Google ADK (Agent Development Kit)** — Native A2A support is built into ADK. ADK agents can expose themselves as A2A servers and act as A2A clients. ADK reached 1.0 GA in early 2026 with support for Python, Go, Java, and TypeScript. The ADK → A2A integration is the most polished and first-party.

**LangGraph** — One of the two most popular open-source agent frameworks (alongside CrewAI), with graph-based state management. Supports A2A, allowing LangGraph agents to be invoked by and invoke agents in other frameworks. LangGraph surpassed CrewAI in GitHub stars in early 2026 due to enterprise adoption.

**CrewAI** — Role-based multi-agent framework (45,900+ GitHub stars as of early 2026) with adoption at roughly 60% of the Fortune 500. Added native MCP and A2A support. CrewAI's crew-based abstraction maps well onto A2A's client/server roles.

**Amazon Bedrock AgentCore** — AWS integrated A2A natively into Bedrock's agent infrastructure, enabling Bedrock-hosted agents to participate in A2A networks.

**Azure AI Foundry** — Microsoft integrated A2A natively, making Azure-deployed agents interoperable with the broader A2A ecosystem.

**Google Cloud Vertex AI** — Supports A2A natively for agents deployed on Google's cloud infrastructure.

**SDK Availability:** The A2A SDK ecosystem grew from a single Python reference implementation at launch to production-ready libraries in Python, JavaScript, Java, Go, and .NET by 2026.

---

## 10. Real-World Use Cases

### Supply Chain Coordination

Perhaps the clearest illustration of A2A's value: a supply chain where separate specialist agents handle forecasting, inventory management, logistics, and supplier communication. Without a common protocol, these agents (potentially from different vendors) cannot communicate. With A2A, they form a collaborative network — the inventory agent detects low stock, delegates an order request to a procurement agent, which uses A2A to communicate with external supplier agents that fulfill the order.

### IT Operations

ServiceNow partnered with Google Cloud specifically to establish A2A as the industry standard for IT operations agent interoperability. In this context, agents that handle ticket triaging, system diagnostics, alert correlation, and runbook execution collaborate via A2A without requiring a monolithic system.

### HR and Finance Automation

Workday adopted A2A to enable HR and finance agent workflows, where agents for payroll processing, benefits administration, compliance checking, and reporting can delegate to each other dynamically.

### Financial Services

Banks and insurers are using A2A to coordinate agents across fraud detection, risk assessment, customer service, and compliance — each a specialist agent that can be developed and upgraded independently while collaborating via the shared protocol.

### Cross-Organization Agent Networks

A2A's most powerful use case is enabling agents from entirely different organizations to collaborate securely. For example, a buyer's procurement agent can directly negotiate and coordinate with a supplier's order management agent — two different companies, two different tech stacks, connected via A2A Agent Cards and authenticated requests.

### Research and Analysis Workflows

Long-running research tasks where an orchestrator agent delegates to specialized agents (web search agent, document analysis agent, synthesis agent, fact-checking agent) and coordinates their outputs into a final report — all asynchronously, with streaming updates to the user.

---

## 11. Ecosystem & Adoption

### Organizational Support (as of April 2026)

**Cloud Platforms (native integration):** Google Cloud, AWS (Bedrock AgentCore), Microsoft Azure (AI Foundry)

**Enterprise Software:** Salesforce, SAP, ServiceNow, Workday, IBM

**Consulting/Services:** Accenture, Deloitte

**Total organizations supporting A2A:** 150+, under the Linux Foundation umbrella

### Market Context

The explosive adoption of A2A reflects the broader agentic AI wave:

- Gartner projects 40% of enterprise applications will feature task-specific AI agents by 2026 (up from <5% in 2025)
- IDC forecasts agentic AI spending to exceed $1.3 trillion by 2029 (31.9% CAGR)
- A2A emerged as the coordination layer that makes large-scale agent deployment viable

### Developer Ecosystem

- Production-ready SDKs in 5 languages (Python, JavaScript, Java, Go, .NET)
- Growing number of tutorials, courses (including DeepLearning.AI's A2A course), and reference implementations
- Active open-source community under the Linux Foundation's governance

---

## 12. Governance & Open Source

**License:** Apache 2.0

**Governing Body:** Linux Foundation (as of June 2025)

**Project Home:** [a2a-protocol.org](https://a2a-protocol.org) / [github.com/a2aproject/A2A](https://github.com/a2aproject/A2A)

A2A is part of a broader Linux Foundation umbrella that, as of 2026, also governs MCP — creating a unified **Agentic AI Foundation (AAIF)** with 146+ member organizations including Anthropic, Google, OpenAI, Microsoft, and AWS. This convergence under neutral governance is significant: it signals that the major AI players have aligned on these two complementary protocols (MCP + A2A) as the foundational standards for the agentic AI era.

### Version History

- **v0.1** (April 2025) — Initial release, 50+ enterprise partners at launch
- **v0.2** (mid-2025) — Stateless interaction support, official Python SDK
- **v0.2.5** — Incremental improvements, broader SDK support
- **v0.3** (late 2025) — gRPC support, Agent Card signing, extended Python SDK client support
- **v1.0** (early 2026) — Production-ready GA release; SDK in 5 languages; native integration in AWS, Azure, and Google Cloud

---

## 13. Future Trajectory

Several trends signal where A2A is headed:

**Standardization Maturity** — With v1.0 released and 150+ organizations committed, A2A is moving from "emerging standard" to "assumed infrastructure" in enterprise agent engineering. Much like REST or OAuth, it is becoming a default assumption in how multi-agent systems are built.

**Protocol Convergence** — The MCP and A2A communities are increasingly coordinating under the Linux Foundation. A unified framework that seamlessly bridges tool access (MCP) and agent coordination (A2A) is the logical next step.

**Agent Marketplaces** — Agent Cards naturally enable agent discovery marketplaces where organizations can publish, find, and subscribe to specialized agents — a "App Store" model for enterprise AI capabilities.

**Cross-Organizational Agent Networks** — As enterprise security patterns mature around A2A, we'll see more inter-company agent collaboration where supplier, buyer, and logistics agents from entirely different organizations form ad-hoc task networks.

**Regulatory and Compliance Integration** — A2A's observable, auditable task model positions it well for emerging AI governance requirements. Expect compliance frameworks to reference A2A-style logging and task traceability standards.

---

## 14. Summary

A2A (Agent2Agent) is Google-originated, Linux-Foundation-governed open protocol that solves one of the central problems in modern agent engineering: how do AI agents from different vendors, frameworks, and organizations collaborate at scale? By standardizing agent discovery (Agent Cards), work exchange (Tasks and Artifacts), and transport (HTTP, JSON-RPC, SSE), A2A provides the coordination layer that makes multi-agent systems practical beyond toy demos.

For agent engineers, A2A is foundational infrastructure — the equivalent of HTTP for the web, but for agent networks. It enables framework-agnostic composition, long-running task management, observable workflows, and enterprise-grade security. Paired with MCP (for tool access), it forms the two-protocol foundation on which serious agentic AI systems are built in 2026.

In just one year, A2A went from an announcement to production use at dozens of Fortune 500 companies, with native integration in every major cloud platform. It is not a niche specification — it is rapidly becoming the assumed communication standard for the agentic AI era.

---

## 15. Sources

- [Announcing the Agent2Agent Protocol (A2A) — Google Developers Blog](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [A2A Protocol Official Site](https://a2a-protocol.org/latest/)
- [What is A2A? — A2A Protocol Docs](https://a2a-protocol.org/latest/topics/what-is-a2a/)
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [A2A Core Concepts — Key Concepts](https://a2a-protocol.org/latest/topics/key-concepts/)
- [GitHub — a2aproject/A2A](https://github.com/a2aproject/A2A)
- [A2A Protocol Surpasses 150 Organizations — Linux Foundation Press Release](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year)
- [Agent2Agent Protocol is Getting an Upgrade — Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)
- [What Is Agent2Agent (A2A) Protocol? — IBM](https://www.ibm.com/think/topics/agent2agent-protocol)
- [Google A2A Protocol: How Agent-to-Agent Coordination Works — Atlan](https://atlan.com/know/google-a2a-protocol/)
- [Google's Agent2Agent Protocol Explained — Galileo AI](https://galileo.ai/blog/google-agent2agent-a2a-protocol-guide)
- [MCP vs A2A: A Guide to AI Agent Communication Protocols — Auth0](https://auth0.com/blog/mcp-vs-a2a/)
- [A2A vs MCP — Descope](https://www.descope.com/blog/post/mcp-vs-a2a)
- [ACP vs MCP vs A2A: Agent Protocol Comparison (2026) — Morph](https://www.morphllm.com/comparisons/acp-vs-mcp-vs-a2a)
- [ADK with Agent2Agent (A2A) Protocol — Google ADK Docs](https://google.github.io/adk-docs/a2a/)
- [Google ADK 1.0 and A2A Protocol: Defining the 2026 Multi-Agent Standard — n1n.ai](https://explore.n1n.ai/google-adk-1-0-a2a-protocol-multi-agent-standard-2026-05-04)
- [Agent2Agent Protocol: The Standard for AI Agent Interoperability — Salesforce](https://www.salesforce.com/agentforce/ai-agents/agent2agent-protocol/)
- [Open Protocols for Agent Interoperability Part 4: A2A — AWS Open Source Blog](https://aws.amazon.com/blogs/opensource/open-protocols-for-agent-interoperability-part-4-inter-agent-communication-on-a2a/)
- [How to Build a Multi-Agent AI System with LangGraph, MCP, and A2A — freeCodeCamp](https://www.freecodecamp.org/news/how-to-build-a-multi-agent-ai-system-with-langgraph-mcp-and-a2a-full-book/)
- [Scale Agents with CrewAI, LangGraph, A2A, and ADK — Google Codelabs](https://codelabs.developers.google.com/next26/scale-agents)
- [A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP — arXiv](https://arxiv.org/html/2505.02279v1)
- [Linux Foundation Launches Agent2Agent Protocol Project](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)
- [A2A: The Agent2Agent Protocol Course — DeepLearning.AI](https://www.deeplearning.ai/courses/a2a-the-agent2agent-protocol)
