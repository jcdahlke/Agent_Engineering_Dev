# Agent Framwork Report

## Potential Framework Points

- Year Released
- Price (beginner, startup, enterprise) and features of that price
- Summary of how it works (brief, include analogy and a few key components)
- Who uses it (a couple of the most notable users and what they use it for)
- Industry Adoption (How widespread is its adoption in industry, and what usecase does it stand out in)
- Number of users who use this framework (number of github stars and the number of downloads monthly)
- Why choose this framework over another (What makes this framework special that other frameowrks do not have)
- Coding Language
- Has GUI (does it have an interface for people to use, whether to work with or monitor the process)
- Includes HITL integration?
- Still adding new features?

## Frameworks

---

### AutoGen / AG2

**Year Released:** August 2023 (by Microsoft Research); community fork AG2 established November 2024

**Coding Language:** Python (also .NET / C# via a separate SDK)

**Has GUI:** Yes — AutoGen Studio is a no-code visual interface for building and prototyping multi-agent conversations, installable via `pip install autogenstudio`

**HITL Integration:** Yes — the UserProxyAgent pattern allows a human to provide input at any turn, and AG2's human-in-the-loop support lets humans review and approve agent actions before execution continues

**Still Adding New Features:** Split situation — the Microsoft-maintained `microsoft/autogen` is in maintenance mode (security patches only, no new features) as of October 2025. The community fork `ag2ai/ag2` is actively developed with new features shipping regularly.

**Summary:** AutoGen treats conversation as the coordination primitive for multi-agent systems. Instead of defining an explicit workflow graph, you let agents negotiate the solution through dialogue — one agent proposes code, another executes it, a critic reviews the result, a manager decides what's next. Think of it as a round-table meeting where AI specialists debate and iterate until a task is done. Key components: AssistantAgent (the reasoning/generation role), UserProxyAgent (executes code or relays human input), GroupChat (manages multi-agent conversations), and GroupChatManager (routes speaking turns).

**Notable Users:**
- **Microsoft Research** — internal multi-agent research and prototyping across code generation, data analysis, and scientific reasoning
- **KPMG** — production agent deployments on Azure AI Foundry for audit automation and enterprise knowledge management
- **BMW** — Azure AI Foundry Agent Service for manufacturing operations automation
- **Alice Labs** — code generation and data analysis workflows for clients in financial services, media, and public sector
- **Academic institutions globally** — most widely cited agent framework paper in the field (~1,300 citations), used extensively in university AI research labs

**Industry Adoption:** AutoGen's widest adoption is in research and code-generation contexts. It is the most academically influential multi-agent framework ever published and is used extensively in enterprise deployments through Microsoft's Azure AI Foundry channel. Primary industries: software development/code generation, financial services (quantitative analysis), healthcare/biotech research, government document processing, and academic research computing. Less common in production workloads requiring deterministic, auditable workflows.

**Community Size:** AG2 repo: 50,000+ GitHub stars; Microsoft AutoGen repo: ~40,000+ stars (maintenance mode); Discord community: 20,000+ members; original paper: ~1,300 academic citations

**Why Choose This Framework:** AutoGen is the best framework for iterative code generation workflows where the write-run-debug loop is the core value — no other framework handles code execution and self-correction as naturally. It excels when the problem structure is genuinely unknown in advance and agents need to explore and adapt rather than follow a predetermined workflow. .NET/C# support is unique among agent frameworks. The AG2 fork is the right choice for teams that value community governance over commercial backing and want the original AutoGen design philosophy under active development.

**Pricing:**
- **Beginner:** $0 — both `microsoft/autogen` and `ag2ai/ag2` are MIT-licensed and completely free. LLM API costs (pay-per-token with your chosen provider) are the only expense.
- **Startup:** $0 framework cost. Self-managed AG2 on cloud infrastructure. LLM API costs at production volume are the main expense — approximately $500–$2,000/month for a 4-agent GroupChat running 50,000 conversations/month, depending on model choice.
- **Enterprise:** Azure AI Foundry Agent Service for managed, governed deployment (consumption-based Azure pricing). Full enterprise contracts run $100,000–$500,000+/year at scale, primarily driven by LLM consumption and Azure infrastructure, not framework fees.

---

### CrewAI

**Year Released:** 2023, founded by João Moura

**Coding Language:** Python

**Has GUI:** Yes — the CrewAI Platform includes a web-based Studio for building, deploying, and monitoring crews without writing code

**HITL Integration:** Yes — a `human_input=True` parameter on any Task pauses execution and prompts a human for input before the agent continues

**Still Adding New Features:** Yes — actively developed with frequent releases; Flows (deterministic workflow orchestration), CrewAI Studio, and enterprise platform features all added through 2025–2026

**Summary:** CrewAI uses an organizational team metaphor — you define a "crew" of agents, each with a distinct role, goal, and backstory, assign tasks, and let them collaborate. Think of it as assembling a specialist team: a Researcher, an Analyst, and a Writer working together on a report. The framework handles task delegation, inter-agent communication, and execution flow automatically. Key components: Agent (role + goal + backstory + tools), Task (what needs to be done), Crew (the team), Process (sequential, hierarchical, or parallel execution), and Flows (deterministic workflow control for production).

**Notable Users:**
- **PwC** — agent-driven code development assistance, improving accuracy from ~10% to 70%+
- **Fortune 500 CPG company** — back-office operations automation with 75% reduction in processing time
- **Marketing agencies** — content generation automation with 50% volume increase and 20% cost reduction
- **Federal government agencies** — mission-critical workflow automation via Flows
- **Financial services firms** — automated research, analysis, and report generation

**Industry Adoption:** Used by nearly half of the Fortune 500 in some capacity, with 150+ named enterprise customers. Highest adoption in IT operations (52%), followed by marketing/content, financial services, software development, and customer support. Stands out for rapid multi-agent prototyping and workflows that map naturally to human team structures.

**Community Size:** 47,800+ GitHub stars; 27M+ monthly PyPI downloads; 2 billion agent runs in the past 12 months; 10M+ open-source agents per month; 150+ countries

**Why Choose This Framework:** CrewAI delivers the fastest time-to-working-demo of any Python agent framework — a developer can define a crew with readable roles and get something running in under an hour. The role-based metaphor is uniquely accessible to non-engineers, making it practical for stakeholders to read and configure agent behavior. It is the dominant framework for business workflow automation, content pipelines, and role-delegation tasks. Its 2B+ execution count demonstrates genuine production-scale adoption, not just developer experimentation.

**Pricing:**
- **Beginner:** $0 — the open-source library (`pip install crewai`) is MIT-licensed with no usage limits or fees. The free platform tier allows 50 executions/month and 1 deployed crew.
- **Startup:** ~$25/month (Professional) for ~100 executions/month and expanded deployments. LLM API costs dominate actual spend.
- **Enterprise:** Custom pricing starting ~$6,000+/year, including 10,000+ executions/month, unlimited seats, SSO, SOC 2 compliance, PII masking, SLA guarantees, and on-premise deployment. Ultra tier at ~$120,000/year for maximum scale and support.

---

### Haystack

**Year Released:** 2020 (original Haystack); Haystack 2.0 architectural rewrite released early 2024

**Coding Language:** Python

**Has GUI:** Yes — deepset Studio is a visual pipeline editor included in the Haystack Enterprise Platform, allowing teams to design, inspect, and manage pipelines without writing code

**HITL Integration:** Yes — pipelines support human review components as explicit pipeline nodes, enabling workflows where generated outputs are reviewed and approved by humans before passing to the next stage

**Still Adding New Features:** Yes — Haystack 2.0 is under active development with releases every few weeks; recent additions include expanded agent support, multimodal components, and MCP integration

**Summary:** Haystack treats AI application architecture as an engineering discipline — every step is a named component with typed inputs and outputs, wired together into an explicit directed graph pipeline. Think of it like a factory assembly line where you can see and inspect every station. Unlike frameworks that hide retrieval inside an agent loop, Haystack puts the retrieval pipeline center stage: documents flow through loaders, indexers, retrievers, rerankers, and generators in a transparent, debuggable sequence. Key components: Pipeline (the directed graph), Component (any typed processing unit), DocumentStore (the retrieval backend), Retriever, Reranker, and Generator.

**Notable Users:**
- **Airbus** — QA system for cockpit manuals that retrieves precise answers from 1,000+ page technical documents in under one second
- **Airbus Defence and Space** — automated compliance checking against military regulations
- **The Economist** — content discovery and editorial AI applications
- **Oxford University Press** — semantic navigation over academic publishing catalogs
- **Siemens, LEGO, Comcast** — large enterprise knowledge management and internal search
- **Lufthansa Industry Solutions** — compliance-aware AI knowledge assistant for regulated aviation operations
- **Manz** — legal research AI transformation for navigating complex legal corpora

**Industry Adoption:** Strongest adoption in aerospace/defense (technical document QA), media/publishing (content discovery), legal/compliance (high-precision document retrieval), financial services (policy and regulatory document processing), and manufacturing (operational knowledge management). Named a 2024 Gartner Cool Vendor in AI Engineering. Particularly strong with European enterprises due to its cloud-neutral, GDPR-compatible architecture.

**Community Size:** 24,000+ GitHub stars; 2,300+ forks; 100+ community-contributed integrations; total funding: $45.2M (Series B led by Balderton Capital with GV)

**Why Choose This Framework:** Haystack is the deepest retrieval and document intelligence framework in the category — when the hard engineering problem is retrieving accurate, relevant information from large, complex document corpora, no other framework matches its native hybrid retrieval, reranking, table-aware extraction, and pipeline evaluation infrastructure. Its explicit pipeline model makes retrieval behavior debuggable and auditable at every stage, which is a hard requirement in regulated industries. Its cloud-neutral, EU-native architecture is often a compliance requirement for European teams.

**Pricing:**
- **Beginner:** $0 — the framework (`pip install haystack-ai`) is Apache 2.0 licensed and fully free. Infrastructure costs (vector store, LLM API) are the only expense.
- **Startup:** Enterprise Starter (contact sales) — provides up to 4 hours/month of direct consultation with deepset engineers, email support, and extended version maintenance. Infrastructure + LLM API: approximately $300–$1,000/month at small startup scale.
- **Enterprise:** Haystack Enterprise Platform (custom, contact sales) — adds deepset Studio visual editor, managed cloud/hybrid/on-premise hosting, dedicated SLA-backed support, and Expert Services. Estimated $100,000–$500,000+/year at large enterprise scale based on comparable platforms. Available on AWS Marketplace.

---

### LangGraph

**Year Released:** 2023 (as part of the LangChain ecosystem); v1.0 stable release October 2025

**Coding Language:** Python (primary); TypeScript also maintained

**Has GUI:** Yes — LangGraph Studio provides a visual graph editor for building and debugging workflows locally; LangSmith provides a production trace viewer with time-travel debugging and graph visualization

**HITL Integration:** Yes — first-class primitive. `interrupt_before` and `interrupt_after` parameters on any node pause graph execution at that point, allowing a human to review state, provide input, or approve actions before execution resumes

**Still Adding New Features:** Yes — v1.0 released October 2025; actively developed with frequent releases, including expanded multi-agent patterns, improved persistence backends, and LangSmith platform integrations

**Summary:** LangGraph models agent workflows as explicit state machines — nodes in a directed graph share a typed state schema, and edges define the routing logic between them. Think of it as a flowchart that can loop, branch, pause, and resume, with the entire execution history saved to a checkpoint store. This makes complex workflows predictable, debuggable, and recoverable. Key components: StateGraph (the workflow), Node (any function that reads/writes state), Edge (routing logic, including conditional edges), Checkpointer (durable persistence), and LangSmith (observability, time-travel debugging, and deployment).

**Notable Users:**
- **LinkedIn** — AI recruiter automating candidate sourcing, matching, and outreach
- **Uber** — large-scale code migration with specialized agent networks
- **Replit** — AI coding copilot with human-in-the-loop for software generation
- **Elastic** — real-time threat detection orchestration
- **AppFolio** — property management copilot (10+ hours saved per manager per week, 2x decision accuracy)
- **BlackRock, JPMorgan, Klarna** — production AI agent workflows in financial services
- **~400 companies** on LangGraph Platform (LangSmith Deployment) as of early 2026

**Industry Adoption:** The de facto production standard for stateful agent workflows. Widest adoption in financial services (JPMorgan, BlackRock, Klarna), technology/software development (Uber, Replit, Elastic), cybersecurity, real estate, and enterprise SaaS. Consistently cited in 2026 surveys as the framework teams "graduate to" when moving beyond prototypes to production. Forecasted market of $47B by 2030 (44.8% CAGR).

**Community Size:** 24,600+ GitHub stars; 34.5 million monthly PyPI downloads; surpassed CrewAI in GitHub stars in early 2026; ~400 companies on managed LangGraph Platform

**Why Choose This Framework:** LangGraph is the most battle-tested production agent framework available. Its explicit state machine model makes complex workflows predictable and auditable. Durable checkpoint-based persistence means long-running workflows survive process failures. LangSmith's time-travel debugging reduces production incident resolution time dramatically. Human-in-the-loop is a first-class primitive rather than a workaround. It is the right framework when workflow correctness, durability, and observability are non-negotiable.

**Pricing:**
- **Beginner:** $0 — LangGraph (the library) is MIT-licensed with no fees. LangSmith Developer tier is free: 5,000 traces/month, 14-day retention, 1 seat.
- **Startup:** LangSmith Plus at $39/seat/month — 10,000 traces/month included, $0.50 per 1,000 additional traces, 1 free dev deployment. LLM API costs vary by provider.
- **Enterprise:** LangSmith Enterprise (custom pricing) — unlimited seats, custom trace volume, up to 400-day retention, dedicated support and SLA, custom deployment options. Annual contracts required; pricing scales with trace volume and seat count.

---

### LlamaIndex

**Year Released:** November 2022 (as GPT Index); rebranded LlamaIndex and incorporated April 2023

**Coding Language:** Python (primary); TypeScript/JavaScript SDK maintained with full feature parity on core capabilities

**Has GUI:** Yes — LlamaCloud provides a web interface for managing document indexes, monitoring parsing pipelines, and configuring retrieval configurations without writing code

**HITL Integration:** Limited — basic support; human-in-the-loop is not a first-class design primitive. Teams typically implement HITL by wrapping LlamaIndex query engines inside a broader orchestrator (LangGraph, CrewAI) that provides the pause/resume mechanism.

**Still Adding New Features:** Yes — actively developed; LlamaParse V2, LlamaCloud managed pipelines, agentic document workflows, and multi-modal parsing capabilities all shipped in 2025–2026

**Summary:** LlamaIndex treats the data pipeline as the first citizen of AI application architecture. Raw documents flow through loaders and parsers, get chunked into nodes, are organized into indexes, and become retrievable context for LLMs. Think of it as a sophisticated library cataloging system — before an agent can answer questions, LlamaIndex has already ingested, parsed, organized, and indexed everything it might need. Agents and workflows sit on top of this retrieval foundation. Key components: LlamaParse (document parsing), Node (chunked document unit), VectorStoreIndex (semantic search index), QueryEngine (retrieval + generation), and LlamaCloud (managed indexing pipeline).

**Notable Users:**
- **Experian** — AI customer support agents; reduced time-to-first-token from 8s to 1s with optimized retrieval
- **Carlyle Group** — LlamaParse in investment analytics pipeline for complex financial document layouts
- **Salesforce** — LlamaParse for Agentforce document preprocessing, previously required multiple engineers
- **KPMG** — enterprise AI applications for financial and audit document retrieval (also a strategic investor)
- **NTT DATA** — enterprise document parsing and RAG applications for clients across industries
- **Cemex** — small data science team shipped 10 production AI use cases in a few months

**Industry Adoption:** 40% of Fortune 500 companies and 5,000+ startups in the user base; 1 billion+ production queries processed. Strongest in financial services/private equity (complex document parsing), enterprise IT (data pipeline automation), customer service (RAG-powered support agents), legal technology (contract and regulatory document intelligence), and manufacturing (operational document access). Default recommendation in AWS Prescriptive Guidance for document-heavy RAG use cases.

**Community Size:** ~40,000 GitHub stars; 3M+ monthly PyPI downloads; 300+ LlamaHub integration packages; 230,000 LinkedIn followers; total funding: $27.5M (Greylock + Norwest, with Databricks and KPMG as strategic investors)

**Why Choose This Framework:** LlamaIndex provides the deepest document parsing and retrieval infrastructure in the category. LlamaParse handles complex enterprise documents — nested tables, multi-column layouts, embedded charts, mixed modalities — at a quality level that open-source alternatives cannot match. The breadth of LlamaHub data connectors (300+) means almost any enterprise data source can be ingested. When the core engineering challenge is "how do we get accurate answers out of our documents," LlamaIndex is the framework to reach for.

**Pricing:**
- **Beginner:** $0 — the open-source framework is MIT-licensed and free. LlamaCloud Free tier: 10,000 credits/month (~1,000 pages of standard parsing), 1 user, 5 indexes.
- **Startup:** LlamaCloud Starter tier: 40,000 credits/month (~4,000 pages standard) + pay-as-you-go up to $500 cap/month, 5 users, 50 indexes. Credits: 1,000 credits = $1; complex document parsing costs more per page.
- **Enterprise:** LlamaCloud Enterprise (custom) — private VPC deployment (documents never leave your cloud tenant), Enterprise SSO, unlimited users and indexes, volume discounts, dedicated support with SLAs. Required for regulated industries. Annual contracts range from $100,000 to $500,000+ for large document-intensive deployments.

---

### Mastra

**Year Released:** October 2024 (initial launch); v1.0 January 2026; Y Combinator W25 batch

**Coding Language:** TypeScript / JavaScript (Node.js)

**Has GUI:** Yes — Mastra Studio is a local development UI for inspecting agents, tracing workflows, and testing tool calls during development. Mastra Cloud provides a production monitoring and deployment dashboard.

**HITL Integration:** Yes — durable workflows support interrupt/resume patterns that pause execution at defined points for human review, approval, or input before the workflow continues

**Still Adding New Features:** Yes — v1.0 released January 2026, Series A closed April 2026; actively developed with new integrations, platform features, and MCP enhancements shipping regularly

**Summary:** Mastra is the "batteries-included" TypeScript-native agent framework — it ships the full set of production agent primitives (agents, durable workflows, all four memory types, RAG, evals, observability) pre-assembled and interoperable, designed from the ground up for JavaScript and TypeScript developers. Think of it as what a TypeScript engineer would build if starting fresh knowing all the production problems: working memory that persists across sessions, workflows that survive server restarts, retrieval that actually scales. Built on top of the Vercel AI SDK. Key components: Agent, Workflow (durable execution), Memory (working/semantic/entity/procedural), RAG pipeline, Tools, Evals, and Mastra Studio.

**Notable Users:**
- **Replit** — Agent 3 is built on Mastra, powering their autonomous coding agent for millions of users
- **Marsh McLennan** — agentic enterprise search tool deployed to 100,000+ employees across the global insurance firm
- **PayPal** — production AI agents for internal and customer-facing workflows
- **Adobe** — production agent implementations for creative and enterprise workflows
- **Factorial** — "One" HR AI agent with strict permission controls and hallucination prevention, now extended to workflow automation
- **Brex, Sanity, Docker, Elastic** — various production workflow automation use cases

**Industry Adoption:** Dominant in the TypeScript ecosystem — the only production-capable TypeScript-native agent framework. Strongest adoption in developer tools/platforms, financial services/insurance, HR technology, content/creative platforms, and enterprise search. Growing rapidly in enterprise after the Marsh McLennan (100K+ employees) and Replit (millions of users) production deployments became public.

**Community Size:** 22,300+ GitHub stars; ~1.8M monthly npm downloads (grew 30x in eleven months); 4,800+ Discord members; 300+ contributors; total funding: $35.5M ($13M seed Oct 2025, $22M Series A led by Spark Capital, April 2026)

**Why Choose This Framework:** Mastra is the only production-grade agent framework built natively for TypeScript. Teams that live in Node.js, Next.js, or Cloudflare Workers no longer need to introduce a Python runtime for production agent capabilities. Batteries-included means memory that actually works across sessions, durable workflows that survive failures, and built-in RAG — not bolt-on integrations. The seed investor list (Vercel CEO, Replit CEO, Elastic founder) represents the platforms Mastra deploys on and the products it powers, signaling deep ecosystem alignment.

**Pricing:**
- **Beginner:** $0 — the framework (Apache 2.0) is completely free. Mastra Cloud Starter tier provides managed deployment at no cost for basic usage. LLM API costs only.
- **Startup:** Mastra Cloud Pro (contact/usage-based, pricing TBD at time of research — estimated $50–$200/month for small teams). LLM API costs for moderate production volume: approximately $300–$800/month depending on model.
- **Enterprise:** Mastra Platform Enterprise (contact sales) — enterprise governance, SSO/SAML, SOC 2 compliance (in progress), on-premise or hybrid deployment, dedicated SLA-backed support. Estimated $50,000–$300,000+/year at large enterprise scale.

---

### Microsoft Agent Framework

**Year Released:** Public preview October 1, 2025; General Availability (v1.0) April 3, 2026

**Coding Language:** Python and .NET (C#), with full feature parity across both runtimes

**Has GUI:** Yes — the Foundry Portal on Azure provides an observability dashboard, trace viewer, and agent management interface; the broader Azure AI Foundry platform includes visual tooling for deployment and monitoring

**HITL Integration:** Yes — checkpoint-based human approval gates can be inserted at any point in a workflow; the middleware pipeline architecture supports content review and human sign-off steps before execution continues

**Still Adding New Features:** Yes — GA April 2026, version 1.3.0 as of May 8, 2026; active development with releases shipping every few weeks

**Summary:** Microsoft Agent Framework uses a dual-track architecture — Agent Orchestration (open-ended LLM-driven reasoning for tasks where the path is unknown) and Workflow Orchestration (deterministic business logic for tasks with defined steps), both composable within the same application. Think of it as a factory floor where some stations follow strict procedures (workflow) and some involve skilled workers making judgment calls (agent), and the two work seamlessly together. It is the official successor to AutoGen and Semantic Kernel, merging both into one enterprise-grade platform. Key components: AgentOrchestration, WorkflowOrchestration, SemanticKernel (middleware layer), Foundry Agent Service (managed runtime), SessionState, and OpenTelemetry observability.

**Notable Users:**
- **Novo Nordisk** — multi-agent system helping data scientists derive insights from complex pharmaceutical data in production
- **KPMG** — "Clara AI," a multi-agent audit system with governance and observability for regulated audit workflows
- **BMW** — agents analyzing terabytes of vehicle telemetry to deliver actionable engineering insights
- **Commerzbank** — avatar-driven customer support agents in financial services
- **Fujitsu** — enterprise integration services enabling human-AI collaborative workflows
- **TCS, Accenture** — framework embedded in enterprise AI offerings for Fortune 500 clients
- **Microsoft internal** — Copilot feature development and developer tools across multiple product teams

**Industry Adoption:** Primarily enterprise Azure deployments in regulated industries. Strongest adoption in financial services/audit, pharma/life sciences, automotive/manufacturing, enterprise IT, and professional services/consulting. Named alongside LangGraph in Gartner and Forrester analyst notes as one of two frameworks most likely to see meaningful enterprise adoption in 2026. Strong tailwinds from the AutoGen and Semantic Kernel migration paths.

**Community Size:** `microsoft/agent-framework` repo: ~10,000+ stars (growing rapidly post-GA); predecessor `microsoft/autogen`: ~43,000+ stars; `microsoft/semantic-kernel`: ~28,000+ stars; PyPI: crossed 1 million monthly downloads within first 3 months of GA; Microsoft Build 2025 and 2026 featured prominently

**Why Choose This Framework:** Microsoft Agent Framework is the right choice for Azure-committed enterprise teams. It inherits AutoGen's multi-agent orchestration and Semantic Kernel's enterprise plumbing — session persistence, middleware pipelines, compliance hooks, and Azure Durable Functions checkpointing — in a single coherent platform. The only framework with full .NET/C# + Python dual runtime support. AutoGen users migrating to production have a published migration guide. For organizations within the Microsoft ecosystem (Azure, Microsoft 365, Foundry), it provides the deepest native integration of any framework.

**Pricing:**
- **Beginner:** $0 — the framework (MIT licensed) is completely free. Runs on any infrastructure against any LLM provider with no Microsoft platform fees required.
- **Startup:** Foundry Agent Service pay-as-you-go (no orchestration fees, model inference billed at Azure rates: ~$2.50/M input tokens, $15/M output tokens for GPT-4o-class). Estimated $250–$1,500/month in model costs for moderate production volume.
- **Enterprise:** Azure Enterprise Agreement pricing (custom) for Foundry Agent Service with managed compliance, private regions, and dedicated support. Agent 365 at $15/user/month (also bundled in Microsoft 365 E7 at $99/user/month) for end-user agent deployment. Large enterprise deployments run $50,000–$500,000+/year driven primarily by model consumption and Azure infrastructure.

---

### OpenAI Agents SDK

**Year Released:** March 11, 2025 (successor to the experimental Swarm project from 2024); significant harness/sandbox update April 2026

**Coding Language:** Python and TypeScript (both officially maintained with full feature parity)

**Has GUI:** Yes — the OpenAI platform dashboard provides native trace visualization for all agent runs; the OpenAI Frontier portal offers enterprise agent management and monitoring; no-code Workspace Agents builder is available in ChatGPT Business/Enterprise tiers

**HITL Integration:** Limited — the April 2026 harness update added task interruption support for long-horizon workflows, but HITL is not a first-class framework primitive in the way it is for LangGraph. Teams requiring robust human approval gates typically layer an orchestrator (LangGraph) on top.

**Still Adding New Features:** Yes — actively developed by OpenAI; major April 2026 update added harness for long-horizon tasks and native sandbox for file/code operations; TypeScript SDK released with voice and MCP support

**Summary:** The OpenAI Agents SDK is built around deliberate minimalism — four primitives, clean API, opinionated defaults. Think of it as a well-designed starter kit where OpenAI has made all the common decisions for you. An Agent has instructions, tools, and optionally a handoff list (other agents it can delegate to). Guardrails run in parallel to validate inputs and outputs. The Runner manages the execution loop. What you give up in flexibility you gain in speed and native platform integration. Key components: Agent (instructions + tools), Handoff (agent-to-agent delegation), Guardrail (input/output safety checks), Tool (any callable function or hosted tool), and Runner (the execution loop).

**Notable Users:**
- **Uber** — customer support agents handling driver inquiries via OpenAI Frontier
- **State Farm** — claims processing agents that accelerate the insurance claims pipeline
- **GitHub** — multi-agent systems for end-to-end engineering work including code review
- **Notion** — multi-agent productivity and workspace automation
- **Carlyle Group** — due diligence framework with 50% reduction in development time and 30% improvement in agent accuracy
- **BBVA, Cisco, T-Mobile, Oracle, HP, Intuit** — early Frontier enterprise adopters across financial services, IT, and telecom

**Industry Adoption:** Broadest potential user base of any agent framework — every OpenAI API customer is a natural adopter. Strongest production adoption in financial services, software engineering/developer tools, customer support, insurance, and enterprise productivity. Voice agent adoption growing through the TypeScript SDK's RealtimeAgent support. 10.3 million monthly PyPI downloads reflects the scale of the OpenAI platform rather than purely agent framework adoption.

**Community Size:** ~20,700 GitHub stars (Python repo, launched March 2025); 10.3M monthly PyPI downloads; 4,900+ dependent projects; TypeScript repo maintained separately; 1 million+ businesses on the broader OpenAI platform

**Why Choose This Framework:** The OpenAI Agents SDK is the fastest on-ramp to production agents for teams already committed to OpenAI. Its deliberate minimalism means less time fighting the framework and more time building. Native tracing to the OpenAI dashboard, built-in guardrails, hosted tools (web search, code interpreter, file search), and first-class voice agents via RealtimeAgent are not available at this level of integration in any other framework. TypeScript parity means full-stack teams can use one SDK. If your stack is OpenAI, this is the obvious starting point.

**Pricing:**
- **Beginner:** $0 SDK cost + OpenAI API pay-per-token. GPT-5.4 mini at $0.75/M input, $4.50/M output is the cost-efficient starting point. Light development usage: ~$5–$20/month in API costs.
- **Startup:** API pay-as-you-go (GPT-5.4 at $2.50/M input, $15/M output). At 50,000 agent turns/month: approximately $250–$750/month. ChatGPT Business ($25/user/month) adds data privacy guarantees and Workspace Agents for non-developer users.
- **Enterprise:** OpenAI Frontier (custom, contact sales, reported six-figure annual floor) for managed deployment with Forward Deployed Engineers and governance. ChatGPT Enterprise (~$40–60/user/month) for end-user access with SSO and SOC 2. Large enterprise total annual cost: $200,000–$1,000,000+ at significant API volume.

---

### Pydantic AI

**Year Released:** December 2024 (initial release); v1.0 September 2025

**Coding Language:** Python

**Has GUI:** Yes — Pydantic Logfire provides a production observability dashboard with trace visualization, LLM experiment comparison, and monitoring, available both as a cloud service and as a self-hosted deployment via open-sourced Helm chart

**HITL Integration:** Limited — basic support for pausing and resuming agent runs; HITL is not a first-class design primitive. Teams requiring robust human-in-the-loop workflows typically combine Pydantic AI with LangGraph (using Pydantic AI agents inside LangGraph nodes that have interrupt support).

**Still Adding New Features:** Yes — among the most actively maintained frameworks; releases ship every few days; v1.93.0 as of May 9, 2026

**Summary:** Pydantic AI treats agents as software — not personas or workflow nodes, but typed Python objects with declared output schemas, validated tool calls, and dependency containers that can be swapped for mocks in tests. Think of it as bringing the same engineering discipline to agents that you would apply to any production API: type checking, dependency injection, unit testing, and structured error handling. Built by the team behind Pydantic (the most widely used data validation library in Python), the framework uses Pydantic models for all inputs and outputs, ensuring LLM responses are always validated before they reach your application code. Key components: Agent (typed Python object with tools and deps), Tool (type-annotated function with validated inputs/outputs), RunContext (dependency injection container), and Logfire (observability).

**Notable Users:**
- **MindsDB** — migrated from LangChain; achieved 10x agent performance improvement and 150x query performance improvement within one month, enabling an enterprise deal
- **Lema AI** — Agentic Risk Engineer for autonomous third-party security investigation
- **Sophos** — SecOps AI team uses Pydantic Logfire for unified tracing of AI-powered security solutions
- **OpenBB** — FinAI platform for structured, type-safe financial model interactions
- **ARIJ Network** — RAG-based AI chatbot for investigative journalists across 22 countries in the Middle East and North Africa

**Industry Adoption:** Growing rapidly across cybersecurity (structured threat intelligence output), financial services/fintech (validated model outputs for financial calculations), data analysis and developer tools, journalism/media (fact-critical generation), and security operations. Described by practitioners as "the adult choice for production Python agents" — strongest with engineering-driven organizations moving from demo to maintainable production system. Thoughtworks Technology Radar: "Trial" designation (practitioner-relevant, worth adopting where it fits).

**Community Size:** ~17,000 GitHub stars (launched December 2024); releases every few days; parent Pydantic library: 300M+ monthly PyPI downloads, 10 billion+ total downloads crossed in 2026; total funding: $17M Series A from Sequoia

**Why Choose This Framework:** Pydantic AI is the right framework for Python engineering teams who want to apply production software discipline to agent development. Type-safe validated outputs mean LLM responses are programmatically verified before reaching your application — eliminating a whole class of runtime errors. Dependency injection enables proper unit testing of agent logic by swapping real LLM calls for deterministic mocks. Support for 30+ model providers means no vendor lock-in. Built by the team behind the most widely used Python data validation library — this is not a startup framework but an extension of infrastructure that already runs in the OpenAI SDK, Anthropic SDK, FastAPI, and LangChain.

**Pricing:**
- **Beginner:** $0 — Pydantic AI SDK is MIT-licensed and completely free. Logfire Personal tier: $0/month with 10 million spans included (generous for solo development). LLM API costs only.
- **Startup:** Logfire Team at $49/month for 5 seats + 10M spans/month. Overage at $2/million spans. LLM API costs for moderate production volume (100K agent runs/month): approximately $200–$400/month. Total: ~$250–$450/month.
- **Enterprise:** Logfire Enterprise (custom, contact sales) — SLA guarantees, HIPAA BAA, custom data retention, SSO/SAML, and self-hosted deployment via Helm chart. Available on AWS Marketplace. Estimated $50,000–$200,000+/year at large enterprise scale based on comparable observability platform pricing.

---

### Strands Agents

**Year Released:** May 2025 (public preview); v1.0 July 2025 (by AWS)

**Coding Language:** Python (primary); TypeScript SDK in development

**Has GUI:** No official visual IDE or Studio interface. Observability is handled through first-class OpenTelemetry tracing forwarded to third-party dashboards (AWS CloudWatch, Langfuse, Arize AX). A community-built `strands.my` dashboard tracks ecosystem metrics but is not an authoring or debugging tool.

**HITL Integration:** Yes — the `handoff_to_user` built-in tool pauses agent execution and transfers control to a human, preserving full conversation context during the handoff. The agent can also be configured to ask clarifying questions inline before continuing. More structured approval gates are available via the Graphs multi-agent primitive, which supports conditional routing with defined decision points.

**Still Adding New Features:** Yes — v1.0 released July 2025, less than two months after the May 2025 preview; active development with regular releases adding model providers, community-contributed tools, MCP integrations, and ecosystem features

**Summary:** Strands takes a model-driven approach — instead of you wiring up a workflow graph, the LLM drives its own agent loop. You hand it a system prompt, a list of tools, and a model; it decides what to call and when, looping until it produces a final answer. Think of it as a smart contractor: you give them the job description and a toolbox, and they figure out the sequence of steps on their own without a project manager micromanaging each decision. The framework was built internally at AWS and open-sourced in May 2025 after it had already proven itself in production on Amazon Q Developer, AWS Glue, and Kiro. Version 1.0 extended this single-agent loop into four multi-agent patterns: **Agents-as-Tools** (specialist agents callable as tools by an orchestrator), **Handoffs** (explicit control transfer with preserved context), **Swarms** (autonomous agent teams coordinating through shared memory), and **Graphs** (deterministic workflows with conditional routing). All four patterns compose freely. Key components: Agent, Tool (`@tool` decorator), Model (any of 15+ supported providers), SessionManager (durable state persistence), and AgentCore (AWS managed hosting platform).

**Notable Users:**
- **Amazon Q Developer** — internal agentic AI assistant for software development; Strands cut agent deployment time from months to days on the Q Developer team
- **AWS Glue** — data integration and ETL pipeline agents in production
- **VPC Reachability Analyzer** — network diagnostics agents that reason over AWS infrastructure topology
- **Kiro (AWS IDE)** — AI-powered coding assistant with multi-step agentic task automation
- **Amazon Transform** — application modernization at scale, using Strands to analyze and rewrite legacy codebases
- **Accenture** — enterprise agent deployments across industries; contributed code to the Strands SDK
- **PwC** — finance and life sciences agent workflows under a collaboration with Anthropic for regulated, mission-critical deployments

**Industry Adoption:** Heaviest adoption inside the AWS ecosystem, where the zero-configuration Bedrock, IAM, and S3 integrations make Strands the default path. Production deployments span cloud infrastructure management, software development, application modernization, financial services (meeting action capture, compliance workflows), and life sciences. AWS reported 5 million+ total PyPI downloads as of early 2026, driven largely by AWS-native engineering teams. Model-agnostic design (Bedrock, Anthropic, OpenAI, Ollama, LiteLLM, Llama, and more) means it is usable beyond AWS, but the gravitational pull of the ecosystem is strong.

**Community Size:** 2,000+ GitHub stars (as of July 2025 v1.0 launch); 5+ million total PyPI downloads; ~1.4 million weekly PyPI downloads as of May 2026; 22% of v1.0 pull requests contributed by community members; contributors from Accenture, Anthropic, Meta, PwC, Cohere, Mistral, Writer, Langfuse, mem0.ai, and Tavily

**Why Choose This Framework:** Strands is the fastest path from zero to a working AWS-native agent. The model-driven loop eliminates the graph-definition overhead that makes LangGraph steep for prototyping, while still delivering production-grade tooling — built-in OpenTelemetry, durable session management, async support, and a clear path to Bedrock AgentCore for managed hosting. For teams already running on AWS, IAM and Bedrock connect automatically without any configuration. The A2A protocol support and MCP integration mean Strands agents can interoperate with agents built on other frameworks and tap a large ecosystem of pre-built tools immediately. Four composable multi-agent primitives cover the most common real-world patterns without forcing a choice between them.

**Pricing:**
- **Beginner:** $0 — the Strands SDK is Apache 2.0 licensed and completely free. LLM API costs (billed by your chosen model provider) are the only expense. Running locally against Bedrock or the Anthropic API: approximately $0–$20/month for testing and development.
- **Startup:** $0 framework cost + AWS infrastructure + LLM inference. A production agent on AgentCore Runtime with moderate traffic (a few hundred sessions/day) runs approximately $100–$400/month: Runtime compute at $0.0895/vCPU-hour + $0.00945/GB-hour (charged only for active compute time), Memory operations, and LLM tokens via Bedrock. Self-hosting on Lambda or Fargate is cheaper but requires more DevOps effort.
- **Enterprise:** Amazon Bedrock AgentCore at consumption-based rates with no minimums: Runtime compute, Gateway ($0.005/1,000 tool API calls), Memory ($0.25–$0.75/1,000 operations), and Identity. Large enterprise deployments with dozens of agents and high tool-call volumes run $5,000–$20,000+/month before LLM inference. AWS Enterprise Discount Program (EDP) agreements typically provide 20–40% discounts at scale. Pricing verified at `aws.amazon.com/bedrock/agentcore/pricing/` as of early 2026.

---

## Framework Comparison Tables

### Table 1 — Orchestration: Controller and Worker Behaviour

| Framework | Controller Behaviour (How control decisions are made) | Worker Behaviour (How actions run) |
|---|---|---|
| **AutoGen / AG2** | A `GroupChatManager` (LLM-backed) selects the next speaker at every turn through emergent LLM reasoning — no explicit graph. In two-agent loops, the initiating agent drives turns directly. Stopping is via a TERMINATE signal, a max-turn cap, or a custom condition. | `AssistantAgent` generates text or code using its configured LLM. `UserProxyAgent` executes code blocks (Python or shell) in a local, Docker, or Jupyter sandbox and returns stdout/stderr as the next conversation message. Each agent at each turn receives the full accumulated conversation history. |
| **CrewAI** | **Sequential process**: tasks run in predefined order, each output passed as context to the next. **Hierarchical process**: an LLM-backed manager agent dynamically assigns tasks and reviews outputs. **Flows**: a `@start`/`@listen`/`@router` decorator chain drives deterministic, event-based step execution in code. | Each Agent executes its assigned task using its configured LLM, tools, and backstory. Agents can delegate subtasks to each other mid-task (inter-agent delegation) when they determine another agent is better suited. Flow steps can trigger full Crew executions as sub-tasks. |
| **Haystack** | The `Pipeline` directed graph routes data through components via explicit developer-defined connections. `ConditionalRouter` and `MetadataRouter` components inspect incoming data and branch flow in code — no LLM decides routing in the pipeline itself. Inside an `Agent` component, the LLM does drive tool selection for that one loop. | Individual `Component` objects (retrievers, rerankers, generators, etc.) execute typed, validated operations and pass results downstream via their declared output sockets. The `Agent` component manages its own LLM tool-calling loop as a single node inside the larger pipeline, accumulating results in a `state_schema` dict. |
| **LangGraph** | A `StateGraph` defines all possible paths as nodes and edges in code. **Conditional edges** call a routing function that inspects shared typed state and returns the name of the next node — fully deterministic. `interrupt_before`/`interrupt_after` on any node pauses execution for human review before routing continues. | Each Node is a Python function (or async function) that reads the current state, does its work (LLM call, tool call, retrieval, arbitrary logic), and returns state updates. The framework merges those updates into the shared state and sends it to the next node. No node knows about other nodes directly. |
| **LlamaIndex** | **Agents**: the LLM selects which tool to call via ReAct prompting or native function-calling (FunctionCallingAgent). **Workflows**: an event-passing model where `@step`-decorated async functions handle specific event types and emit new events; routing is explicit `if/else` logic inside step functions. Router Query Engines select between sub-engines in code. | `QueryEngine` handles the full retrieval-plus-generation cycle for a single query. `Agent` executes tool calls (which may be query engines, function tools, or other agents) and appends results to the context. Workflow steps are stateless by default — state must be explicitly threaded through the `Context` object. |
| **Mastra** | **Agents**: autonomous LLM-driven loop — the model selects which tools to call and when. **Workflows**: a deterministic step graph where `Step` nodes have Zod-validated typed I/O; conditional branching is declared in code. The two modes compose freely (a workflow step can invoke an agent; an agent can call another agent as a tool). | Tools are TypeScript functions with Zod-validated schemas; the agent loop calls them, receives typed results, and continues until a final answer. Workflow steps execute in sequence or parallel and persist state to storage at each step — a failed step restarts from its last checkpoint, not from the beginning. |
| **Microsoft Agent Framework** | **Workflow Orchestration**: a typed graph of Executors connected by deterministic, code-defined edges (Sequential, Concurrent, Handoff patterns). **Agent Orchestration**: LLM-driven routing (Group Chat, Magentic patterns) where a manager agent builds a task ledger and dispatches. A Middleware pipeline of interceptors wraps every invocation (pre- and post-processing). | Each `Agent` (ChatCompletionAgent, StructuredOutputAgent, ToolCallAgent) processes its turn with an LLM, executes tool calls, and returns a result. `AgentSession` accumulates multi-turn state separately from the agent, keeping agents stateless and reusable. Context Providers enrich inputs before the model is called. |
| **OpenAI Agents SDK** | Routing is entirely LLM-driven. The orchestrating agent's LLM decides which tool to call or which `handoff` to invoke; there are no conditional edges or developer-defined routing tables. Guardrails run in parallel to validate inputs and outputs and can cancel execution via a tripwire if they trip. `max_turns` prevents infinite loops. | The `Runner` manages the agent loop: send context + tool schemas → receive model response → execute tool call or handoff → repeat. Tools can be local `@function_tool` callables, hosted tools (web search, code interpreter, file search), or MCP server tools. Handoffs transfer the full conversation history to the receiving agent. |
| **Pydantic AI** | Single-agent: the LLM selects tools to call within a typed, validated loop. Multi-agent: **agent-as-tool** (one agent calls another's `run()` inside a tool function) or **pydantic-graph** state machine for deterministic routing. Usage limits (`max_tokens`, `max_tool_calls`) can cap any run before costs escalate. | Tools are type-annotated Python functions that receive a `RunContext[Deps]` carrying injected dependencies; the framework validates their arguments against auto-generated schemas and retries automatically on validation failure. The final output is validated against the agent's declared Pydantic output model before it leaves the framework. |
| **Strands** | Model-driven loop — the LLM reads full conversation history and decides which tool to call (or signals completion) with no explicit routing logic. For multi-agent: **Agents-as-Tools** (orchestrator delegates to specialists), **Handoffs** (explicit control transfer with `handoff_to_user`), **Swarms** (agents coordinate via shared memory), or **Graphs** (deterministic conditional routing via `GraphBuilder`). | Tools are `@tool`-decorated Python functions; Strands uses their docstring and signature to generate a tool spec. The agent loop appends tool results to conversation history and runs the model again until a final answer is produced. For Swarms, each specialist agent reads and writes to a shared memory store without a central coordinator directing the order. |

---

### Table 2 — Integration: How Tools and APIs Are Wired

| Framework | How tools and APIs are wired | Behaviour you should expect |
|---|---|---|
| **AutoGen / AG2** | Any Python callable registered on an agent becomes a tool. `UserProxyAgent` executes code blocks produced by the LLM in a local or Docker sandbox. AG2 supports MCP server connections for external tool registries. Pre-built integrations exist for web search, file I/O, and cloud APIs. | The LLM may produce code instead of tool calls — the code execution loop is first-class. Tool calls can repeat across many turns. In GroupChat, the manager LLM may invoke the same tool multiple times with slightly different arguments before converging. No framework-level retry on bad arguments. |
| **CrewAI** | Tools are Python callables assigned to individual agents at crew assembly. LiteLLM provides a unified 200+ model interface. 1,200+ integrations available via native connectors (web search, file I/O, database, browser, code execution, RAG). MCP-compatible. Tools can also be other CrewAI agents. | Each tool call goes through the agent's LLM decision loop. In hierarchical mode, a manager may reassign tool-equipped agents mid-task. Expect agents to use tools autonomously without strict call-order guarantees; unexpected delegation loops are the most common failure mode. |
| **Haystack** | `ComponentTool` wraps any Haystack component as a callable tool; `PipelineTool` wraps an entire retrieval pipeline as a single tool. `SearchableToolset` enables keyword-based tool discovery for large tool catalogs. `Hayhooks` exposes pipelines as REST APIs and as MCP tool endpoints — any MCP client can call a Haystack pipeline as a tool. | Every component enforces typed input/output sockets validated at connection time and at run time. Calling a `PipelineTool` internally runs a full multi-component retrieval pipeline (retrieve → rerank → generate) and returns the result as a single structured output. Loops (e.g., self-correction) are expressed as explicit cycle edges in the pipeline graph. |
| **LangGraph** | Tools are Python functions wrapped using LangChain's `@tool` decorator and invoked inside dedicated `ToolNode` graph nodes. Any LangChain-compatible integration (100s of retrievers, search APIs, databases, REST clients) works natively. Human-in-the-loop pauses are wired into the graph itself via `interrupt_before`/`interrupt_after` rather than tool callbacks. | Tool execution is scoped to specific graph nodes — the graph controls which tools are available at which step. Results are written to shared typed state, not appended to a message history. The same tool can be called from multiple nodes in different graph contexts. Node-level caching (added May 2025) can skip redundant tool calls during development. |
| **LlamaIndex** | 300+ LlamaHub data loaders ingest from cloud storage, databases, SaaS APIs, and file formats. Query engines registered on agents become tools. The `FunctionTool` wraps any Python callable. The `QueryEngineTool` wraps a retrieval pipeline. Additional agents can be called as tools (nested agents). Cloud APIs accessed via LlamaHub connector packages. | Tool selection is LLM-driven (ReAct or function-calling). When a query engine is a tool, calling it runs the full retrieval + generation pipeline internally before returning a response. Long-running Workflow tasks have no built-in checkpoint recovery — a failed step restarts the workflow from the beginning. |
| **Mastra** | Tools created with `createTool` and Zod input/output schemas — the framework auto-generates the JSON schema the model uses. ~50–60 `@mastra/*` npm integration packages (GitHub, Slack, Google services, databases, vector stores). MCP supported bidirectionally: Mastra agents can consume external MCP tools and expose their own tools as an MCP server. Every agent is automatically an HTTP endpoint with an OpenAPI spec. | Zod schema validation runs on every tool input and output; type errors surface as structured exceptions rather than unstructured strings. One agent can call another as a tool (`agent-as-tool` pattern). All tool calls are instrumented with OpenTelemetry traces automatically. Failed tools raise typed exceptions; framework does not auto-retry — retry logic is handled in the workflow abstraction. |
| **Microsoft Agent Framework** | Tools are typed Python or .NET callables registered on agents. Native first-class MCP support: agents connect to MCP servers at startup or dynamically at runtime, auto-discovering available tools. Foundry Tools marketplace provides managed connectors for Microsoft Graph, SharePoint, Bing Search, Azure Blob Storage, and SaaS APIs. Azure Durable Functions integration enables long-running orchestration across days or weeks. | The Middleware pipeline runs on every invocation (before the model call and after the response) — safety filters, logging, compliance policies, and retry logic are applied globally without modifying agent logic. Context Providers inject retrieval results (RAG, structured data) into the agent's context before each LLM call. Checkpoint-based recovery means a failed workflow executor restarts from its last saved state, not from scratch. |
| **OpenAI Agents SDK** | `@function_tool` decorator on any Python/TypeScript function auto-generates a JSON schema from type annotations and docstring (via Pydantic). Hosted tools (web search, code interpreter, file search, computer use) are available with no external configuration — just include them in the tool list. MCP server connections work natively in both SDKs. Voice tools available via `RealtimeAgent` in the TypeScript SDK. | The LLM decides which tool to call and in what order — there is no framework-level routing. Guardrails run in parallel using a smaller, faster model; if a guardrail trips, it cancels the main execution via tripwire. The April 2026 harness adds checkpoint-based state persistence for long-horizon sandbox tasks, but standard runs without the harness have no built-in recovery. |
| **Pydantic AI** | `@agent.tool` or `@agent.tool_plain` decorators auto-generate JSON schemas from type annotations and docstrings. Dependencies are injected into every tool via `RunContext[Deps]` — swapping database connections or API clients for test mocks requires no agent code changes. MCP tool server connections and Agent-to-Agent (A2A) protocol supported. 30+ LLM providers switchable via a one-line model string change. | Invalid tool arguments auto-retry with an error message fed back to the LLM. Outputs are validated against the agent's declared Pydantic output model — the run raises `ValidationError` if the model cannot produce a conformant response after retries. `UsageLimitExceeded` raises before token or tool-call caps are hit, preventing runaway costs. All instrumented with OpenTelemetry automatically. |
| **Strands** | `@tool` decorator on any Python function extracts docstring + type signature to create the tool spec the model reads. `strands_tools` package ships 40+ pre-built tools (calculator, web search, file I/O, shell, memory, code execution, etc.). MCP tools are surfaced identically to function tools — the agent does not distinguish between them. AgentCore Gateway provides a managed API proxy for enterprise integrations. | The model decides tool call order autonomously; the full conversation history is sent on every loop iteration. There is no framework-level retry on bad arguments — retry behavior is delegated to individual tool implementations or the model provider SDK. MCP tools auto-integrate when a new MCP server is connected; no additional wiring is required. Strands exposes A2A protocol endpoints so agents from other frameworks can call Strands agents as network services. |

---

### Table 3 — State and Knowledge: Memory and Knowledge Base Integration

| Framework | How memory and session state work | How knowledge bases usually plug in |
|---|---|---|
| **AutoGen / AG2** | Memory is the conversation history — the accumulated list of messages passed to every agent at every turn. No built-in cross-session persistence in v0.2; the v0.4/AG2 rewrite adds a `ChatHistory` abstraction and externalized state, but durable session recovery across process failures remains an application-layer responsibility. | No native RAG or document store primitives. Knowledge bases plug in as tools called by agents — typically a Haystack retrieval pipeline or LlamaIndex query engine wrapped as a callable tool. The entire retrieval result is injected into the conversation as a tool message. |
| **CrewAI** | Four memory layers: **short-term** (RAG over recent interactions for in-session context), **long-term** (valuable insights persisted across sessions to an external store), **entity memory** (facts about named entities extracted from tasks), **external memory** (LangMem or custom integrations for cross-conversation knowledge). Flows maintain a Pydantic or dict-based state object shared across all steps, serializable between runs. | Vector store retrieval is configured as an agent tool (any vector database via a custom `Tool` or the `CrewAI Knowledge` component for simple document Q&A). For richer retrieval, teams expose a LlamaIndex or Haystack pipeline as a CrewAI tool. The RAG result is injected into the agent's task context before the LLM processes it. |
| **Haystack** | Within an agent run, the `state_schema` dict accumulates typed data across all tool call iterations — tools can read prior results and append findings. Pipeline state is in-memory per run; there is no built-in cross-session persistence or checkpoint recovery for agents (individual components are stateless). | `DocumentStore` (Elasticsearch, Qdrant, Pinecone, Weaviate, pgvector, Chroma, MongoDB, etc.) is a first-class framework primitive. Hybrid retrieval (BM25 + dense), semantic reranking, table-aware retrieval, and multimodal retrieval are all built-in pipeline components. A full retrieval pipeline can be exposed as a `PipelineTool` callable by an agent — encapsulating the entire knowledge base behind a single tool interface. |
| **LangGraph** | Shared typed state (defined as a `TypedDict` or Pydantic model) threads through all graph nodes. A `Checkpointer` (in-memory, SQLite, PostgreSQL, or Redis) saves the full state after every node execution. `thread_id` scopes conversations so state persists across user sessions, server restarts, and process failures. Time-travel debugging lets developers rewind state to any prior checkpoint and re-run from that point. | LangChain retrievers, vector stores (Pinecone, Weaviate, Chroma, pgvector, etc.), and Haystack/LlamaIndex pipelines integrate as standard tool nodes in the graph. A typical pattern is a `retrieval_node` that calls a retriever and writes the results to state; a downstream `generation_node` reads those results from state and calls the LLM. |
| **LlamaIndex** | Agents use a `ChatMemoryBuffer` (rolling window of conversation history) for short-term memory. Long-term memory requires explicit memory modules backed by a separate vector store, surfacing past facts via semantic similarity — less seamlessly integrated than the document retrieval layer. Workflows are stateless by default — state must be explicitly passed via the `Context` object. No built-in checkpoint-based recovery for failed long-running workflows. | Data pipeline primitives are the framework's core: 300+ LlamaHub integration packages (data loaders, vector store connectors, LLM providers, embedding models, and tools) covering virtually any source; Node parsers handle chunking strategy; `VectorStoreIndex`, `SummaryIndex`, `KeywordTableIndex`, and `KnowledgeGraphIndex` organize nodes for different query strategies. `LlamaParse` handles complex enterprise document formats. `LlamaCloud` provides managed indexing pipelines that keep indexes current as documents change. Query engines become agent tools directly. |
| **Mastra** | Four-tier memory system: **message history** (configurable retention window), **working memory** (a Zod-typed structured object persisted across sessions — not conversation history, but structured state like user preferences or accumulated task context), **semantic recall** (vector similarity search over past messages, surfacing relevant context by meaning rather than recency), and **observational memory** (background compression achieving 5–40× more effective context use within the same token window, ~95% LongMemEval accuracy). All tiers backed by configurable storage adapters. | Built-in RAG via `MDocument` chunking, embedding, and vector retrieval (Pinecone, Qdrant, Weaviate, Chroma supported). For enterprise document intelligence (complex PDFs, tables, mixed formats), teams typically call an external service (e.g., a LlamaIndex or Haystack API) as a tool. Semantic recall memory itself functions as a knowledge base over past agent interactions, retrieved by meaning at query time. |
| **Microsoft Agent Framework** | `AgentSession` is a scoped container that accumulates message history, tool call records, and metadata across multi-turn conversations. Sessions are serializable and checkpointable to durable storage — a long-running workflow that fails resumes from its last checkpoint rather than restarting. The Middleware pipeline captures the full interaction history as a compliance audit log. Long-term memory has no built-in implementation; it is left to external Context Providers. | Context Providers inject retrieval results into the agent's context before each LLM call — plugging in Azure AI Search, Azure Cosmos DB, or any vector store via MCP. Foundry Tools marketplace provides managed retrieval integrations for Microsoft Graph and SharePoint. The framework provides the injection hook; the storage and retrieval strategy are externally configured. |
| **OpenAI Agents SDK** | Within a single run, conversation history is the short-term memory. The April 2026 harness adds a configurable working memory layer for maintaining context across steps of a long-horizon task, plus checkpoint-based state persistence so sandbox-based runs can survive container loss. Without the harness, there is no built-in cross-session persistence — interrupted runs restart from the beginning. Long-term memory across separate runs is application-layer responsibility. | The `file_search` hosted tool provides vector search over uploaded files with no external infrastructure required — files are uploaded to OpenAI's API and queried directly. For external knowledge bases, function tools call any retrieval system (Pinecone, Weaviate, Elasticsearch, etc.). External memory services (e.g., Mem0) integrate via function tools. There is no native document parsing or index management in the framework. |
| **Pydantic AI** | No built-in session persistence — conversation history is in-memory per run. Durable execution for multi-step workflows is available via the `pydantic-graph` state machine, which supports explicit node transitions and can checkpoint between agent invocations. Long-term memory is left entirely to the application layer. The Dependency Injection system enables clean separation between agent logic and state-holding services (databases, caches) that persist externally. | No native document store or RAG primitives. Knowledge bases plug in as tool functions that call external retrievers (any vector store or search API). `pydantic-graph` enables structured knowledge state transitions for applications that model knowledge as explicit typed states. The Pydantic Harness capability bundles can package retrieval tool sets as reusable, attachable capabilities. |
| **Strands** | Short-term memory is the full conversation history, sent to the model on every loop iteration. `SessionManager` (introduced in v1.0) can persist and restore conversation history to file, S3, or a custom backend — enabling agents to survive compute restarts and scaling events. Long-term semantic memory is available via the `memory` tool from `strands_tools` or through Amazon Bedrock AgentCore Memory (managed semantic memory with configurable retention). For Swarms, a shared memory store is the coordination medium between agents. | MCP tool integration means any MCP-based knowledge tool (vector store, search index, document retrieval service) connects without custom code. Amazon Bedrock Knowledge Bases integrate as a tool with zero-configuration IAM-based access. For teams outside AWS, teams typically wire a vector database (Pinecone, Qdrant, etc.) or search service as a `@tool`-decorated function. AgentCore Memory provides managed short- and long-term semantic memory as a hosted service. |

---

## When to Introduce Fameworks to the Course

Options:

1. After introducing multi-agent systems and using a YAML to organize the different agents.
2. After introducing agents as tools
