# Agent Framework Research Guide

**Purpose:** This document is an instruction manual for AI agents tasked with researching agent frameworks. Follow it to produce reports that match the style, depth, and structure of the reference reports in this folder (`LangGraph_Research.md`, `CrewAI_Research.md`).

---

## Quick Reference

- **Output file naming:** `[FrameworkName]_Research.md` (e.g., `AutoGen_Research.md`)
- **Output location:** Save to the workspace folder (same directory as this guide)
- **Target length:** 400–600 lines of markdown
- **Number of sections:** 11 (see structure below)
- **Research rounds:** 3–4 parallel search batches before writing
- **Tone:** Professional, direct, opinionated where the evidence supports it — not a press release

---

## Part 1: Document Structure

Every framework research report must contain exactly these 11 sections, in this order. Do not skip or rename sections.

```
## 1. What Is [Framework]?
## 2. How It Works — Architecture Deep Dive
## 3. The [Framework] Ecosystem
## 4. Who Uses [Framework]?
## 5. Industries and Use Cases
## 6. Why People Choose [Framework]
## 7. Why People Don't Choose [Framework]
## 8. [Framework] vs Competing Frameworks
## 9. Community and Market Position
## 10. Pricing
## 11. Summary and Verdict
## Sources
```

Include a linked **Table of Contents** at the top referencing all 11 sections. After the title block and before the TOC, add a horizontal rule. After the TOC, add another horizontal rule before Section 1.

### Title Block Format

```markdown
# [Framework Name] Agent Framework — Deep Research Report

**Research Date:** [Month Day, Year]  
**Subject:** [Framework] — Architecture, Adoption, Use Cases, and Competitive Landscape

---
```

---

## Part 2: What to Cover in Each Section

### Section 1 — What Is [Framework]?

Answer: What is this thing, and why does it exist? Cover:
- A one-paragraph plain-English definition
- The founding story: who built it, when, and why (what problem were they solving?)
- The core metaphor or mental model the framework uses (graphs, crews/roles, conversations, etc.)
- License (MIT, Apache 2.0, etc.) and whether it is truly open source
- Current version/maturity milestone (e.g., "reached v1.0 in October 2025")
- 2–3 headline metrics that establish market presence (GitHub stars, monthly downloads, known users)
- One pull quote from official documentation or a founder that captures the essence

The section should end with the reader having a clear, single-sentence answer to "what is this framework and what makes it different."

### Section 2 — How It Works — Architecture Deep Dive

This is the most technical section. Go deep. Cover:
- The core primitives (the 3–5 fundamental concepts everything else is built on)
- How data/state flows through the system
- How the framework handles decision-making and routing (static vs. dynamic, explicit vs. emergent)
- What a minimal working code example looks like (include a short annotated code block)
- Any distinct execution modes or architectural patterns the framework supports
- How the framework handles errors, retries, and failure
- How it manages memory or context (short-term, long-term, none)
- Multi-agent coordination patterns it supports natively

**Code block guidance:** Include one short code example (10–20 lines) showing the simplest meaningful usage — typically: define an agent or node, wire two of them together, run it. Use real syntax, not pseudocode.

### Section 3 — The [Framework] Ecosystem

Cover the broader product family around the framework:
- What other tools does the same company make that integrate with it?
- Is there a managed cloud / SaaS hosting option? (Name it specifically)
- Is there a visual IDE, debugger, or Studio-style interface?
- Are there official integrations with cloud providers (AWS, Azure, GCP)?
- Is there a marketplace of tools, plugins, or templates?
- What observability/tracing tools are available (first-party or recommended third-party)?

### Section 4 — Who Uses [Framework]?

Name real companies. Do not be vague. Format as a table:

```markdown
| Company | Use Case |
|---|---|
| **Company Name** | One-sentence description of their specific use case |
```

Include 8–15 rows. Source from: official case studies, blog posts, conference talks, and credible third-party reporting. If a company has been named by the vendor in marketing material, include them. If possible, note the scale (e.g., "saves 10 hours/week per manager," "processes 500 queries/day").

### Section 5 — Industries and Use Cases

Organize by industry vertical, not by company. For each industry:
- Name the industry
- Describe the specific use pattern (what kind of workflow is being automated)
- Give 1–2 concrete examples tied to companies or published results from Section 4
- Note any particular framework feature that makes it well-suited for this vertical

Aim for 6–9 industry verticals. Common ones to check: financial services, software development, cybersecurity, HR/recruiting, customer support, marketing/content, healthcare, government/federal, legal, real estate/operations.

### Section 6 — Why People Choose [Framework]

This section argues the affirmative case. Write it as the strongest honest argument for the framework, grounded in evidence. Cover:
- The primary differentiating capability (the one thing it does best that others don't)
- Developer ergonomics (ease of getting started, documentation quality, mental model clarity)
- Ecosystem advantages (integrations, tooling, observability)
- Production-readiness signals (stability, persistence, error handling, scalability)
- Community and support quality
- Any features that are genuinely unique compared to all alternatives

Each reason should be a named subsection (`### Reason Name`) with 2–4 sentences of supporting detail. Aim for 6–9 reasons. Avoid bullet-point lists of adjectives — write in prose.

### Section 7 — Why People Don't Choose [Framework]

This section argues the honest negative case. It must be as substantive as Section 6. A framework that has no real weaknesses is not being described honestly. Cover:
- The primary architectural limitation
- Scenarios where the framework is a poor fit
- Developer experience pain points (documented in community forums, GitHub issues, blog posts)
- Performance or cost concerns
- Vendor/lock-in risks
- Anything the community consistently complains about

Same format as Section 6: named subsections, prose, 2–4 sentences each. Aim for 5–8 reasons. Do not soften weaknesses — honest assessments are more useful than marketing copy.

### Section 8 — [Framework] vs Competing Frameworks

Start with a comparison table covering all major competing frameworks:

```markdown
| Framework | Core Metaphor | Best For | Time-to-Demo | Production Maturity |
|---|---|---|---|---|
```

Then write a dedicated subsection for each major head-to-head comparison (typically 3–4 comparisons). For each:
- Explain the competing framework's core approach in 1–2 sentences
- State clearly: "Choose [this framework] when: ..."
- State clearly: "Choose [the competitor] when: ..."
- Name the key differentiating dimension (control vs. speed, explicit vs. emergent, etc.)
- Note whether the competitor is in active development, maintenance mode, or deprecated

Always cover at minimum: vs. LangGraph, vs. CrewAI, vs. AutoGen. Add others as relevant (LlamaIndex, OpenAI Swarm, Mastra, Google ADK, etc.).

### Section 9 — Community and Market Position

Cover:
- Key quantitative metrics in a named list: GitHub stars, monthly downloads, active deployments, community size
- Funding and company background (amount raised, investors, founder background, company location)
- Industry recognition (awards, rankings, analyst coverage, conference presence)
- Community sentiment summary (what practitioners consistently praise vs. consistently criticize — source from forums, Reddit, Discord, blog posts)
- Market context: where the framework sits in the broader AI agent ecosystem, and whether it is growing, plateauing, or declining in relevance

### Section 10 — Pricing

This section has a required structure. Follow it closely.

**Opening paragraph:** Immediately clarify what is free and what costs money. Most frameworks have a free open-source component and a paid commercial platform — state this distinction explicitly before any pricing table appears.

**Pricing table:**

```markdown
| Plan | Price | [Key Unit, e.g., Traces/Month] | Seats | [Key Feature] | Support |
|---|---|---|---|---|---|
```

Use 3–5 rows. Use bold for the Plan name column. Note if pricing requires contacting sales or logging in (meaning you are working from third-party sources).

**For each tier, write a paragraph** explaining:
- What type of user or organization this tier is designed for
- What the key unlocks are vs. the tier below
- Whether the limits are realistic for the stated audience
- Any important billing mechanics (overages vs. hard caps, annual vs. monthly billing)

**Real-world cost scenarios section:** Include 4 concrete scenarios at different scales:
1. Solo developer / side project
2. Small startup (3–5 people)
3. Mid-size team in production (20–50 people)
4. Large enterprise (100+ people)

For each: state total monthly/annual cost range and what plan they'd likely be on.

**Pricing caveats:** Note if prices require a login to view, if they change frequently, or if the source is third-party. Recommend verifying with the vendor for procurement decisions.

**Self-host option:** Always note whether a self-hosted path exists, what it costs (usually just infrastructure + LLM API fees), and what enterprise features you sacrifice by not using the commercial platform.

### Section 11 — Summary and Verdict

Write three components:

1. **One-sentence positioning statement** that captures the framework's core tradeoff clearly and memorably (e.g., "CrewAI trades fine-grained control for intuitive ergonomics, and production precision for prototyping speed.")

2. **When to choose this framework** — a bullet list of 4–6 specific, concrete conditions that describe the ideal use case. Avoid vague adjectives. Bad: "when you need flexibility." Good: "when your workflow requires human-in-the-loop approval gates at specific decision points."

3. **When not to choose this framework** — same format, 3–5 conditions.

4. **A closing paragraph** that places the framework in context of the overall ecosystem — what tier does it occupy, who is it competing with most directly, and what is the trajectory?

### Sources Section

List all sources as markdown links. Minimum 15 sources. Organize roughly by type: official docs first, then official blog posts, then third-party analyses, then community content. Format:

```markdown
- [Descriptive Title — Publisher](URL)
```

---

## Part 3: Research Methodology

### Step 1: Task Tracking

Create a task list before starting any research. Minimum tasks:
1. Research [Framework] fundamentals and architecture
2. Research [Framework] users, industries, and adoption
3. Research [Framework] competitive position and criticisms
4. Research [Framework] pricing
5. Write [Framework]_Research.md
6. Verify and review the document

Mark each task `in_progress` when you start it and `completed` when done.

### Step 2: Research Batches

Run searches in **parallel batches** (multiple simultaneous queries) to minimize time. Plan 3–4 batches:

**Batch 1 — Fundamentals (run in parallel):**
- `"[Framework] agent framework what is it how does it work architecture [current year]"`
- `"[Framework] [core concept] [core concept] [core concept] explained deep dive"`
- `"[Framework] who uses it enterprise customers industries adoption case studies [current year]"`

**Batch 2 — Comparisons and criticisms (run in parallel):**
- `"[Framework] vs [Competitor1] vs [Competitor2] comparison pros cons [current year]"`
- `"[Framework] criticisms weaknesses disadvantages limitations why not use it"`
- `"[Framework] real world production examples case studies [current year]"`

**Batch 3 — Pricing and ecosystem (run in parallel):**
- `"[Framework] pricing tiers [current year] exact cost free enterprise [plan names]"`
- `"[Framework] platform cloud deployment features ecosystem [current year]"`
- `"[Framework] GitHub stars downloads community popularity metrics [current year]"`

**Batch 4 — Fill gaps (run after reviewing batch 1–3 results):**
- Target any specific subsections where the first three batches left gaps
- Examples: specific integrations, specific case study details, specific pricing numbers, founding story details

### Step 3: Pricing Research

Pricing is the hardest section to get right. Use this sequence:

1. **Try fetching the official pricing page directly** using a web fetch tool. If successful, use those exact numbers and note the date.
2. **If direct fetch fails, run two targeted searches:**
   - `"[Framework] pricing page [plan names] [current year] exact cost"`
   - `"[Framework] '[price point]' OR '[price point]' plan executions seats features [current year]"`
3. **Cross-reference at least 2–3 sources.** Pricing aggregator sites (costbench.com, checkthat.ai, zenml.io/blog) often have more detail than the vendor's own marketing pages.
4. **Always note discrepancies.** If two sources disagree on a price, note the range and recommend verifying with the vendor.
5. **Check for login-gated pricing** — some vendors hide pricing behind account creation. Note this explicitly in the section.

### Step 4: Before Writing

Before opening any file to write, verify you have answers to all of the following:
- [ ] What is the framework's core metaphor/abstraction?
- [ ] What are its 3–5 core primitives?
- [ ] Can you write a minimal code example from memory?
- [ ] Who are 8+ named enterprise users?
- [ ] What are 3+ documented case studies with measurable outcomes?
- [ ] What are the top 3 reasons practitioners choose it?
- [ ] What are the top 3 reasons practitioners avoid it?
- [ ] What does the free tier include?
- [ ] What does the first paid tier cost and include?
- [ ] What does the enterprise tier require?
- [ ] What are its 3 closest competitors, and what is the deciding factor for each comparison?
- [ ] What are the framework's GitHub star count and monthly download count?

If you are missing answers to more than 3 of these, run another search batch before writing.

---

## Part 4: Style and Formatting Rules

### Tone

Write like a senior engineer who has used the framework and is briefing a colleague who hasn't. Be direct, specific, and honest. Do not write like a vendor press release. Do not write like an academic survey paper.

- **Good:** "CrewAI's debugging experience for multi-agent loops is genuinely painful — normal Python logging doesn't work inside Task, and tracing agent delegation chains requires third-party tooling."
- **Bad:** "CrewAI may present some challenges in certain debugging scenarios for some users."

Name things. Quantify things. Take positions. The summary and verdict should have a clear recommendation, not a hedge.

### Markdown Formatting

- Use `##` for the 11 main section headers only
- Use `###` for subsections within a section
- Use `**bold**` for named key concepts the first time they appear, and for table header cells
- Use `code blocks` for: code examples, CLI commands, class names, method names, parameter names
- Use tables for: pricing tiers, framework comparisons, customer lists — anything with 3+ attributes across multiple items
- Use bullet lists for: feature lists, "when to choose" conditions, sources — not for prose reasoning
- Never use bullet points as a substitute for prose in body text of sections 1–7

### Metrics and Numbers

- Always include the unit (e.g., "34.5 million monthly downloads" not "34.5M")
- Always include the timeframe when citing metrics (e.g., "as of early 2026," "as of Q1 2026")
- When citing case study results, be specific (e.g., "saves 10+ hours/week per property manager" not "significant time savings")
- When pricing is approximate or from third-party sources, say so

### Code Examples

- Keep code examples to 10–20 lines
- Use the framework's actual import paths and class names
- Include a brief comment on the most non-obvious line
- Show the minimal path: define → connect → run
- Do not show production-ready code with error handling — keep it illustrative

### Tables

All tables must have:
- A header row with bolded column names
- At least 3 columns
- Consistent alignment (use `|---|` for left-align, `|---:|` for right-align numbers)
- No empty cells — use "N/A," "Custom," or "Unlimited" as appropriate

### The Sources Section

- Minimum 15 links
- No bare URLs — every link needs a descriptive title
- Format: `- [Descriptive Title — Publisher or Site](URL)`
- Order: official docs → official blog → third-party analyses → community content
- Include at least: the official docs page, the official GitHub repo, one pricing source, one competitor comparison source, and one case study

---

## Part 5: Quality Checklist

Before saving the final file, verify each item:

**Structure:**
- [ ] Title block includes research date and subject line
- [ ] Table of Contents links to all 11 sections
- [ ] All 11 sections are present and in order
- [ ] Sources section has 15+ linked entries
- [ ] File is saved as `[FrameworkName]_Research.md` in the workspace folder

**Content depth:**
- [ ] Section 2 includes a code example
- [ ] Section 4 has a table with 8+ named companies
- [ ] Section 8 covers at least 3 head-to-head comparisons
- [ ] Section 10 includes a pricing table, per-tier paragraphs, and 4 cost scenarios
- [ ] Section 11 includes both "right choice when" and "wrong choice when" conditions

**Accuracy:**
- [ ] All metrics have a timeframe attached
- [ ] Pricing figures note if they are sourced from third parties
- [ ] Deprecated or maintenance-mode frameworks are labeled as such in Section 8
- [ ] No section is copy-pasted from vendor marketing without paraphrase and critical context

**Style:**
- [ ] No section reads like a press release
- [ ] Section 7 has equally substantive criticism as Section 6 has praise
- [ ] Section 11 takes a clear position rather than hedging

---

## Part 6: Reference Examples

The following reports in this folder were produced using this methodology and serve as the canonical style reference:

- `LangGraph_Research.md` — example of a production-grade, graph-based framework with strong enterprise adoption and a distinct ecosystem (LangSmith/LangChain)
- `CrewAI_Research.md` — example of a role-based framework with a wide accessibility gap between OSS (free, unlimited) and the commercial platform

When in doubt about length, depth, or tone for any section, open one of these files and calibrate to match.

---

## Part 7: Frameworks Worth Researching

The following frameworks are candidates for future research reports using this guide. Suggested priority order based on relevance and production adoption as of 2026:

1. **AutoGen** (Microsoft) — conversational multi-agent, currently in maintenance mode; important for historical context and comparison baseline
2. **Google ADK (Agent Development Kit)** — Google's entry into the agentic framework space, growing rapidly with Gemini integration
3. **Mastra** — TypeScript-native agent framework, relevant for JS/TS-first engineering teams
4. **LlamaIndex** (Workflows) — data-centric agent framework, strong in RAG-heavy use cases
5. **OpenAI Swarm** — minimal handoff framework from OpenAI, primarily educational but widely referenced
6. **Semantic Kernel** (Microsoft) — enterprise-oriented, multi-language (C#, Python, Java), strong Azure integration
7. **Haystack** (deepset) — document and NLP pipeline framework with agentic capabilities, strong in search/retrieval
8. **Dify** — low-code/no-code agent builder platform, different audience than code-first frameworks
9. **n8n / Zapier AI** — workflow automation platforms adding AI agent capabilities, relevant for non-developer users
10. **Bee Agent Framework** (IBM Research) — newer entrant targeting enterprise observability and TypeScript use cases

---

*This guide was authored based on the methodology used to produce the LangGraph and CrewAI research reports in this folder. Update it if the methodology or style standards evolve.*
