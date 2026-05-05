---
name: html-slides
description: Use this skill whenever the user asks to make slides, create a presentation, build a slideshow, or generate HTML slides
---

# Skill Instructions

## When to trigger
Apply this skill whenever the user asks to:
- "make slides" / "create slides" / "build slides"
- "make a presentation" / "create a presentation"
- "make a slideshow" / "build a slideshow"
- "make a deck" / "create a deck"
- Generate any HTML-based slide content

## Output format
Always produce a single self-contained `.html` file. No external dependencies unless the user specifically asks for a framework like Reveal.js.

## Structure rules
1. The first slide is always a title slide: large centered title, subtitle, and optional author/date.
2. Every subsequent slide has a clear heading (`<h2>`) and concise bullet points or a single visual concept — never walls of text.
3. Add a slide counter in the bottom-right corner (e.g. "3 / 8").
4. Include a progress bar at the top that fills as the user advances.
5. The last slide is always a closing/summary slide.

## Navigation
- Use the keyboard navigation script from `references/slide-scripts.md` — supports arrow keys, space bar, and swipe on mobile.
- Show clickable Previous / Next buttons in the bottom corners as a fallback.

## Visual design
- Follow the design principles in `references/design-principles.md`.
- Default to the "Clean Dark" palette unless the user specifies a theme.
- Use a CSS variable block at the top of the `<style>` tag so themes are easy to swap.
- Maximum 6 bullet points per slide. If content is longer, split it across two slides.
- Keep font sizes large enough to be readable projected: body ≥ 22px, headings ≥ 36px.

## Animations
- Slides transition with a smooth fade or slide (CSS transition, ~300 ms) — never jarring flashes.
- Bullet points may animate in one-by-one if the user requests it, otherwise they appear all at once.

## Accessibility
- Every slide section must have `role="region"` and `aria-label="Slide N of M"`.
- Ensure color contrast ratio ≥ 4.5:1 for all text.

## Code slides
- If the user wants a code example on a slide, use `<pre><code>` with a dark monospace background block inside the slide.
- Syntax highlighting should use inline styles (no external CDN) or a small embedded highlight function.
