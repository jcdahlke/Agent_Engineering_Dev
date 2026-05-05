# HTML Slides Design Principles

## Core rules (always apply these)

1. **One idea per slide.** If you need two ideas, use two slides.
2. **No more than 6 bullet points per slide.** Prefer 3–4.
3. **Never use full sentences in bullets** — use fragments or short phrases. The speaker fills in the rest.
4. **Font sizes**: headings ≥ 36px, body ≥ 22px. Anything smaller is unreadable when projected.
5. **Whitespace is your friend.** Dense slides lose audiences. Embrace padding.

---

## Color palettes

### Clean Dark (default)
| Token     | Value     | Use                      |
|-----------|-----------|--------------------------|
| `--bg`    | `#0f1117` | Page background          |
| `--surface` | `#1a1d27` | Slide card             |
| `--accent` | `#4f8ef7` | Headings underline, bullets, buttons |
| `--text`  | `#e8eaf0` | Body text                |
| `--muted` | `#8b90a0` | Counter, captions        |
| `--heading` | `#ffffff` | Slide titles            |

### Clean Light
| Token     | Value     | Use                      |
|-----------|-----------|--------------------------|
| `--bg`    | `#f4f5f7` | Page background          |
| `--surface` | `#ffffff` | Slide card             |
| `--accent` | `#2563eb` | Accent                   |
| `--text`  | `#1e293b` | Body text                |
| `--muted` | `#64748b` | Counter, captions        |
| `--heading` | `#0f172a` | Slide titles            |

### BYU Brand
| Token     | Value     | Use                      |
|-----------|-----------|--------------------------|
| `--bg`    | `#00205b` | Navy background          |
| `--surface` | `#0a2d78` | Slide card             |
| `--accent` | `#ffffff` | White accent             |
| `--text`  | `#dce8ff` | Body text                |
| `--muted` | `#a0b4d8` | Counter, captions        |
| `--heading` | `#ffffff` | Slide titles            |

### Tech/Code (good for coding courses)
| Token     | Value     | Use                      |
|-----------|-----------|--------------------------|
| `--bg`    | `#0d1117` | GitHub-dark background   |
| `--surface` | `#161b22` | Slide card             |
| `--accent` | `#58a6ff` | Blue accent              |
| `--text`  | `#c9d1d9` | Body text                |
| `--muted` | `#6e7681` | Counter, captions        |
| `--heading` | `#f0f6fc` | Slide titles            |

---

## Typography guidance

- **Headings**: bold, sentence case (not ALL CAPS).
- **Body**: regular weight, good line-height (1.4–1.6).
- **Code**: always monospace, never serif.
- Avoid more than 2 type sizes on a single slide.

---

## Layout patterns

### Text-only slide
```
[ Heading                          ]
[ • Point 1                        ]
[ • Point 2                        ]
[ • Point 3                        ]
```

### Two-column slide (use CSS grid: `grid-template-columns: 1fr 1fr`)
```
[ Heading                          ]
[ • Left point 1  | Right content  ]
[ • Left point 2  | or image       ]
```

### Code example slide
```
[ Heading                          ]
[ Short intro sentence             ]
[ ┌──────────────────────────────┐ ]
[ │  code block                  │ ]
[ └──────────────────────────────┘ ]
```

### Big-stat slide (for emphasis)
```
[                                  ]
[          42%                     ]
[   of users abandon after         ]
[   3 seconds of load time         ]
[                                  ]
```
Use large font (4–6rem) for the stat, small text for the label.

---

## What to avoid

- **Avoid gradients on text** — hard to read when projected.
- **Avoid background images** unless contrast is guaranteed.
- **Avoid red/green only differentiation** — colorblind-unfriendly.
- **Avoid animations longer than 400ms** — feels sluggish.
- **Never put important content at the very bottom** — projector cutoff risk.
- **No Comic Sans, Papyrus, or decorative fonts** for content.

---

## Checklist before delivering slides

- [ ] Every slide has a heading (except full-bleed visual slides)
- [ ] Slide counter is visible
- [ ] Progress bar present
- [ ] Keyboard navigation works (test arrow keys + spacebar)
- [ ] Font size is ≥ 22px for body text
- [ ] Color contrast passes 4.5:1 ratio
- [ ] File is a single self-contained `.html` with no broken external links
- [ ] Title slide and closing slide are present
