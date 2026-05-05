# Slide Scripts

Copy these blocks verbatim into the generated HTML file.

---

## CSS Variable Block (paste inside `<style>`)

```css
/* ── Theme tokens ─────────────────────────────── */
:root {
  --bg:        #0f1117;
  --surface:   #1a1d27;
  --accent:    #4f8ef7;
  --text:      #e8eaf0;
  --muted:     #8b90a0;
  --heading:   #ffffff;
  --progress:  #4f8ef7;
  --radius:    12px;
  --slide-w:   900px;
  --slide-h:   540px;
  --font-body: 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'Cascadia Code', 'Fira Code', monospace;
}
```

---

## Core Layout CSS (paste inside `<style>`)

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 20px;
}

/* Progress bar */
#progress-bar {
  position: fixed;
  top: 0; left: 0;
  height: 4px;
  background: var(--progress);
  transition: width 0.3s ease;
  z-index: 100;
}

/* Slide wrapper */
#slideshow {
  position: relative;
  width: var(--slide-w);
  max-width: 100%;
}

/* Individual slide */
.slide {
  display: none;
  background: var(--surface);
  border-radius: var(--radius);
  padding: 52px 60px;
  min-height: var(--slide-h);
  flex-direction: column;
  justify-content: center;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5);
  animation: fadeIn 0.3s ease;
}
.slide.active { display: flex; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0);   }
}

/* Title slide */
.slide.title-slide { text-align: center; gap: 16px; }
.slide.title-slide h1 { font-size: 2.6rem; color: var(--heading); line-height: 1.2; }
.slide.title-slide .subtitle { font-size: 1.2rem; color: var(--accent); }
.slide.title-slide .meta { font-size: 0.95rem; color: var(--muted); margin-top: 8px; }

/* Content slides */
.slide h2 {
  font-size: 1.9rem;
  color: var(--heading);
  margin-bottom: 28px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--accent);
}
.slide ul { list-style: none; display: flex; flex-direction: column; gap: 14px; }
.slide ul li {
  font-size: 1.2rem;
  padding-left: 24px;
  position: relative;
  line-height: 1.5;
}
.slide ul li::before {
  content: '▸';
  position: absolute;
  left: 0;
  color: var(--accent);
}

/* Code blocks */
.slide pre {
  background: #0d0f18;
  border-radius: 8px;
  padding: 20px 24px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 0.95rem;
  line-height: 1.6;
  border-left: 3px solid var(--accent);
  margin-top: 16px;
}

/* Slide counter */
#counter {
  position: absolute;
  bottom: -32px;
  right: 0;
  font-size: 0.85rem;
  color: var(--muted);
}

/* Nav buttons */
#nav {
  display: flex;
  gap: 12px;
  margin-top: 48px;
}
#nav button {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--accent);
  border-radius: 8px;
  padding: 10px 28px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;
}
#nav button:hover { background: var(--accent); color: #fff; }
#nav button:disabled { opacity: 0.3; cursor: default; }
```

---

## Navigation JavaScript (paste before `</body>`)

```js
(function () {
  const slides = document.querySelectorAll('.slide');
  const total = slides.length;
  let current = 0;

  const progressBar = document.getElementById('progress-bar');
  const counter     = document.getElementById('counter');
  const prevBtn     = document.getElementById('prev');
  const nextBtn     = document.getElementById('next');

  function goTo(n) {
    slides[current].classList.remove('active');
    current = Math.max(0, Math.min(n, total - 1));
    slides[current].classList.add('active');
    slides[current].setAttribute('aria-label', `Slide ${current + 1} of ${total}`);
    progressBar.style.width = `${((current + 1) / total) * 100}%`;
    counter.textContent = `${current + 1} / ${total}`;
    prevBtn.disabled = current === 0;
    nextBtn.disabled = current === total - 1;
  }

  prevBtn.addEventListener('click', () => goTo(current - 1));
  nextBtn.addEventListener('click', () => goTo(current + 1));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ')  goTo(current + 1);
    if (e.key === 'ArrowLeft')                     goTo(current - 1);
  });

  // Touch / swipe support
  let touchStartX = 0;
  document.addEventListener('touchstart', (e) => { touchStartX = e.changedTouches[0].screenX; });
  document.addEventListener('touchend',   (e) => {
    const dx = e.changedTouches[0].screenX - touchStartX;
    if (dx < -50) goTo(current + 1);
    if (dx >  50) goTo(current - 1);
  });

  goTo(0); // initialize
})();
```

---

## Minimal HTML Shell

Use this as the skeleton when building slides from scratch:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Presentation Title</title>
  <style>
    /* PASTE: CSS Variable Block */
    /* PASTE: Core Layout CSS   */
  </style>
</head>
<body>
  <div id="progress-bar"></div>

  <div id="slideshow">
    <!-- TITLE SLIDE -->
    <section class="slide title-slide active" role="region" aria-label="Slide 1 of N">
      <h1>Presentation Title</h1>
      <p class="subtitle">Subtitle or tagline</p>
      <p class="meta">Author · Date</p>
    </section>

    <!-- CONTENT SLIDE TEMPLATE (repeat as needed) -->
    <section class="slide" role="region" aria-label="Slide 2 of N">
      <h2>Slide Heading</h2>
      <ul>
        <li>Point one</li>
        <li>Point two</li>
        <li>Point three</li>
      </ul>
    </section>

    <!-- CLOSING SLIDE -->
    <section class="slide title-slide" role="region" aria-label="Slide N of N">
      <h1>Thank You</h1>
      <p class="subtitle">Questions?</p>
    </section>

    <div id="counter"></div>
  </div>

  <div id="nav">
    <button id="prev">← Prev</button>
    <button id="next">Next →</button>
  </div>

  <script>
    /* PASTE: Navigation JavaScript */
  </script>
</body>
</html>
```
