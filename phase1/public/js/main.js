document.addEventListener('DOMContentLoaded', () => {
  const POWER4 = 'power4.out';

  // ---------- splash sequence ----------
  const splash = document.getElementById('splash');
  const progressFill = document.getElementById('progress-fill');
  gsap.to(progressFill, {
    width: '100%',
    duration: 3.5,
    ease: POWER4,
    onComplete: () => {
      gsap.to(splash, {
        autoAlpha: 0,
        duration: 0.8,
        ease: POWER4,
        onComplete: () => splash.remove(),
      });
    },
  });

  // ---------- menu toggle + staggered panel ----------
  const menuToggle = document.getElementById('menu-toggle');
  const panelRoot = document.getElementById('panel-root');
  const layers = panelRoot.querySelectorAll('.panel-layer');
  const navList = document.getElementById('nav-list');
  let isOpen = false;

  gsap.set(layers, { xPercent: 100 });

  function setToggleWord(state) {
    menuToggle.querySelectorAll('.menu-toggle-word').forEach((w) => {
      const active = w.dataset.state === state;
      gsap.to(w, { y: active ? 0 : 20, autoAlpha: active ? 1 : 0, duration: 0.4, ease: POWER4 });
    });
  }

  menuToggle.addEventListener('click', () => {
    isOpen = !isOpen;
    menuToggle.classList.toggle('is-open', isOpen);
    panelRoot.classList.toggle('is-open', isOpen);
    setToggleWord(isOpen ? 'Close' : 'Menu');

    if (isOpen) {
      gsap.to(layers, { xPercent: 0, duration: 0.6, ease: POWER4, stagger: 0.1 });
      gsap.fromTo(
        '.nav-item',
        { x: 40, rotation: 5, autoAlpha: 0 },
        { x: 0, rotation: 0, autoAlpha: 1, duration: 0.6, ease: POWER4, stagger: 0.08, delay: 0.25 },
      );
    } else {
      gsap.to(layers, { xPercent: 100, duration: 0.5, ease: POWER4, stagger: { each: 0.08, from: 'end' } });
    }
  });

  // ---------- tools + nav ----------
  const statusEl = document.getElementById('tool-status');
  const stageContainer = document.getElementById('stage-container');
  const ghostText = document.getElementById('ghost-text');
  const idleContent = document.getElementById('idle-content');
  const toolView = document.getElementById('tool-view');
  const toolStream = document.getElementById('tool-stream');
  const toolLog = document.getElementById('tool-log');
  const toolBarName = document.getElementById('tool-bar-name');
  const toolBarHint = document.getElementById('tool-bar-hint');
  const toolStopBtn = document.getElementById('tool-stop');

  let activeTool = null;
  let activeEventSource = null;
  let streamRetryTimer = null;

  function connectStream(tool) {
    clearTimeout(streamRetryTimer);
    toolStream.onerror = () => {
      // /api/launch returns as soon as the process is spawned, but the Python
      // side (camera open + model init) can take a second or two before its
      // StreamServer is actually listening -- the first request often lands
      // before that, so keep retrying until it does.
      streamRetryTimer = setTimeout(() => {
        if (activeTool === tool) connectStream(tool);
      }, 400);
    };
    toolStream.src = `http://localhost:${tool.port}/stream?t=${Date.now()}`;
  }

  function closeMenu() {
    if (!isOpen) return;
    isOpen = false;
    menuToggle.classList.remove('is-open');
    panelRoot.classList.remove('is-open');
    setToggleWord('Menu');
    gsap.to(layers, { xPercent: 100, duration: 0.5, ease: POWER4, stagger: { each: 0.08, from: 'end' } });
  }

  function showIdle() {
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    clearTimeout(streamRetryTimer);
    toolStream.onerror = null;
    activeTool = null;
    toolView.hidden = true;
    toolStream.hidden = true;
    toolStream.removeAttribute('src');
    toolLog.hidden = true;
    toolLog.textContent = '';
    ghostText.hidden = false;
    idleContent.hidden = false;
    statusEl.textContent = 'Idle';
  }

  function showTool(tool) {
    activeTool = tool;
    ghostText.hidden = true;
    idleContent.hidden = true;
    toolView.hidden = false;
    toolBarName.textContent = tool.name;
    toolBarHint.textContent = tool.keys || '';
    statusEl.textContent = `${tool.name} running`;

    if (tool.mode === 'stream') {
      toolLog.hidden = true;
      toolStream.hidden = false;
      connectStream(tool); // cache-busts and retries until the tool's StreamServer is actually listening
      stageContainer.focus();
    } else {
      toolStream.hidden = true;
      toolLog.hidden = false;
      toolLog.textContent = '';
      activeEventSource = new EventSource(`/api/logs/${tool.id}`);
      activeEventSource.onmessage = (e) => {
        toolLog.textContent += JSON.parse(e.data) + '\n';
        toolLog.scrollTop = toolLog.scrollHeight;
      };
    }
  }

  stageContainer.addEventListener('keydown', (e) => {
    if (!activeTool || activeTool.mode !== 'stream') return;
    const code = e.key.length === 1 ? e.key.charCodeAt(0) : null;
    if (code === null) return;
    fetch(`/api/key/${activeTool.id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
  });

  toolStopBtn.addEventListener('click', () => {
    if (!activeTool) return;
    fetch(`/api/stop/${activeTool.id}`, { method: 'POST' }).finally(showIdle);
  });

  function marqueeContent(name) {
    let html = '';
    for (let i = 0; i < 6; i++) {
      html += `<span>${name}</span><div class="marquee-img"></div>`;
    }
    return html;
  }

  function wireMarquee(item) {
    const overlay = item.querySelector('.marquee-overlay');
    const track = item.querySelector('.marquee-track');

    gsap.set(overlay, { yPercent: -100 });
    const distance = track.scrollWidth / 2;
    gsap.to(track, { x: -distance, duration: 10, ease: 'none', repeat: -1 });

    item.addEventListener('mouseenter', (e) => {
      const rect = item.getBoundingClientRect();
      const fromTop = e.clientY - rect.top < rect.height / 2;
      gsap.fromTo(overlay, { yPercent: fromTop ? -100 : 100 }, { yPercent: 0, duration: 0.5, ease: POWER4 });
    });
    item.addEventListener('mouseleave', (e) => {
      const rect = item.getBoundingClientRect();
      const towardTop = e.clientY - rect.top < rect.height / 2;
      gsap.to(overlay, { yPercent: towardTop ? -100 : 100, duration: 0.5, ease: POWER4 });
    });
  }

  function launchTool(tool) {
    statusEl.textContent = `Launching ${tool.name}...`;
    fetch(`/api/launch/${tool.id}`, { method: 'POST' })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          statusEl.textContent = `Error: ${data.error}`;
          return;
        }
        showTool(tool);
        closeMenu();
      })
      .catch((err) => {
        statusEl.textContent = `Error: ${err.message}`;
      });
  }

  function renderNav(tools) {
    navList.innerHTML = '';
    tools.forEach((tool) => {
      const item = document.createElement('div');
      item.className = 'nav-item';
      item.dataset.id = tool.id;
      item.innerHTML = `
        <span class="nav-index">${tool.index}</span>
        <span class="nav-name">${tool.name}</span>
        <span class="nav-tag">${tool.tag}</span>
        <div class="marquee-overlay"><div class="marquee-track">${marqueeContent(tool.name)}</div></div>
      `;
      item.addEventListener('click', () => launchTool(tool));
      navList.appendChild(item);
      wireMarquee(item);
    });
  }

  fetch('/api/tools')
    .then((r) => r.json())
    .then(renderNav)
    .catch(() => {
      navList.innerHTML = '<p style="font-family: var(--font-body); color:#900;">Could not load tool list. Is the server running?</p>';
    });
});
