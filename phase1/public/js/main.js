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
  const toolSensitivity = document.getElementById('tool-sensitivity');
  const toolSensitivitySlider = document.getElementById('tool-sensitivity-slider');
  const toolSensitivityValue = document.getElementById('tool-sensitivity-value');
  const toolGesturesToggle = document.getElementById('tool-gestures-toggle');
  const gesturePanel = document.getElementById('gesture-panel');
  const gestureRows = document.getElementById('gesture-rows');
  const SENSITIVITY_DEFAULT = 5.0; // keep in sync with SENSITIVITY in src/gesture_mouse.py
  const GESTURE_ACTIONS = [
    { id: 'accel_up', label: 'Accelerate Up' },
    { id: 'accel_down', label: 'Accelerate Down' },
    { id: 'accel_left', label: 'Accelerate Left' },
    { id: 'accel_right', label: 'Accelerate Right' },
    { id: 'mute', label: 'Mute' },
    { id: 'unmute', label: 'Unmute' },
  ];

  let activeTool = null;
  let activeEventSource = null;
  let streamRetryTimer = null;
  let gesturePanelOpen = false;
  const gesturePollTimers = {}; // action id -> timer id, while a recording is in flight

  function gestureDot(actionId) {
    const row = gestureRows.querySelector(`[data-id="${actionId}"]`);
    return row ? row.querySelector('.gesture-dot') : null;
  }

  function renderGestureRows() {
    gestureRows.innerHTML = '';
    GESTURE_ACTIONS.forEach((a) => {
      const row = document.createElement('div');
      row.className = 'gesture-row';
      row.dataset.id = a.id;
      row.innerHTML = `
        <span class="gesture-dot" data-state="unset"></span>
        <span class="gesture-name">${a.label}</span>
        <button class="gesture-record" type="button">Record</button>
        <button class="gesture-clear" type="button">Clear</button>
      `;
      row.querySelector('.gesture-record').addEventListener('click', () => recordGesture(a.id));
      row.querySelector('.gesture-clear').addEventListener('click', () => clearGesture(a.id));
      gestureRows.appendChild(row);
    });
  }

  function fetchGestureStatus() {
    if (!activeTool) return;
    fetch(`/api/gesture/${activeTool.id}`)
      .then((r) => r.json())
      .then((data) => {
        const recorded = new Set(data.actions || []);
        GESTURE_ACTIONS.forEach((a) => {
          const dot = gestureDot(a.id);
          if (!dot || dot.dataset.state === 'pending') return; // don't clobber an in-flight recording
          dot.dataset.state = recorded.has(a.id) ? 'recorded' : 'unset';
        });
      })
      .catch(() => {});
  }

  function recordGesture(actionId) {
    if (!activeTool) return;
    const dot = gestureDot(actionId);
    if (dot) dot.dataset.state = 'pending';
    fetch(`/api/gesture/${activeTool.id}/record`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: actionId }),
    });
    // Recording itself takes ~1.8s server-side (prep + hold) -- poll a few times after
    // that until the file shows it saved; give up after a few tries (likely no hand
    // was in frame during capture) rather than polling forever.
    clearTimeout(gesturePollTimers[actionId]);
    let attempts = 0;
    const poll = () => {
      attempts += 1;
      fetch(`/api/gesture/${activeTool.id}`)
        .then((r) => r.json())
        .then((data) => {
          const recorded = new Set(data.actions || []);
          if (recorded.has(actionId)) {
            if (dot) dot.dataset.state = 'recorded';
          } else if (attempts < 6) {
            gesturePollTimers[actionId] = setTimeout(poll, 500);
          } else if (dot) {
            dot.dataset.state = 'unset';
          }
        });
    };
    gesturePollTimers[actionId] = setTimeout(poll, 2000);
  }

  function clearGesture(actionId) {
    if (!activeTool) return;
    fetch(`/api/gesture/${activeTool.id}/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: actionId }),
    }).finally(fetchGestureStatus);
  }

  toolGesturesToggle.addEventListener('click', () => {
    gesturePanelOpen = !gesturePanelOpen;
    gesturePanel.hidden = !gesturePanelOpen;
    toolGesturesToggle.classList.toggle('is-active', gesturePanelOpen);
    if (gesturePanelOpen) fetchGestureStatus();
  });

  renderGestureRows();

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
    toolSensitivity.hidden = true;
    toolGesturesToggle.hidden = true;
    gesturePanel.hidden = true;
    gesturePanelOpen = false;
    toolGesturesToggle.classList.remove('is-active');
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

    // Only gesture_mouse.py understands a "sensitivity" param -- a freshly
    // spawned process starts back at its own default, so reset the slider
    // to match instead of carrying over a stale value from last time.
    toolSensitivity.hidden = tool.id !== 'mouse';
    toolGesturesToggle.hidden = tool.id !== 'mouse';
    gesturePanel.hidden = true;
    gesturePanelOpen = false;
    toolGesturesToggle.classList.remove('is-active');
    if (tool.id === 'mouse') {
      toolSensitivitySlider.value = SENSITIVITY_DEFAULT;
      toolSensitivityValue.textContent = SENSITIVITY_DEFAULT.toFixed(1);
    }

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

  // Closing/reloading the tab doesn't run our JS long enough for a normal fetch,
  // and the launcher's child process is detached -- it survives the tab closing
  // and keeps holding the webcam. sendBeacon fires a best-effort request that
  // survives unload; the server doesn't care about a body for /api/stop.
  window.addEventListener('pagehide', () => {
    if (activeTool) navigator.sendBeacon(`/api/stop/${activeTool.id}`);
  });

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

  toolSensitivitySlider.addEventListener('input', () => {
    if (!activeTool || activeTool.id !== 'mouse') return;
    const value = toolSensitivitySlider.value;
    toolSensitivityValue.textContent = Number(value).toFixed(1);
    fetch('/api/param/mouse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'sensitivity', value }),
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
