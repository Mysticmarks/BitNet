const state = {
  theme: 'dark',
  accentSeed: 264,
};

function setTheme(mode, accentSeed) {
  state.theme = mode;
  state.accentSeed = accentSeed;
  document.body.dataset.theme = mode === 'system' ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : mode;
  document.documentElement.style.setProperty('--accent-hue', accentSeed);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

function bindTelemetry() {
  const tokensEl = document.querySelector('#tokens-chart');
  const queueEl = document.querySelector('#queue-chart');
  const tempEl = document.querySelector('#temperature-chart');
  const presetEl = document.querySelector('#preset-status .value');
  const modelEl = document.querySelector('#model-status .value');

  fetchJSON('/api/config').then(({ config }) => {
    modelEl.textContent = config.model || 'n/a';
    presetEl.textContent = config.preset || 'custom';
    syncForm(config);
  });

  const source = new EventSource('/api/metrics/stream');
  source.addEventListener('telemetry', (event) => {
    const payload = JSON.parse(event.data);
    tokensEl.textContent = payload.tokensPerSecond.toFixed(1);
    queueEl.textContent = payload.queueDepth;
    tempEl.textContent = payload.temperature.toFixed(2);
  });
}

function syncForm(config) {
  const form = document.querySelector('#config-form');
  for (const [key, value] of Object.entries(config)) {
    const field = form.elements.namedItem(key);
    if (field) {
      field.value = value ?? '';
    }
  }
}

function initForm() {
  const form = document.querySelector('#config-form');
  const feedback = document.querySelector('#config-feedback');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      if (value === '') {
        return;
      }
      const numberFields = ['n_predict', 'ctx_size', 'temperature', 'batch_size', 'gpu_layers'];
      payload[key] = numberFields.includes(key) ? Number(value) : value;
    });
    try {
      const result = await fetchJSON('/api/config', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      feedback.textContent = 'Configuration updated successfully';
      feedback.className = 'success';
      syncForm(result.config);
      document.querySelector('#preset-status .value').textContent = result.config.preset || 'custom';
      if (payload.model) {
        document.querySelector('#model-status .value').textContent = payload.model;
      }
    } catch (error) {
      feedback.textContent = error.message;
      feedback.className = 'error';
    }
  });

  window.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.key.toLowerCase() === 's') {
      event.preventDefault();
      form.requestSubmit();
    }
  });
}

function initThemeControls() {
  const modeSelect = document.querySelector('#theme-mode');
  const accentSlider = document.querySelector('#accent-seed');
  const applyButton = document.querySelector('#apply-theme');

  fetchJSON('/api/theme').then(({ theme }) => {
    modeSelect.value = theme.mode;
    accentSlider.value = theme.accentSeed;
    setTheme(theme.mode, theme.accentSeed);
  });

  applyButton.addEventListener('click', async () => {
    try {
      const accentSeed = Number(accentSlider.value);
      const mode = modeSelect.value;
      await fetchJSON('/api/theme', {
        method: 'POST',
        body: JSON.stringify({ mode, accent_seed: accentSeed }),
      });
      setTheme(mode, accentSeed);
    } catch (error) {
      console.error('Theme update failed', error);
    }
  });
}

function initKeyboardShortcuts() {
  const overlay = document.querySelector('#help-overlay');
  const presetSelect = document.querySelector('select[name="preset"]');
  window.addEventListener('keydown', (event) => {
    if (event.shiftKey && event.key.toLowerCase() === 'l') {
      event.preventDefault();
      const mode = state.theme === 'dark' ? 'light' : 'dark';
      document.querySelector('#theme-mode').value = mode;
      document.querySelector('#apply-theme').click();
    }
    if (event.shiftKey && event.key.toLowerCase() === 'p') {
      event.preventDefault();
      presetSelect.focus();
    }
    if (event.key === '?') {
      event.preventDefault();
      overlay.hidden = !overlay.hidden;
    }
    if (event.key === 'Escape' && !overlay.hidden) {
      overlay.hidden = true;
    }
  });
}

bindTelemetry();
initForm();
initThemeControls();
initKeyboardShortcuts();
