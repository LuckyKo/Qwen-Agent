/**
 * Qwen-Agent Console — Frontend Application
 * 
 * Connects to the API server via WebSocket for real-time streaming.
 * Renders messages with markdown, handles editing, deletion, and approvals.
 */

// ── Markdown setup ───────────────────────────────────────────────────────────
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try { return hljs.highlight(code, { language: lang }).value; } catch { }
    }
    return hljs.highlightAuto(code).value;
  },
});

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  messages: [],
  subAgents: {},
  activeStack: [],
  approvals: [],
  generating: false,
  agents: [],
  agentIndex: 0,
  sessionName: 'Maine',
  connected: false,
  editingIndex: null,  // Which message index is being edited
  activeSubTab: null,
};

let ws = null;
let reconnectTimer = null;
let lastRenderedCount = -1;
let lastLastContent = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const messagesEl = $('#messages');
const chatInput = $('#chatInput');
const sendBtn = $('#sendBtn');
const stopBtn = $('#stopBtn');
const retryBtn = $('#retryBtn');
const resetBtn = $('#resetBtn');
const agentSelect = $('#agentSelect');
const sessionNameInput = $('#sessionName');
const statusText = $('#statusText');
const connectionDot = $('#connectionDot');
const approvalBar = $('#approvalBar');
const mainTabBar = $('#mainTabBar');
const mainTabChat = $('#mainTabChat');
const mainTabPanels = document.querySelector('.main-tab-panels');

// New CWrite-style DOM refs
const btnToggleSettings = $('#btn-toggle-settings');
const sidePanel = $('#side-panel');
const statusWords = $('#status-words');
const statusTokens = $('#status-tokens');
const statusTokensSec = $('#status-tokens-sec');
const statusGenInfo = $('#status-gen-info');
const settingFontSize = $('#setting-font-size');
const valFontSize = $('#val-font-size');
const settingLinesEnabled = $('#setting-lines-enabled');

const settingUserColor = $('#setting-user-color');
const settingAssistantColor = $('#setting-assistant-color');
const settingRawEditColor = $('#setting-raw-edit-color');
const settingTruncateTools = $('#setting-truncate-tools');

const settingVisionEnabled = $('#setting-vision-enabled');
const settingImageDetail = $('#setting-image-detail');
const settingMaxImageSize = $('#setting-max-image-size');
const insertImageBtn = $('#insertImageBtn');
const imageInput = $('#imageInput');

// Range outputs
const ranges = [
  { input: $('#setting-temperature'), output: $('#val-temperature') },
  { input: $('#setting-top-p'), output: $('#val-top-p') },
  { input: $('#setting-top-k'), output: $('#val-top-k') },
  { input: $('#setting-min-p'), output: $('#val-min-p') },
  { input: $('#setting-repeat-penalty'), output: $('#val-repeat-penalty') },
  { input: $('#setting-presence-penalty'), output: $('#val-presence-penalty') },
  { input: $('#setting-frequency-penalty'), output: $('#val-frequency-penalty') },
];

// ── Initialization ───────────────────────────────────────────────────────────

// Side panel toggle
if (btnToggleSettings && sidePanel) {
  btnToggleSettings.addEventListener('click', () => {
    sidePanel.classList.toggle('collapsed');
  });
}

// Sidebar toggle (Left)
const btnToggleSidebar = $('#btn-toggle-sidebar');
const appSidebar = $('#app-sidebar');
if (btnToggleSidebar && appSidebar) {
  btnToggleSidebar.addEventListener('click', () => {
    appSidebar.classList.toggle('collapsed');
  });
}

// Session Manager DOM refs
const refreshSessionsBtn = $('#refreshSessionsBtn');
const sessionSearch = $('#sessionSearch');
const sessionsList = $('#sessionsList');

// State for sessions
let sessions = [];

// Fetch sessions from API
async function fetchSessions() {
  try {
    if (sessionsList) sessionsList.innerHTML = '<div class="sessions-loading">Loading...</div>';
    const res = await fetch('/api/sessions');
    const data = await res.json();
    sessions = data.sessions || [];
    renderSessions();
  } catch (err) {
    console.error('Failed to fetch sessions:', err);
    if (sessionsList) sessionsList.innerHTML = '<div class="sessions-placeholder">Error loading sessions.</div>';
  }
}

// Initial fetch
fetchSessions();


// Render session list
function renderSessions() {
  if (!sessionsList) return;
  const query = sessionSearch ? sessionSearch.value.toLowerCase() : '';
  const filtered = sessions.filter(s => 
    s.name.toLowerCase().includes(query) || 
    s.agent.toLowerCase().includes(query)
  );

  if (filtered.length === 0) {
    sessionsList.innerHTML = `<div class="sessions-placeholder">${query ? 'No matching sessions.' : 'No sessions found.'}</div>`;
    return;
  }

  sessionsList.innerHTML = filtered.map(s => `
    <div class="session-item" data-path="${s.path.replace(/\\/g, '/')}">
      <div class="session-item-header">
        <span class="session-item-name">${s.name}</span>
        <span class="session-item-agent">${s.agent}</span>
      </div>
      <div class="session-item-meta">
        <span>${formatDate(s.mtime * 1000)}</span>
        <span>${formatSize(s.size)}</span>
      </div>
    </div>
  `).join('');

  // Add click listeners to session items
  document.querySelectorAll('.session-item').forEach(item => {
    item.addEventListener('click', () => {
      const path = item.dataset.path;
      loadSession(path);
    });
  });
}

function loadSession(path) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (confirm('Load this session? Current unsaved state will be lost.')) {
      ws.send(JSON.stringify({
        type: 'load_session',
        path: path
      }));
    }
  }
}

function formatDate(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Search and Refresh
if (sessionSearch) {
  sessionSearch.addEventListener('input', renderSessions);
}
if (refreshSessionsBtn) {
  refreshSessionsBtn.addEventListener('click', fetchSessions);
}

// Appearance settings
if (settingFontSize && valFontSize) {
  settingFontSize.addEventListener('input', (e) => {
    const val = e.target.value;
    valFontSize.textContent = val;
    document.documentElement.style.setProperty('--font-size-base', `${val}px`);
  });
}

if (settingLinesEnabled) {
  settingLinesEnabled.addEventListener('change', (e) => {
    const show = e.target.checked;
    document.documentElement.style.setProperty('--show-line-numbers', show ? 'block' : 'none');
  });
}

// Appearance colors
if (settingUserColor) {
  settingUserColor.addEventListener('input', (e) => {
    document.documentElement.style.setProperty('--user-bg', e.target.value);
  });
}
if (settingAssistantColor) {
  settingAssistantColor.addEventListener('input', (e) => {
    document.documentElement.style.setProperty('--assistant-bg', e.target.value);
  });
}
if (settingRawEditColor) {
  settingRawEditColor.addEventListener('input', (e) => {
    document.documentElement.style.setProperty('--raw-edit-bg', e.target.value);
  });
}

// Ranges
ranges.forEach(r => {
  if (r.input && r.output) {
    r.input.addEventListener('input', (e) => {
      let val = parseFloat(e.target.value);
      if (e.target.step && e.target.step.includes('.')) {
        const decimals = e.target.step.split('.')[1].length;
        r.output.textContent = val.toFixed(decimals);
      } else {
        r.output.textContent = val;
      }
    });
  }
});

// ── Settings Persistence ─────────────────────────────────────────────────────

function saveSettings() {
  const s = {};
  ranges.forEach(r => {
    if (r.input) s[r.input.id] = r.input.value;
  });
  if (settingLinesEnabled) s['setting-lines-enabled'] = settingLinesEnabled.checked;
  if (settingTruncateTools) s['setting-truncate-tools'] = settingTruncateTools.checked;
  if (settingUserColor) s['setting-user-color'] = settingUserColor.value;
  if (settingAssistantColor) s['setting-assistant-color'] = settingAssistantColor.value;
  if (settingRawEditColor) s['setting-raw-edit-color'] = settingRawEditColor.value;
  if (settingFontSize) s['setting-font-size'] = settingFontSize.value;
  
  if (settingVisionEnabled) s['setting-vision-enabled'] = settingVisionEnabled.checked;
  if (settingImageDetail) s['setting-image-detail'] = settingImageDetail.value;
  if (settingMaxImageSize) s['setting-max-image-size'] = settingMaxImageSize.value;

  localStorage.setItem('qwen-settings', JSON.stringify(s));
}

function loadSettings() {
  try {
    const raw = localStorage.getItem('qwen-settings');
    if (!raw) return;
    const s = JSON.parse(raw);

    ranges.forEach(r => {
      if (r.input && s[r.input.id] !== undefined) {
        r.input.value = s[r.input.id];
        r.input.dispatchEvent(new Event('input'));
      }
    });

    if (settingFontSize && s['setting-font-size'] !== undefined) {
      settingFontSize.value = s['setting-font-size'];
      settingFontSize.dispatchEvent(new Event('input'));
    }

    if (settingLinesEnabled && s['setting-lines-enabled'] !== undefined) {
      settingLinesEnabled.checked = s['setting-lines-enabled'];
      settingLinesEnabled.dispatchEvent(new Event('change'));
    }

    if (settingTruncateTools && s['setting-truncate-tools'] !== undefined) {
      settingTruncateTools.checked = s['setting-truncate-tools'];
    }

    if (settingUserColor && s['setting-user-color'] !== undefined) {
      settingUserColor.value = s['setting-user-color'];
      settingUserColor.dispatchEvent(new Event('input'));
    }
    if (settingAssistantColor && s['setting-assistant-color'] !== undefined) {
      settingAssistantColor.value = s['setting-assistant-color'];
      settingAssistantColor.dispatchEvent(new Event('input'));
    }
    if (settingRawEditColor && s['setting-raw-edit-color'] !== undefined) {
      settingRawEditColor.value = s['setting-raw-edit-color'];
      settingRawEditColor.dispatchEvent(new Event('input'));
    }
    
    if (settingVisionEnabled && s['setting-vision-enabled'] !== undefined) {
      settingVisionEnabled.checked = s['setting-vision-enabled'];
    }
    if (settingImageDetail && s['setting-image-detail'] !== undefined) {
      settingImageDetail.value = s['setting-image-detail'];
    }
    if (settingMaxImageSize && s['setting-max-image-size'] !== undefined) {
      settingMaxImageSize.value = s['setting-max-image-size'];
    }
  } catch (e) {
    console.error('Failed to load settings', e);
  }
}

// Auto-save settings on any change in the panel
if (sidePanel) {
  sidePanel.addEventListener('change', saveSettings);
  // Optional: save on input for color pickers to save while dragging
  sidePanel.addEventListener('input', saveSettings);
}

loadSettings();

// ── WebSocket ────────────────────────────────────────────────────────────────

function connect() {
  if (ws && ws.readyState <= 1) return;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/chat`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    state.connected = true;
    connectionDot.classList.add('connected');
    connectionDot.title = 'Connected';
    statusText.textContent = '';
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onclose = () => {
    state.connected = false;
    connectionDot.classList.remove('connected');
    connectionDot.title = 'Disconnected';
    statusText.textContent = 'Disconnected — reconnecting...';
    scheduleReconnect();
  };

  ws.onerror = () => {
    state.connected = false;
    connectionDot.classList.remove('connected');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleServerMessage(data);
    } catch { }
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 2000);
}

function send(obj) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify(obj));
  }
}

// ── Server message handlers ──────────────────────────────────────────────────

function handleServerMessage(data) {
  switch (data.type) {
    case 'state':
    case 'done':
      // Full state update
      state.messages = data.messages || [];
      state.subAgents = data.sub_agents || {};
      state.activeStack = data.active_stack || [];
      state.generating = data.generating ?? false;
      if (data.agents) {
        state.agents = data.agents;
        renderAgentSelect();
      }
      if (data.session_name) state.sessionName = data.session_name;
      if (data.agent_index !== undefined) state.agentIndex = data.agent_index;
      if (data.approvals) {
        state.approvals = data.approvals;
        renderApprovals();
      }
      renderMessages();
      renderSubAgents();
      updateControls();
      break;

    case 'approvals':
      state.approvals = data.approvals || [];
      renderApprovals();
      break;

    case 'error':
      state.generating = false;
      appendSystemBubble(`⚠️ Error: ${data.message}`);
      updateControls();
      break;
  }
}

// ── Rendering ────────────────────────────────────────────────────────────────

function renderMessages() {
  const msgs = state.messages;
  const container = messagesEl;

  // Calculate word count and token estimation for Status Bar
  if (statusWords || statusTokens) {
    const allText = msgs.map(m => (m.content || '') + (m.function_call ? JSON.stringify(m.function_call) : '') + (m.reasoning_content || '')).join(' ').trim();
    if (statusWords) {
      const words = allText ? allText.split(/\s+/).length : 0;
      statusWords.textContent = `${words} words`;
    }
    if (statusTokens) {
      statusTokens.textContent = `${estimateTokens(allText)} tokens`;
    }
  }

  // Quick check: if nothing meaningful changed, skip heavy re-render
  const currentCount = msgs.length;
  const lastMsg = msgs.length > 0 ? msgs[msgs.length - 1] : null;
  const lastContent = lastMsg ? (lastMsg.content || '') + (lastMsg.function_call ? JSON.stringify(lastMsg.function_call) : '') + (lastMsg.reasoning_content || '') : '';

  // Full re-render if count changed significantly or decreased
  if (currentCount < lastRenderedCount || currentCount === 0 || Math.abs(currentCount - lastRenderedCount) > 1) {
    fullRender(msgs, container);
    lastRenderedCount = currentCount;
    lastLastContent = lastContent;
    return;
  }

  // Append new messages
  if (currentCount > lastRenderedCount) {
    for (let i = lastRenderedCount; i < currentCount; i++) {
      container.appendChild(createMessageEl(msgs[i], i));
    }
    lastRenderedCount = currentCount;
  }

  // Update last message content (streaming)
  if (lastContent !== lastLastContent && container.lastElementChild) {
    const lastBubble = container.lastElementChild;
    const idx = parseInt(lastBubble.dataset.index);
    if (idx === currentCount - 1 && state.editingIndex !== idx) {
      updateBubbleContent(lastBubble, msgs[currentCount - 1]);
    }
  }
  lastLastContent = lastContent;

  // Auto-scroll
  scrollToBottom();
}

function fullRender(msgs, container) {
  container.innerHTML = '';
  for (let i = 0; i < msgs.length; i++) {
    if (msgs[i].role === 'system') continue; // Hide system messages
    container.appendChild(createMessageEl(msgs[i], i));
  }
  scrollToBottom();
}

function createMessageEl(msg, index) {
  const div = document.createElement('div');
  div.className = `message msg-${msg.role || 'unknown'}`;
  div.dataset.index = index;

  const isEditable = !msg.function_call && msg.role !== 'function' && msg.role !== 'system';

  // Header
  const header = document.createElement('div');
  header.className = 'msg-header';
  const nameSpan = document.createElement('span');
  nameSpan.className = 'msg-name';
  if (msg.role === 'user') {
    nameSpan.textContent = 'You';
  } else if (msg.role === 'function') {
    nameSpan.textContent = `✅ ${msg.name || 'Tool Result'}`;
  } else {
    nameSpan.textContent = msg.name || 'Assistant';
  }
  header.appendChild(nameSpan);

  // Actions
  const actions = document.createElement('div');
  actions.className = 'msg-actions';

  if (isEditable) {
    const editBtn = document.createElement('button');
    editBtn.className = 'msg-action-btn';
    editBtn.textContent = '✏️';
    editBtn.title = 'Edit message';
    editBtn.onclick = (e) => { e.stopPropagation(); startEdit(index); };
    actions.appendChild(editBtn);
  }

  const delBtn = document.createElement('button');
  delBtn.className = 'msg-action-btn msg-action-delete';
  delBtn.textContent = '🗑️';
  delBtn.title = 'Delete message';
  delBtn.onclick = (e) => { e.stopPropagation(); deleteMessage(index); };
  actions.appendChild(delBtn);

  header.appendChild(actions);

  div.appendChild(header);

  // Double click edit
  div.addEventListener('dblclick', (e) => {
    if (state.generating || !isEditable) return;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;

    let selectedText = sel.toString().trim();
    if (!selectedText) return;

    if (e.target.closest('.msg-header')) return;

    const contentDiv = div.querySelector('.msg-content');
    if (!contentDiv) return;

    const range = sel.getRangeAt(0);
    const preCaretRange = range.cloneRange();
    preCaretRange.selectNodeContents(contentDiv);
    preCaretRange.setEnd(range.startContainer, range.startOffset);
    const renderedOffset = preCaretRange.toString().length;
    const renderedLength = contentDiv.textContent.length;

    const proportion = renderedLength > 0 ? renderedOffset / renderedLength : 0;

    startEdit(index, selectedText, proportion);
  });

  // Content
  const contentDiv = document.createElement('div');
  contentDiv.className = 'msg-content';

  if (msg.function_call) {
    // Tool call bubble
    contentDiv.innerHTML = renderToolCall(msg);
  } else if (msg.role === 'function') {
    // Tool result bubble
    contentDiv.innerHTML = renderToolResult(msg);
  } else {
    // Regular text (user or assistant)
    const textContent = msg.content || '';
    let html = '';

    // Handle reasoning/thinking content
    const reasoning = msg.reasoning_content;
    if (reasoning) {
      html += renderThinkingBlock(reasoning, state.generating && index === state.messages.length - 1);
    }

    // Handle <think> tags in content
    const thinkMatch = textContent.match(/<think>([\s\S]*?)(<\/think>|$)/);
    if (thinkMatch) {
      const thought = thinkMatch[1];
      const isOpen = !textContent.includes('</think>');
      const before = textContent.substring(0, textContent.indexOf('<think>'));
      const after = textContent.includes('</think>')
        ? textContent.substring(textContent.indexOf('</think>') + 8)
        : '';
      if (before.trim()) html += renderMarkdown(before);
      html += renderThinkingBlock(thought, isOpen);
      if (after.trim()) html += renderMarkdown(after);
    } else {
      html += renderMarkdown(textContent);
    }

    contentDiv.innerHTML = html;
  }

  div.appendChild(contentDiv);
  return div;
}

function updateBubbleContent(bubble, msg) {
  const contentDiv = bubble.querySelector('.msg-content');
  if (!contentDiv) return;

  if (msg.function_call) {
    contentDiv.innerHTML = renderToolCall(msg);
  } else if (msg.role === 'function') {
    contentDiv.innerHTML = renderToolResult(msg);
  } else {
    let html = '';
    const reasoning = msg.reasoning_content;
    if (reasoning) {
      html += renderThinkingBlock(reasoning, state.generating);
    }
    const text = msg.content || '';
    const thinkMatch = text.match(/<think>([\s\S]*?)(<\/think>|$)/);
    if (thinkMatch) {
      const thought = thinkMatch[1];
      const isOpen = !text.includes('</think>');
      const before = text.substring(0, text.indexOf('<think>'));
      const after = text.includes('</think>') ? text.substring(text.indexOf('</think>') + 8) : '';
      if (before.trim()) html += renderMarkdown(before);
      html += renderThinkingBlock(thought, isOpen);
      if (after.trim()) html += renderMarkdown(after);
    } else {
      html += renderMarkdown(text);
    }
    contentDiv.innerHTML = html;
  }
}

function renderMarkdown(text) {
  if (!text || !text.trim()) return '';
  try {
    return marked.parse(text);
  } catch {
    return `<p>${escapeHtml(text)}</p>`;
  }
}

function renderToolCall(msg) {
  const fc = msg.function_call;
  let argsHtml;
  try {
    const parsed = JSON.parse(fc.arguments);
    argsHtml = escapeHtml(JSON.stringify(parsed, null, 2));
  } catch {
    argsHtml = escapeHtml(fc.arguments || '');
  }
  return `
    <details class="tool-call" open>
      <summary>🛠️ <strong>${escapeHtml(fc.name)}</strong></summary>
      <pre><code>${argsHtml}</code></pre>
    </details>
  `;
}

function renderToolResult(msg) {
  const content = msg.content || '';
  const shouldTruncate = settingTruncateTools ? settingTruncateTools.checked : true;
  const truncated = (shouldTruncate && content.length > 2000) ? content.substring(0, 2000) + '\n\n... (truncated)' : content;
  return `
    <details class="tool-result">
      <summary>📋 Result from <strong>${escapeHtml(msg.name || 'tool')}</strong>${shouldTruncate && content.length > 2000 ? ` <span class="truncation-hint">(${content.length.toLocaleString()} chars)</span>` : ''}</summary>
      <pre><code>${escapeHtml(truncated)}</code></pre>
    </details>
  `;
}

function renderThinkingBlock(thought, isOpen) {
  return `
    <details class="thinking-block" ${isOpen ? 'open' : ''}>
      <summary>💭 Thinking...</summary>
      <div class="thinking-content">${renderMarkdown(thought)}</div>
    </details>
  `;
}

function appendSystemBubble(text) {
  const div = document.createElement('div');
  div.className = 'message msg-system';
  div.innerHTML = `<div class="msg-content">${renderMarkdown(text)}</div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

// ── Message editing ──────────────────────────────────────────────────────────

let editClone = null;
function getEditClone(textarea) {
  if (!editClone) {
    editClone = document.createElement('div');
    editClone.className = 'edit-textarea message-edit-clone';
    document.body.appendChild(editClone);
  }
  const style = window.getComputedStyle(textarea);
  editClone.style.width = style.width;
  editClone.style.fontFamily = style.fontFamily;
  editClone.style.fontSize = style.fontSize;
  editClone.style.lineHeight = style.lineHeight;
  editClone.style.paddingLeft = style.paddingLeft;
  editClone.style.paddingRight = style.paddingRight;
  editClone.style.whiteSpace = 'pre-wrap';
  editClone.style.wordWrap = 'break-word';
  editClone.style.paddingTop = '0px';
  editClone.style.paddingBottom = '0px';
  editClone.style.minHeight = '0px';
  return editClone;
}

function startEdit(index, selectedText = '', proportion = 0) {
  const msg = state.messages[index];
  if (!msg || msg.function_call || msg.role === 'function' || state.generating) return;

  state.editingIndex = index;

  const bubble = messagesEl.querySelector(`.message[data-index="${index}"]`);
  if (!bubble) return;

  const contentDiv = bubble.querySelector('.msg-content');
  const originalContent = msg.content || '';

  contentDiv.innerHTML = '';
  contentDiv.classList.add('editing');

  const textarea = document.createElement('textarea');
  textarea.className = 'edit-textarea';
  textarea.value = originalContent;
  textarea.dataset.index = index;

  const container = document.createElement('div');
  container.className = 'message-edit-container';

  const gutter = document.createElement('div');
  gutter.className = 'line-numbers-gutter';

  container.appendChild(gutter);
  container.appendChild(textarea);

  const toolbar = document.createElement('div');
  toolbar.className = 'edit-toolbar';

  const saveBtn = document.createElement('button');
  saveBtn.className = 'btn btn-primary btn-sm';
  saveBtn.textContent = '✓ Save';
  saveBtn.onclick = () => finishEdit(index, textarea.value);

  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn btn-secondary btn-sm';
  cancelBtn.textContent = '✗ Cancel';
  cancelBtn.onclick = () => cancelEdit(index);

  toolbar.appendChild(saveBtn);
  toolbar.appendChild(cancelBtn);

  contentDiv.appendChild(container);
  contentDiv.appendChild(toolbar);

  const updateGutter = () => {
    const clone = getEditClone(textarea);
    const lines = textarea.value.split('\n');
    let html = '';
    for (let i = 0; i < lines.length; i++) {
      clone.textContent = lines[i] || ' ';
      const height = clone.getBoundingClientRect().height;
      html += `<div style="height: ${height}px">${i + 1}</div>`;
    }
    gutter.innerHTML = html;
  };

  const autoResize = () => {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
    updateGutter();
  };

  textarea.addEventListener('input', autoResize);

  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      finishEdit(index, textarea.value);
    } else if (e.key === 'Escape') {
      cancelEdit(index);
    }
  });

  // Calculate cursor
  let bestIdx = -1;
  if (selectedText) {
    const targetRawOffset = proportion * originalContent.length;
    let minDiff = Infinity;
    const safeText = selectedText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    let regex = new RegExp(`\\b${safeText}\\b`, 'gi');
    let wordMatchFound = false;
    let match;
    while ((match = regex.exec(originalContent)) !== null) {
      wordMatchFound = true;
      const diff = Math.abs(match.index - targetRawOffset);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = match.index;
      }
    }

    if (!wordMatchFound) {
      regex = new RegExp(safeText, 'gi');
      while ((match = regex.exec(originalContent)) !== null) {
        const diff = Math.abs(match.index - targetRawOffset);
        if (diff < minDiff) {
          minDiff = diff;
          bestIdx = match.index;
        }
      }
    }
  }

  requestAnimationFrame(() => {
    autoResize();
    textarea.focus();
    if (bestIdx !== -1) {
      textarea.setSelectionRange(bestIdx, bestIdx + selectedText.length);
    }
  });
}

function finishEdit(index, newContent) {
  if (state.editingIndex === index) state.editingIndex = null;
  state.messages[index].content = newContent; // Optimistic update
  send({ type: 'edit_message', index, content: newContent });

  // Localized re-render
  const bubble = messagesEl.querySelector(`.message[data-index="${index}"]`);
  if (!bubble) return;
  bubble.querySelector('.msg-content').classList.remove('editing');
  updateBubbleContent(bubble, state.messages[index]);
}

function cancelEdit(index) {
  if (state.editingIndex === index) state.editingIndex = null;

  // Localized re-render
  const bubble = messagesEl.querySelector(`.message[data-index="${index}"]`);
  if (!bubble) return;
  bubble.querySelector('.msg-content').classList.remove('editing');
  updateBubbleContent(bubble, state.messages[index]);
}

function deleteMessage(index) {
  const msg = state.messages[index];
  if (!msg) return;

  // If deleting an assistant message with a function_call, also delete the function result
  const indicesToDelete = [index];
  if (msg.function_call && index + 1 < state.messages.length && state.messages[index + 1].role === 'function') {
    indicesToDelete.push(index + 1);
  }
  // If deleting a function result, also delete the preceding function call
  if (msg.role === 'function' && index - 1 >= 0 && state.messages[index - 1].function_call) {
    indicesToDelete.push(index - 1);
  }

  send({ type: 'delete_messages', indices: [...new Set(indicesToDelete)] });
}

// ── Approvals ────────────────────────────────────────────────────────────────

function renderApprovals() {
  const bar = approvalBar;
  if (!state.approvals || state.approvals.length === 0) {
    bar.style.display = 'none';
    return;
  }

  bar.style.display = 'block';
  bar.innerHTML = '';

  for (const ap of state.approvals) {
    const card = document.createElement('div');
    card.className = 'approval-card';

    let argsHtml = '';
    try {
      argsHtml = escapeHtml(JSON.stringify(ap.tool_args, null, 2));
    } catch {
      argsHtml = escapeHtml(String(ap.tool_args));
    }

    card.innerHTML = `
      <div class="approval-header">
        <span class="approval-icon">🛡️</span>
        <strong>Approval Required</strong>
      </div>
      <div class="approval-meta">
        <span>Agent: <strong>${escapeHtml(ap.agent_name)}</strong></span>
        <span>Tool: <strong>${escapeHtml(ap.tool_name)}</strong></span>
      </div>
      <div class="approval-desc">${escapeHtml(ap.description)}</div>
      <details class="approval-args">
        <summary>Arguments</summary>
        <pre><code>${argsHtml}</code></pre>
      </details>
      <div class="approval-actions">
        <button class="btn btn-primary btn-sm" onclick="approveRequest('${ap.request_id}')">✅ Approve</button>
        <button class="btn btn-danger btn-sm" onclick="showRejectInput('${ap.request_id}', this)">❌ Reject</button>
      </div>
    `;
    bar.appendChild(card);
  }
}

// Global functions for inline onclick handlers
window.approveRequest = function (requestId) {
  send({ type: 'approve', request_id: requestId });
};

window.showRejectInput = function (requestId, btn) {
  const card = btn.closest('.approval-card');
  const existing = card.querySelector('.reject-input-area');
  if (existing) { existing.remove(); return; }

  const area = document.createElement('div');
  area.className = 'reject-input-area';
  area.innerHTML = `
    <input type="text" placeholder="Rejection reason..." class="reject-reason-input" id="reject-${requestId}">
    <button class="btn btn-danger btn-sm" onclick="rejectRequest('${requestId}')">Confirm Reject</button>
  `;
  card.appendChild(area);
  area.querySelector('input').focus();
};

window.rejectRequest = function (requestId) {
  const input = document.getElementById(`reject-${requestId}`);
  const reason = input ? input.value.trim() : 'Rejected by user';
  send({ type: 'reject', request_id: requestId, reason: reason || 'Rejected by user' });
};

// ── Sub-agents ───────────────────────────────────────────────────────────────

function renderSubAgents() {
  const sa = state.subAgents;
  const names = Object.keys(sa);

  // Remove stale sub-agent tabs and panels for agents that no longer exist
  mainTabBar.querySelectorAll('.main-tab[data-tab^="sub-"]').forEach(tab => {
    const agentName = tab.dataset.tab.substring(4);
    if (!names.includes(agentName)) {
      tab.remove();
      const panel = document.getElementById('panelSub-' + agentName);
      if (panel) panel.remove();
    }
  });

  if (names.length === 0) return;

  // Auto-select active tab from stack
  const activeTop = state.activeStack.length > 0 ? state.activeStack[state.activeStack.length - 1] : null;

  for (const name of names) {
    const tabId = 'sub-' + name;
    const isActive = sa[name].active;

    // Create tab button if it doesn't exist
    let tabBtn = mainTabBar.querySelector(`.main-tab[data-tab="${tabId}"]`);
    if (!tabBtn) {
      tabBtn = document.createElement('button');
      tabBtn.className = 'main-tab';
      tabBtn.dataset.tab = tabId;
      tabBtn.onclick = () => switchMainTab(tabId);
      mainTabBar.appendChild(tabBtn);
    }
    tabBtn.innerHTML = `${isActive ? '<span class="sub-tab-pulse"></span>' : '<span class="main-tab-icon">🤖</span>'} ${escapeHtml(name)}`;
    // Highlight the active sub-agent's tab
    if (isActive && activeTop === name) {
      tabBtn.classList.add('has-activity');
    } else {
      tabBtn.classList.remove('has-activity');
    }

    // Create or update panel
    let panel = document.getElementById('panelSub-' + name);
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'main-tab-panel sub-agent-panel';
      panel.id = 'panelSub-' + name;
      mainTabPanels.appendChild(panel);
    }

    // Render sub-agent messages into the panel
    renderSubAgentPanel(panel, sa[name]);
  }
}

function renderSubAgentPanel(panel, agentData) {
  const msgs = agentData.messages || [];
  // Only re-render if content changed (include reasoning_content in key)
  const lastMsg = msgs.length > 0 ? msgs[msgs.length - 1] : null;
  const contentKey = msgs.length + ':' + (lastMsg ? (lastMsg.content || '').length : 0) + ':' + (lastMsg ? (lastMsg.reasoning_content || '').length : 0);
  if (panel.dataset.contentKey === contentKey) return;
  panel.dataset.contentKey = contentKey;

  panel.innerHTML = '';

  const scrollContainer = document.createElement('div');
  scrollContainer.className = 'sub-agent-messages';

  for (const msg of msgs) {
    if (msg.role === 'system') continue;
    const div = document.createElement('div');
    div.className = `sub-msg sub-msg-${msg.role || 'unknown'}`;

    const label = document.createElement('div');
    label.className = 'sub-msg-label';
    label.textContent = msg.role === 'user' ? '📤 Task' :
      msg.role === 'function' ? `📋 ${msg.name || 'result'}` :
        msg.name || 'Agent';

    const content = document.createElement('div');
    content.className = 'sub-msg-content';

    if (msg.function_call) {
      content.innerHTML = renderToolCall(msg);
    } else if (msg.role === 'function') {
      content.innerHTML = renderToolResult(msg);
    } else {
      // Render with thinking/reasoning support (same as main chat)
      let html = '';
      const reasoning = msg.reasoning_content;
      if (reasoning) {
        html += renderThinkingBlock(reasoning, agentData.active && msg === msgs[msgs.length - 1]);
      }
      const textContent = msg.content || '';
      const thinkMatch = textContent.match(/<think>([\s\S]*?)(<\/think>|$)/);
      if (thinkMatch) {
        const thought = thinkMatch[1];
        const isOpen = !textContent.includes('</think>');
        const before = textContent.substring(0, textContent.indexOf('<think>'));
        const after = textContent.includes('</think>') ? textContent.substring(textContent.indexOf('</think>') + 8) : '';
        if (before.trim()) html += renderMarkdown(before);
        html += renderThinkingBlock(thought, isOpen);
        if (after.trim()) html += renderMarkdown(after);
      } else {
        html += renderMarkdown(textContent);
      }
      content.innerHTML = html;
    }

    div.appendChild(label);
    div.appendChild(content);
    scrollContainer.appendChild(div);
  }

  panel.appendChild(scrollContainer);

  // Auto-scroll
  scrollContainer.scrollTop = scrollContainer.scrollHeight;
}

function switchMainTab(tabId) {
  // Update tab buttons
  mainTabBar.querySelectorAll('.main-tab').forEach(t => t.classList.remove('active'));
  const activeTab = mainTabBar.querySelector(`.main-tab[data-tab="${tabId}"]`);
  if (activeTab) activeTab.classList.add('active');

  // Update panels
  mainTabPanels.querySelectorAll('.main-tab-panel').forEach(p => p.classList.remove('active'));
  if (tabId === 'chat') {
    document.getElementById('panelChat').classList.add('active');
  } else {
    const name = tabId.substring(4); // strip 'sub-'
    const panel = document.getElementById('panelSub-' + name);
    if (panel) panel.classList.add('active');
  }
}

// Wire up the static Chat tab
if (mainTabChat) {
  mainTabChat.addEventListener('click', () => switchMainTab('chat'));
}

// ── Agent selector ───────────────────────────────────────────────────────────

const settingAgentSelect = $('#setting-agent-select');
const settingToolsList = $('#setting-tools-list');

if (!localStorage.getItem('qwen-tools-migrated-v2')) {
  localStorage.removeItem('qwen-disabled-tools');
  localStorage.setItem('qwen-tools-migrated-v2', '1');
}

let agentDisabledTools = JSON.parse(localStorage.getItem('qwen-disabled-tools') || '{}');

function renderAgentSelect() {
  if (agentSelect) agentSelect.innerHTML = '';
  if (settingAgentSelect) settingAgentSelect.innerHTML = '';
  
  let updatedDisabledTools = false;
  
  for (const agent of state.agents) {
    if (!agentDisabledTools[agent.name] && agent.tools) {
      const defaultTools = agent.default_tools || agent.tools;
      agentDisabledTools[agent.name] = agent.tools.filter(t => !defaultTools.includes(t));
      updatedDisabledTools = true;
    }
    
    if (agentSelect) {
      const opt = document.createElement('option');
      opt.value = agent.index;
      opt.textContent = agent.name;
      if (agent.index === state.agentIndex) opt.selected = true;
      agentSelect.appendChild(opt);
    }
    if (settingAgentSelect) {
      const opt2 = document.createElement('option');
      opt2.value = agent.index;
      opt2.textContent = agent.name;
      if (agent.index === state.agentIndex) opt2.selected = true;
      settingAgentSelect.appendChild(opt2);
    }
  }
  
  if (updatedDisabledTools) {
    localStorage.setItem('qwen-disabled-tools', JSON.stringify(agentDisabledTools));
  }
  
  renderToolsForSelectedAgent();
}

function renderToolsForSelectedAgent() {
  if (!settingToolsList || !settingAgentSelect) return;
  const idx = parseInt(settingAgentSelect.value);
  if (isNaN(idx)) return;
  const agent = state.agents.find(a => a.index === idx);
  
  if (!agent || !agent.tools || agent.tools.length === 0) {
    settingToolsList.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">No tools available for this agent.</div>';
    return;
  }
  
  const disabled = agentDisabledTools[agent.name] || [];
  
  settingToolsList.innerHTML = agent.tools.map(toolName => `
    <label class="setting-field toggle-field">
      <span>${escapeHtml(toolName)}</span>
      <input type="checkbox" class="tool-toggle" data-agent="${escapeHtml(agent.name)}" data-tool="${escapeHtml(toolName)}" ${!disabled.includes(toolName) ? 'checked' : ''} />
    </label>
  `).join('');
  
  settingToolsList.querySelectorAll('.tool-toggle').forEach(chk => {
    chk.addEventListener('change', (e) => {
      const aName = e.target.dataset.agent;
      const tName = e.target.dataset.tool;
      if (!agentDisabledTools[aName]) agentDisabledTools[aName] = [];
      if (!e.target.checked) {
        if (!agentDisabledTools[aName].includes(tName)) agentDisabledTools[aName].push(tName);
      } else {
        agentDisabledTools[aName] = agentDisabledTools[aName].filter(t => t !== tName);
      }
      localStorage.setItem('qwen-disabled-tools', JSON.stringify(agentDisabledTools));
      saveSettings();
    });
  });
}

if (settingAgentSelect) {
  settingAgentSelect.addEventListener('change', renderToolsForSelectedAgent);
}

// Settings Tabs
document.querySelectorAll('.settings-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.settings-tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(btn.dataset.tab);
    if (panel) panel.classList.add('active');
  });
});

// ── Controls ─────────────────────────────────────────────────────────────────

function updateControls() {
  if (state.generating) {
    sendBtn.disabled = true;
    sendBtn.classList.add('loading');
    resetBtn.disabled = true;
    document.body.classList.add('is-generating');
  } else {
    sendBtn.disabled = false;
    sendBtn.classList.remove('loading');
    resetBtn.disabled = false;
    document.body.classList.remove('is-generating');
  }
  stopBtn.style.display = state.generating ? 'inline-flex' : 'none';
  sendBtn.disabled = !state.connected;
  retryBtn.disabled = state.generating || state.messages.length === 0;

  statusText.textContent = state.generating ? 'Generating...' : '';
}

// ── Auto-resize textarea ─────────────────────────────────────────────────────

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function estimateTokens(text) {
  let imageTokens = 0;
  
  // Strip out base64 image strings so they don't skew the text length
  const imageRegex = /!\[(.*?)\]\((data:image\/[^;]+;base64,[a-zA-Z0-9+/=]+)\)/g;
  const visionEnabled = settingVisionEnabled ? settingVisionEnabled.checked : false;
  
  const cleanedText = text.replace(imageRegex, (match, alt) => {
    // If vision is enabled, we'll estimate roughly 255 tokens per image 
    if (visionEnabled) {
      imageTokens += 255;
    }
    return `[Image: ${alt}]`;
  });

  // Prose estimation: modern tokenizer ratios (LLaMA/Mistral)
  // roughly average 1 token ≈ 4.86 characters
  return Math.ceil(cleanedText.length / 4.86) + imageTokens;
}

// ── Utilities ────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatMultimodalContent(text) {
  if (typeof text !== 'string') return text;
  const visionEnabled = settingVisionEnabled ? settingVisionEnabled.checked : false;
  if (visionEnabled) return text;

  // Strip image data, leave placeholder if vision is disabled
  const imageRegex = /!\[(.*?)\]\((data:image\/[^;]+;base64,[a-zA-Z0-9+/=]+)\)/g;
  return text.replace(imageRegex, "[Image: $1]");
}

// ── Image Handling ───────────────────────────────────────────────────────────

function insertImageMarkdown(base64Data, filename) {
  const markdown = `![${filename}](${base64Data})`;
  const startPos = chatInput.selectionStart;
  const endPos = chatInput.selectionEnd;
  const text = chatInput.value;
  chatInput.value = text.substring(0, startPos) + markdown + text.substring(endPos);
  chatInput.selectionStart = chatInput.selectionEnd = startPos + markdown.length;
  chatInput.focus();
  autoResize(chatInput);
}

function processImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  
  const maxSize = settingMaxImageSize ? parseInt(settingMaxImageSize.value) : 1024;
  const reader = new FileReader();
  
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      let width = img.width;
      let height = img.height;
      
      if (width > maxSize || height > maxSize) {
        if (width > height) {
          height = Math.round((height * maxSize) / width);
          width = maxSize;
        } else {
          width = Math.round((width * maxSize) / height);
          height = maxSize;
        }
      }
      
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);
      
      const mimeType = file.type === 'image/jpeg' ? 'image/jpeg' : 'image/png';
      const dataUrl = canvas.toDataURL(mimeType, 0.9);
      insertImageMarkdown(dataUrl, file.name || 'image');
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

if (insertImageBtn && imageInput) {
  insertImageBtn.addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processImageFile(e.target.files[0]);
    }
    e.target.value = ''; // Reset input
  });
}

chatInput.addEventListener('dragover', (e) => {
  e.preventDefault();
  chatInput.classList.add('drag-over');
});

chatInput.addEventListener('dragleave', () => {
  chatInput.classList.remove('drag-over');
});

chatInput.addEventListener('drop', (e) => {
  e.preventDefault();
  chatInput.classList.remove('drag-over');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    Array.from(e.dataTransfer.files).forEach(processImageFile);
  }
});

chatInput.addEventListener('paste', (e) => {
  if (e.clipboardData && e.clipboardData.items) {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        e.preventDefault();
        const file = items[i].getAsFile();
        processImageFile(file);
      }
    }
  }
});

// ── Event listeners ──────────────────────────────────────────────────────────

chatInput.addEventListener('input', () => autoResize(chatInput));

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', () => send({ type: 'stop' }));
retryBtn.addEventListener('click', () => {
  lastRenderedCount = -1;
  send({ type: 'retry' });
});
resetBtn.addEventListener('click', () => {
  if (confirm('Reset the entire conversation?')) {
    lastRenderedCount = -1;
    send({ type: 'reset' });
  }
});

agentSelect.addEventListener('change', () => {
  state.agentIndex = parseInt(agentSelect.value);
  send({ type: 'select_agent', index: state.agentIndex });
});

sessionNameInput.addEventListener('change', () => {
  state.sessionName = sessionNameInput.value.trim() || 'Maine';
  send({ type: 'set_session_name', name: state.sessionName });
});

function getGenerateCfg() {
  const cfg = {};
  if ($('#setting-endpoint') && $('#setting-endpoint').value.trim()) cfg.api_base = $('#setting-endpoint').value.trim();
  if ($('#setting-api-key') && $('#setting-api-key').value.trim()) cfg.api_key = $('#setting-api-key').value.trim();
  if ($('#setting-model') && $('#setting-model').value.trim()) cfg.model = $('#setting-model').value.trim();

  if ($('#setting-temperature')) cfg.temperature = parseFloat($('#setting-temperature').value);
  if ($('#setting-top-p')) cfg.top_p = parseFloat($('#setting-top-p').value);
  if ($('#setting-top-k')) cfg.top_k = parseInt($('#setting-top-k').value);
  if ($('#setting-min-p')) cfg.min_p = parseFloat($('#setting-min-p').value);
  if ($('#setting-repeat-penalty')) cfg.repeat_penalty = parseFloat($('#setting-repeat-penalty').value);
  if ($('#setting-presence-penalty')) cfg.presence_penalty = parseFloat($('#setting-presence-penalty').value);
  if ($('#setting-frequency-penalty')) cfg.frequency_penalty = parseFloat($('#setting-frequency-penalty').value);
  if ($('#setting-max-tokens')) cfg.max_tokens = parseInt($('#setting-max-tokens').value) || 2048;
  
  if (typeof agentDisabledTools !== 'undefined') {
    cfg.disabled_tools = agentDisabledTools;
  }
  return cfg;
}

function sendMessage() {
  const rawText = chatInput.value.trim();
  if (!rawText) return;
  
  const text = formatMultimodalContent(rawText);
  chatInput.value = '';
  autoResize(chatInput);

  send({
    type: 'message',
    text,
    agent_index: state.agentIndex,
    session_name: state.sessionName,
    generate_cfg: getGenerateCfg()
  });
}

function retryGeneration() {
  if (state.generating) return;
  send({
    type: 'retry',
    agent_index: state.agentIndex,
    session_name: state.sessionName,
    generate_cfg: getGenerateCfg()
  });
}

// ── Init ─────────────────────────────────────────────────────────────────────
connect();
