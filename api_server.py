"""
API Server for Qwen-Agent Multi-Agent Framework

WebSocket + REST API that replaces the Gradio WebUI.
Any frontend (HTML/JS, Electron, etc.) can connect to control the agents.

WebSocket protocol (all JSON):
  Client → Server:
    {"type": "message", "text": "...", "agent_index": 0, "session_name": "..."}
    {"type": "stop"}
    {"type": "retry"}
    {"type": "reset"}
    {"type": "approve", "request_id": "..."}
    {"type": "reject", "request_id": "...", "reason": "..."}
    {"type": "edit_message", "index": N, "content": "new text"}
    {"type": "delete_messages", "indices": [N, M, ...]}
    {"type": "select_agent", "index": N}
    {"type": "set_session_name", "name": "..."}
    {"type": "inject", "text": "..."}

  Server → Client:
    {"type": "state",  ...full state snapshot...}
    {"type": "done",   ...final state snapshot...}
    {"type": "error",  "message": "..."}
    {"type": "approvals", "approvals": [...]}
"""

import asyncio
import copy
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from qwen_agent.llm.schema import (
    ASSISTANT, CONTENT, FUNCTION, NAME, REASONING_CONTENT,
    ROLE, SYSTEM, USER, Message,
)
from qwen_agent.log import logger

try:
    from qwen_agent.agents.user_agent import PENDING_USER_INPUT
except ImportError:
    PENDING_USER_INPUT = 'PENDING_USER_INPUT'


def _parse_multimodal_content(text):
    """
    Parse markdown images ![alt](data:...) and return a list of content items.
    If no images are found, returns the original text.
    """
    pattern = r'!\[([^\]]*)\]\((data:image/[^;]+;base64,[a-zA-Z0-9+/=]+)\)'
    parts = []
    last_end = 0
    for match in re.finditer(pattern, text):
        start, end = match.span()
        if start > last_end:
            parts.append({'text': text[last_end:start]})
        alt, url = match.groups()
        parts.append({'image': url})
        last_end = end
    
    if last_end < len(text):
        parts.append({'text': text[last_end:]})
    
    if not parts:
        return text
    if len(parts) == 1 and 'text' in parts[0]:
        return parts[0]['text']
    return parts


# ─── Message serialization ────────────────────────────────────────────────────

def serialize_message(msg, index=None):
    """Convert a Message object or dict to a JSON-serializable dict."""
    if hasattr(msg, 'model_dump'):
        d = msg.model_dump()
    elif isinstance(msg, dict):
        d = dict(msg)
    else:
        d = {}
        for k in ['role', 'content', 'name', 'function_call', 'reasoning_content']:
            val = getattr(msg, k, None)
            if val is not None:
                d[k] = val

    # Normalize content to string
    content = d.get('content', '')
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if 'text' in item:
                    parts.append(item['text'])
                elif 'image' in item:
                    parts.append(f"![image]({item['image']})")
                elif 'audio' in item:
                    parts.append(f"[Audio: {item['audio']}]")
                elif 'video' in item:
                    parts.append(f"[Video: {item['video']}]")
                elif 'file' in item:
                    parts.append(f"[File: {item['file']}]")
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, 'text') and item.text:
                parts.append(item.text)
            elif hasattr(item, 'image') and item.image:
                parts.append(f"![image]({item.image})")
        content = '\n'.join(parts)
    d['content'] = content or ''

    # Normalize function_call
    fc = d.get('function_call')
    if fc:
        if hasattr(fc, 'name'):
            d['function_call'] = {'name': fc.name, 'arguments': fc.arguments}
        # else: already a dict, keep it
    else:
        d.pop('function_call', None)

    # Strip None values and internal fields
    for key in list(d.keys()):
        if d[key] is None:
            del d[key]
    d.pop('extra', None)

    if index is not None:
        d['index'] = index

    return d


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app(agents, agent_pool, config=None):
    """
    Create the FastAPI application.

    Args:
        agents:     List of Agent objects (orchestrator first, then sub-agents)
        agent_pool: The AgentPool instance
        config:     Optional chatbot config dict
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    config = config or {}
    app = FastAPI(title="Qwen-Agent API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Shared session state ──────────────────────────────────────────────
    session: Dict[str, Any] = {
        'history': [],
        'agent_index': 0,
        'session_name': config.get('session_name', 'Maine'),
        'generating': False,
        'stop_requested': False,
        'generation_id': 0,         # Increment on each run to prevent stale appends
    }

    ws_connections: Set[WebSocket] = set()
    send_queue: asyncio.Queue = asyncio.Queue()

    # ── Helpers ───────────────────────────────────────────────────────────

    def get_agent():
        idx = session['agent_index']
        if 0 <= idx < len(agents):
            return agents[idx]
        return agents[0]

    def get_sub_agent_state():
        result = {}
        if agent_pool and hasattr(agent_pool, 'sub_agent_state'):
            for name, state in agent_pool.sub_agent_state.items():
                msgs = state.get('messages', [])
                result[name] = {
                    'active': state.get('active', False),
                    'agent_name': state.get('agent_name', name),
                    'messages': [serialize_message(m, i) for i, m in enumerate(msgs)],
                }
        return result

    def get_active_stack():
        if agent_pool and hasattr(agent_pool, 'active_stack'):
            return list(agent_pool.active_stack)
        return []

    def get_approvals():
        if agent_pool and hasattr(agent_pool, 'operation_manager'):
            return agent_pool.operation_manager.list_pending_approvals()
        return []

    def build_state(responses=None, generating=None):
        """Build a full state snapshot for the frontend."""
        msgs = list(session['history'])
        if responses:
            msgs.extend(responses)

        return {
            'messages': [serialize_message(m, i) for i, m in enumerate(msgs)],
            'sub_agents': get_sub_agent_state(),
            'active_stack': get_active_stack(),
            'approvals': get_approvals(),
            'generating': generating if generating is not None else session['generating'],
            'session_name': session['session_name'],
            'agent_index': session['agent_index'],
            'agents': [
                {'name': getattr(a, 'name', f'Agent-{i}'), 'index': i,
                 'description': getattr(a, 'description', ''),
                 'tools': list(a.function_map.keys()) if hasattr(a, 'function_map') else [],
                 'default_tools': getattr(a, 'default_tools', list(a.function_map.keys()) if hasattr(a, 'function_map') else [])}
                for i, a in enumerate(agents)
            ],
        }
        if generating:
            orch_tools = st['agents'][0]['tools'] if st['agents'] else []
            print(f"[DEBUG] build_state: orchestrator tools count={len(orch_tools)}")
        return st

    async def broadcast(data):
        """Send JSON to all connected WebSocket clients."""
        nonlocal ws_connections
        text = json.dumps(data, ensure_ascii=False, default=str)
        dead = set()
        for conn in ws_connections:
            try:
                await conn.send_text(text)
            except Exception:
                dead.add(conn)
        if dead:
            ws_connections = ws_connections - dead

    # ── Agent execution thread ────────────────────────────────────────────

    def run_agent_thread(history_for_agent, agent_runner, gen_id, loop):
        """
        Runs agent.run() in a background thread.
        Pushes state updates onto the async send_queue.
        """
        try:
            responses = []
            last_send = 0
            session['generating'] = True

            # Reset pool state
            if agent_pool:
                agent_pool.stopped = False
                if hasattr(agent_pool, 'active_stack'):
                    agent_pool.active_stack.clear()

            if hasattr(agent_runner, 'session_name'):
                agent_runner.session_name = session['session_name']

            # Inject ui sampling params securely
            ui_cfg = copy.deepcopy(session.get('generate_cfg', {}))
            if 'repeat_penalty' in ui_cfg:
                pen = ui_cfg.pop('repeat_penalty')
                ui_cfg['repetition_penalty'] = pen
                ui_cfg['repeat_penalty'] = pen
                ui_cfg['repeatPenalty'] = pen

            # Strip non-sampling params before updating LLM config
            mcp_servers = ui_cfg.pop('mcpServers', None)
            disabled_tools = ui_cfg.pop('disabled_tools', None)

            has_llm = hasattr(agent_runner, 'llm') and agent_runner.llm
            if has_llm:
                old_cfg = copy.deepcopy(agent_runner.llm.generate_cfg)
                # Clear potentially stale non-sampling params from persistent agent
                agent_runner.llm.generate_cfg.pop('mcpServers', None)
                agent_runner.llm.generate_cfg.pop('disabled_tools', None)
                agent_runner.llm.generate_cfg.pop('max_turns', None)
                agent_runner.llm.generate_cfg.pop('auto_continue', None)
                agent_runner.llm.generate_cfg.pop('read_file_limit', None)
                
                # Separate LLM params from Agent settings to avoid OpenAI API errors
                pure_llm_cfg = copy.deepcopy(ui_cfg)
                agent_max_turns = pure_llm_cfg.pop('max_turns', None)
                agent_auto_continue = pure_llm_cfg.pop('auto_continue', None)
                read_file_limit = pure_llm_cfg.pop('read_file_limit', None)

                agent_runner.llm.generate_cfg.update(pure_llm_cfg)
                if agent_pool:
                    agent_pool.llm_cfg.update(pure_llm_cfg)
                    if read_file_limit is not None:
                        agent_pool.llm_cfg['read_file_limit'] = read_file_limit
                
                # Attach agent-level settings to the agent instance
                if agent_max_turns is not None:
                    agent_runner.max_turns = agent_max_turns
                if agent_auto_continue is not None:
                    agent_runner.auto_continue_enabled = agent_auto_continue

            mcp_tools_added = []
            if mcp_servers:
                try:
                    from qwen_agent.tools.mcp_manager import MCPManager
                    mcp_tools = MCPManager().initConfig({'mcpServers': mcp_servers})
                    for tool in mcp_tools:
                        for agent_inst in agents:
                            if tool.name not in agent_inst.function_map:
                                agent_inst.function_map[tool.name] = tool
                        mcp_tools_added.append(tool.name)
                    print(f"[MCP] Successfully loaded {len(mcp_tools)} tools: {mcp_tools_added}")
                except Exception as e:
                    print(f"[MCP] Failed to initialize MCP tools: {e}")
                    traceback.print_exc()

            try:
                for partial in agent_runner.run(history_for_agent):
                    if session['stop_requested'] or session['generation_id'] != gen_id:
                        if agent_pool:
                            agent_pool.stopped = True
                        break

                    responses = partial
                    now = time.time()
                    if now - last_send > 0.08:  # ~12Hz throttle
                        state = build_state(responses, generating=True)
                        asyncio.run_coroutine_threadsafe(
                            send_queue.put({'type': 'state', **state}), loop
                        )
                        last_send = now
            finally:
                if has_llm:
                    agent_runner.llm.generate_cfg = old_cfg

            # ── Finalize: append responses to session history ──
            if session['generation_id'] != gen_id:
                return  # Session was reset, discard

            if responses:
                for r in responses:
                    c = r.get(CONTENT) if isinstance(r, dict) else getattr(r, 'content', '')
                    if c == PENDING_USER_INPUT:
                        continue
                    if isinstance(r, dict):
                        session['history'].append(r)
                    elif hasattr(r, 'model_dump'):
                        session['history'].append(r.model_dump())
                    else:
                        session['history'].append({
                            ROLE: str(getattr(r, 'role', '')),
                            CONTENT: str(getattr(r, 'content', '')),
                        })

            # Handle context compression (turn_final_messages)
            if hasattr(agent_runner, 'turn_final_messages') and agent_runner.turn_final_messages:
                tfm = agent_runner.turn_final_messages
                if len(tfm) < len(session['history']):
                    session['history'].clear()
                    for res in tfm:
                        msg = res.model_dump() if hasattr(res, 'model_dump') else (
                            res if isinstance(res, dict) else {}
                        )
                        if msg.get(ROLE) != SYSTEM:
                            session['history'].append(msg)
                agent_runner.turn_final_messages = None

            session['generating'] = False
            final = build_state(generating=False)
            asyncio.run_coroutine_threadsafe(
                send_queue.put({'type': 'done', **final}), loop
            )

        except Exception as e:
            traceback.print_exc()
            session['generating'] = False
            asyncio.run_coroutine_threadsafe(
                send_queue.put({'type': 'error', 'message': str(e)}), loop
            )

    # ── Background tasks ──────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(_sender_loop())
        asyncio.create_task(_approval_loop())

    async def _sender_loop():
        """Global loop: reads from send_queue → broadcasts to all clients."""
        while True:
            try:
                data = await send_queue.get()
                await broadcast(data)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _approval_loop():
        """Poll for pending approvals and push to clients."""
        known_ids: Set[str] = set()
        while True:
            try:
                await asyncio.sleep(0.3)
                pending = get_approvals()
                current_ids = {a['request_id'] for a in pending}
                if current_ids != known_ids:
                    known_ids = current_ids.copy()
                    await broadcast({'type': 'approvals', 'approvals': pending})
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    # ── REST endpoints ────────────────────────────────────────────────────

    @app.get("/api/agents")
    async def api_list_agents():
        return [
            {
                'name': getattr(a, 'name', f'Agent-{i}'),
                'index': i,
                'description': getattr(a, 'description', ''),
                'tools': list(a.function_map.keys()) if hasattr(a, 'function_map') else [],
            }
            for i, a in enumerate(agents)
        ]

    @app.get("/api/state")
    async def api_get_state():
        return build_state()

    @app.post("/api/reset")
    async def api_reset():
        session['history'] = []
        session['generating'] = False
        session['generation_id'] += 1
        if agent_pool:
            agent_pool.reset()
        await broadcast({'type': 'done', **build_state()})
        return {"status": "ok"}

    @app.post("/api/approve/{request_id}")
    async def api_approve(request_id: str):
        if agent_pool and hasattr(agent_pool, 'operation_manager'):
            result = agent_pool.operation_manager.user_approve(request_id)
            return {"status": "ok", "result": result}
        return {"status": "error", "message": "No operation manager"}

    @app.post("/api/reject/{request_id}")
    async def api_reject(request_id: str, reason: str = "Rejected by user"):
        if agent_pool and hasattr(agent_pool, 'operation_manager'):
            result = agent_pool.operation_manager.user_reject(request_id, reason)
            return {"status": "ok", "result": result}
        return {"status": "error", "message": "No operation manager"}

    @app.get("/api/sessions")
    async def api_list_sessions():
        from pathlib import Path
        log_dir = Path('workspace/logs')
        if not log_dir.exists():
            return {"sessions": []}
        
        sessions = []
        for p in log_dir.glob('*.jsonl'):
            try:
                # Basic info from filename: agent_class_instance_name_timestamp.jsonl
                parts = p.stem.split('_')
                if len(parts) >= 3:
                    agent_class = parts[0]
                    timestamp = parts[-2] + "_" + parts[-1]
                    instance_name = "_".join(parts[1:-2])
                else:
                    agent_class = "Unknown"
                    instance_name = p.stem
                    timestamp = "Unknown"
                
                sessions.append({
                    "path": str(p),
                    "name": instance_name,
                    "agent": agent_class,
                    "timestamp": timestamp,
                    "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime
                })
            except Exception:
                continue
        
        # Sort by mtime descending
        sessions.sort(key=lambda x: x['mtime'], reverse=True)
        return {"sessions": sessions}

    # ── WebSocket ─────────────────────────────────────────────────────────

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        await websocket.accept()
        ws_connections.add(websocket)

        # Send initial state
        try:
            init = {'type': 'state', **build_state()}
            await websocket.send_text(json.dumps(init, ensure_ascii=False, default=str))
        except Exception:
            ws_connections.discard(websocket)
            return

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get('type', '')

                # ── Send message / async inject ──
                if msg_type == 'message':
                    text = data.get('text', '').strip()
                    if not text:
                        continue

                    if session['generating']:
                        # Async injection while agent is running
                        if agent_pool:
                            agent_pool.async_message_queue.append(text)
                        continue

                    # Update session config if provided
                    if 'agent_index' in data:
                        session['agent_index'] = int(data['agent_index'])
                    if 'session_name' in data:
                        session['session_name'] = data['session_name']
                    if 'generate_cfg' in data:
                        session['generate_cfg'] = data['generate_cfg']

                    # Add user message to history (parsed for multimodal items)
                    parsed_content = _parse_multimodal_content(text)
                    session['history'].append({ROLE: USER, CONTENT: parsed_content})

                    # Start agent generation
                    session['generation_id'] += 1
                    gen_id = session['generation_id']
                    agent_runner = get_agent()
                    history_copy = copy.deepcopy(session['history'])
                    loop = asyncio.get_event_loop()

                    thread = threading.Thread(
                        target=run_agent_thread,
                        args=(history_copy, agent_runner, gen_id, loop),
                        daemon=True,
                    )
                    thread.start()

                    await broadcast({'type': 'state', **build_state(generating=True)})

                elif msg_type == 'stop':
                    session['stop_requested'] = True
                    if agent_pool:
                        agent_pool.stopped = True

                elif msg_type == 'retry':
                    if session['generating']:
                        continue
                    # Remove trailing assistant/function messages
                    while (session['history']
                           and session['history'][-1].get(ROLE) in (ASSISTANT, FUNCTION)):
                        session['history'].pop()

                    if not session['history']:
                        await broadcast({'type': 'state', **build_state()})
                        continue

                    if 'generate_cfg' in data:
                        session['generate_cfg'] = data['generate_cfg']

                    session['generation_id'] += 1
                    gen_id = session['generation_id']
                    agent_runner = get_agent()
                    history_copy = copy.deepcopy(session['history'])
                    loop = asyncio.get_event_loop()

                    thread = threading.Thread(
                        target=run_agent_thread,
                        args=(history_copy, agent_runner, gen_id, loop),
                        daemon=True,
                    )
                    thread.start()
                    await broadcast({'type': 'state', **build_state(generating=True)})

                elif msg_type == 'reset':
                    session['history'] = []
                    session['generating'] = False
                    session['stop_requested'] = False
                    session['generation_id'] += 1
                    if agent_pool:
                        agent_pool.stopped = True
                        agent_pool.reset()
                    await broadcast({'type': 'done', **build_state()})

                elif msg_type == 'update_config':
                    if 'generate_cfg' in data:
                        session['generate_cfg'] = data['generate_cfg']
                        ui_cfg = data['generate_cfg']
                        if 'mcpServers' in ui_cfg:
                            mcp_servers = ui_cfg['mcpServers']
                            try:
                                from qwen_agent.tools.mcp_manager import MCPManager
                                mcp_tools = MCPManager().initConfig({'mcpServers': mcp_servers})
                                for tool in mcp_tools:
                                    for agent_inst in agents:
                                        if tool.name not in agent_inst.function_map:
                                            agent_inst.function_map[tool.name] = tool
                                print(f"[MCP] Eagerly loaded {len(mcp_tools)} tools.")
                            except Exception as e:
                                print(f"[MCP] Eager initialization failed: {e}")
                    await broadcast({'type': 'state', **build_state()})

                elif msg_type == 'approve':
                    rid = data.get('request_id')
                    if rid and agent_pool:
                        agent_pool.operation_manager.user_approve(rid)

                elif msg_type == 'reject':
                    rid = data.get('request_id')
                    reason = data.get('reason', 'Rejected by user')
                    if rid and agent_pool:
                        agent_pool.operation_manager.user_reject(rid, reason)

                elif msg_type == 'edit_message':
                    idx = data.get('index')
                    content = data.get('content', '')
                    if (idx is not None
                            and not session['generating']
                            and 0 <= idx < len(session['history'])):
                        msg = session['history'][idx]
                        if isinstance(msg, dict):
                            msg[CONTENT] = _parse_multimodal_content(content)
                    await broadcast({'type': 'state', **build_state()})

                elif msg_type == 'delete_messages':
                    if session['generating']:
                        continue
                    indices = sorted(data.get('indices', []), reverse=True)
                    for idx in indices:
                        if 0 <= idx < len(session['history']):
                            session['history'].pop(idx)
                    await broadcast({'type': 'state', **build_state()})

                elif msg_type == 'select_agent':
                    session['agent_index'] = int(data.get('index', 0))
                    await broadcast({'type': 'state', **build_state()})

                elif msg_type == 'set_session_name':
                    session['session_name'] = data.get('name', 'Maine')

                elif msg_type == 'load_session':
                    path = data.get('path')
                    if path and agent_pool:
                        status = agent_pool.load_session_from_log(path, target_instance=session.get('session_name'))
                        if status.startswith("Error"):
                            await websocket.send_text(json.dumps({"type": "error", "message": status}, ensure_ascii=False))
                        else:
                            # Successfully loaded. Update history in session
                            instance_name = session.get('session_name', 'Maine')
                            if instance_name in agent_pool.instance_conversations:
                                session['history'] = copy.deepcopy(agent_pool.instance_conversations[instance_name])
                                await broadcast({'type': 'state', **build_state()})

                elif msg_type == 'inject':
                    text = data.get('text', '').strip()
                    if text and agent_pool:
                        agent_pool.async_message_queue.append(text)

        except WebSocketDisconnect:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            ws_connections.discard(websocket)

    # ── Serve frontend static files ───────────────────────────────────────
    web_ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_ui')

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(web_ui_dir, 'index.html'))

    @app.get("/{path:path}")
    async def serve_static(path: str):
        file_path = os.path.join(web_ui_dir, path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # SPA fallback
        return FileResponse(os.path.join(web_ui_dir, 'index.html'))

    return app
