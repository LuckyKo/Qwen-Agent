"""
Agent Orchestrator — OrchestratorAgent and sub-agent streaming.

This module contains only the OrchestratorAgent class (the supervisor that
intercepts sub-agent tool calls as streaming generators) and its supporting
schemas / utilities.

Extracted modules:
- agent_pool.py      — AgentPool (agent lifecycle + conversation persistence)
- agent_factory.py   — Tool registration + agent loading
- agent_logger.py    — AgentInstanceLogger (JSONL session logs)

Backward-compatible re-exports are at the bottom of this file.
"""

import copy
import json
import datetime
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from qwen_agent.agents import Assistant
from qwen_agent.log import logger
from qwen_agent.llm.schema import (
    ASSISTANT, CONTENT, FUNCTION, IMAGE, ROLE, SYSTEM, USER, Message,
)
import qwen_agent.settings
qwen_agent.settings.MAX_LLM_CALL_PER_RUN = 100
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools.base import BaseTool
from qwen_agent.utils.utils import (
    extract_text_from_message,
    get_basename_from_url,
    json_loads,
)

from agent_pool import AgentPool

# ─── Sub-agent function schemas ────────────────────────────────────────────────
# These are NOT called via _call_tool; OrchestratorAgent._run intercepts them
# and handles them as streaming generators. They exist only so the LLM sees
# them in the function list.

CALL_AGENT_SCHEMA = {
    'name': 'call_agent',
    'description': (
        'Delegate a task to a specialized sub-agent. '
        'If the instance_name already exists, the session continues with the existing context. '
        'Otherwise, a new session is started using the specified agent_class.\n\n'
        'Example usage:\n'
        '{"name": "call_agent", "arguments": {"agent_class": "coder", "instance_name": "worker1", "task": "Write a script"}}'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'agent_class': {
                'type': 'string',
                'description': 'The class of agent to call (e.g. "coder", "researcher"). Only required when starting a NEW instance.'
            },
            'instance_name': {
                'type': 'string',
                'description': 'A unique name for this agent instance. If this name exists, the existing session is continued regardless of agent_class.'
            },
            'task': {
                'type': 'string',
                'description': 'The task or question to delegate'
            },
            'context': {
                'type': 'string',
                'description': 'Optional background context for the sub-agent'
            },
        },
        'required': ['agent_class', 'instance_name', 'task'],
    },
}

DISMISS_AGENT_SCHEMA = {
    'name': 'dismiss_agent',
    'description': (
        "End a sub-agent's current task and clear its conversation context. "
        "Use when you're done with a sub-agent instance and don't need its context anymore."
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'instance_name': {
                'type': 'string',
                'description': 'Name of the sub-agent instance to dismiss'
            },
        },
        'required': ['instance_name'],
    },
}


# ─── OrchestratorAgent ─────────────────────────────────────────────────────────

class _SubAgentFunctionProxy(BaseTool):
    """
    Placeholder tool so the LLM sees call_agent / continue_with_agent
    in the function list. These are never actually executed via _call_tool;
    OrchestratorAgent._run intercepts them first.
    """

    def __init__(self, schema: dict, **kwargs):
        self.name = schema['name']
        self.description = schema['description']
        self.parameters = schema['parameters']
        super().__init__(**kwargs)

    def call(self, params: str, **kwargs) -> str:
        # Should never be reached — intercepted in _run
        return 'ERROR: This tool should be intercepted by OrchestratorAgent._run'


class OrchestratorAgent(Assistant):
    """
    An orchestrator agent whose _run method intercepts sub-agent tool calls
    and executes them as streaming generators so the WebUI can display
    real-time sub-agent output.
    """

    STREAMING_TOOLS = {'call_agent', 'continue_with_agent'}

    def __init__(self, agent_pool: AgentPool, agent_type: str = 'Orchestrator', **kwargs):
        super().__init__(**kwargs)
        self.agent_pool = agent_pool
        self.agent_type = agent_type
        self.session_name: str = "Maine"
        from qwen_agent.utils.tokenization_qwen import count_tokens
        self._count_tokens = count_tokens

    def _count_message_tokens(self, msg: Union[Message, dict]) -> int:
        from qwen_agent.utils.tokenization_qwen import count_tokens as qwen_count
        
        # Consistent with base.py's `_count_tokens`
        if isinstance(msg, dict):
            role = msg.get('role', '')
            function_call = msg.get('function_call')
            # Important: if an assistant message has a function call, base.py counts ONLY the function call string.
            if role == ASSISTANT and function_call:
                return qwen_count(f'{function_call}')
            msg_obj = Message(**msg)
        else:
            if msg.role == ASSISTANT and msg.function_call:
                return qwen_count(f'{msg.function_call}')
            msg_obj = msg
            
        content = extract_text_from_message(msg_obj, add_upload_info=True)
        return qwen_count(content)

    def _get_history_tokens(self, messages: List[Message]) -> int:
        """Calculate total tokens in a message list."""
        total = 0
        for msg in messages:
            total += self._count_message_tokens(msg)
        return total

    def _get_max_tokens(self) -> int:
        """Resolve the effective max_input_tokens from LLM config.
        
        NOTE: We read from self.llm.cfg (the immutable original), NOT from
        self.llm.generate_cfg, because generate_cfg.pop('max_input_tokens')
        is called during the first chat() call, removing it permanently.
        """
        from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
        max_tokens = DEFAULT_MAX_INPUT_TOKENS  # 58000

        # 1. Try the LLM's original cfg (immutable source of truth)
        if hasattr(self, 'llm') and hasattr(self.llm, 'cfg'):
            cfg = self.llm.cfg
            # Check generate_cfg sub-dict first, then top-level
            agent_max = cfg.get('generate_cfg', {}).get('max_input_tokens') or cfg.get('max_input_tokens')
            if agent_max:
                max_tokens = int(agent_max)

        # 2. Fallback to pool-level config
        if max_tokens == DEFAULT_MAX_INPUT_TOKENS and hasattr(self, 'agent_pool') and self.agent_pool:
            llm_cfg = getattr(self.agent_pool, 'llm_cfg', {})
            pool_max = (
                llm_cfg.get('generate_cfg', {}).get('max_input_tokens')
                or llm_cfg.get('max_input_tokens')
            )
            if pool_max:
                max_tokens = int(pool_max)
        return max_tokens

    def _truncate_tool_result(
        self,
        tool_result: str,
        tool_name: str,
        messages: List[Message],
        instance_name: str,
    ) -> str:
        """Truncate a tool result if it would push context past 95% capacity.
        
        Token accounting mirrors base.py's _truncate_input_messages_roughly:
          available_tokens = max_input_tokens - system_message_tokens
          all_tokens = sum of non-system message tokens
        This ensures our percentage matches the 'ALL tokens / Available tokens'
        log in base.py.
        """
        if not isinstance(tool_result, str):
            return tool_result  # multimodal results pass through
            
        if tool_name in ['compress_context']:
            return tool_result

        max_tokens = self._get_max_tokens()

        # Mirror base.py token accounting: separate system vs non-system
        from qwen_agent.utils.tokenization_qwen import count_tokens
        system_tokens = 0
        non_system_tokens = 0
        for msg in messages:
            tokens = self._count_message_tokens(msg)
            role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'role', '')
            if role == SYSTEM:
                system_tokens += tokens
            else:
                non_system_tokens += tokens

        available_tokens = max_tokens - system_tokens
        if available_tokens <= 0:
            available_tokens = max_tokens  # fallback if system prompt is huge

        threshold = int(available_tokens * 0.95)

        # Inline image content (e.g. from screenshot): ![image/png](iVBOR...) — skip truncation since it's already compact markdown data, not prose the LLM parses as text tokens.
        if '![image/' in tool_result:
            return tool_result

        result_tokens = max(1, len(tool_result) // 3)
        
        if non_system_tokens + result_tokens <= threshold:
            return tool_result  # Fits fine, no truncation needed
        
        # --- Truncation required ---
        remaining_token_budget = max(200, threshold - non_system_tokens)
        # Convert back to chars (use 2.5 multiplier to be safe)
        char_budget = int(remaining_token_budget * 2.5)
        
        # Reserve space for the truncation notice itself (~300 chars)
        char_budget = max(100, char_budget - 300)
        
        # Save full result to spill file
        log_dir = Path('workspace/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_tool = tool_name.replace('/', '_').replace('\\', '_')
        safe_instance = instance_name.replace('/', '_').replace('\\', '_')
        spill_filename = f"{safe_instance}_{safe_tool}_{timestamp}.txt"
        spill_path = log_dir / spill_filename
        
        try:
            with open(spill_path, 'w', encoding='utf-8') as f:
                f.write(tool_result)
        except Exception as e:
            logger.error(f"Failed to write spill file {spill_path}: {e}")
            # Even if spill fails, still truncate to prevent context overflow
        
        truncated = tool_result[:char_budget]
        usage_pct = (non_system_tokens / available_tokens) * 100
        
        notice = (
            f"\n\n[TOOL RESPONSE TRUNCATED — Context at {usage_pct:.0f}% capacity "
            f"({non_system_tokens}/{available_tokens} tokens). "
            f"Full output ({len(tool_result)} chars) saved to: {spill_path}\n"
            f"You can read it with read_file if needed. "
            f"Consider compressing context before continuing.]"
        )
        
        logger.info(
            f"Truncated '{tool_name}' result for {instance_name}: "
            f"{len(tool_result)} chars -> {len(truncated)} chars. "
            f"Context: {non_system_tokens}/{available_tokens} tokens ({usage_pct:.0f}%). "
            f"Spill file: {spill_path}"
        )
        
        return truncated + notice

    def _inject_compression_warning(self, messages: List[Message]):
        """Legacy helper for the orchestrator itself."""
        self._inject_compression_warning_for_agent(self, self.session_name, messages)

    def _inject_compression_warning_for_agent(self, agent, instance_name: str, messages: List[Message]):
        """Inject a warning or force compression if context is getting full for any agent."""
        max_tokens = self._get_max_tokens()

        current_tokens = self._get_history_tokens(messages)
        usage_pct = (current_tokens / max_tokens) * 100
        
        # 1. Critical Threshold Check (> 95%)
        if usage_pct > 95.0:
            logger.info(f"Context usage at {usage_pct:.1f}% for {instance_name} - Triggering FORCEFUL compression.")
            compress_tool = agent.function_map.get('compress_context')
            if compress_tool:
                # Programmatically call the tool's internal compression logic
                
                # Call tool directly with 50% fraction
                params = json.dumps({
                    'fraction': 0.5, 
                    'justification': f'CRITICAL THRESHOLD REACHED ({usage_pct:.1f}%)'
                })
                
                # Pass necessary kwargs for direct application
                result = compress_tool.call(
                    params, 
                    messages=messages, 
                    agent_instance_name=instance_name, 
                    agent_obj=agent
                )
                
                # Check for failure
                is_error = isinstance(result, str) and result.startswith('ERROR')
                if is_error:
                    logger.error(f"Forceful compression failed for {instance_name}: {result}")
                    notification = (
                        f"\n\n[SYSTEM NOTIFICATION: Context window exceeded 95% capacity ({usage_pct:.1f}%), "
                        f"but automatic compression failed ({result}). The upcoming API call will likely fail due to length.]"
                    )
                else:
                    # Add a system message to inform the agent that it happened
                    agent_class = self.agent_pool.instance_classes.get(instance_name, 'Unknown')
                    logger_inst = self.agent_pool.get_logger(instance_name, agent_class)
                    notification = (
                        f"\n\n[SYSTEM NOTIFICATION: Context window exceeded 95% capacity ({usage_pct:.1f}%). "
                        "Forceful compression (50% ratio) has been effectuated to prevent errors. "
                        f"Use your logs at `{logger_inst.log_path}` if you need to restore details from turns that were removed.]"
                    )
                
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg.content, str):
                        # Prevent duplicate notifications from stacking if the loop repeats
                        if "[SYSTEM NOTIFICATION: Context window exceeded 95%" not in last_msg.content:
                            last_msg.content += notification
                    elif isinstance(last_msg.content, list):
                        from qwen_agent.llm.schema import ContentItem
                        
                        # Prevent duplicate notifications from stacking
                        has_notification = any(
                            isinstance(item, ContentItem) and "[SYSTEM NOTIFICATION: Context window exceeded 95%" in getattr(item, 'text', '')
                            for item in last_msg.content
                        )
                        if not has_notification:
                            last_msg.content.append(ContentItem(text=notification))
            return

        # 2. Warning Threshold (> 85%)
        if usage_pct > 85.0:
            warning = (
                f"\n\n[SYSTEM WARNING: Context window at {usage_pct:.1f}% capacity ({current_tokens}/{max_tokens} tokens). "
                "Consider using the `compress_context` tool to summarize old history and free up space. "
                "Propose a fraction (e.g. 0.2 for 20%) and a justification. The summary will be sent for approval.]"
            )
            # Find the most recent message to append warning (temporarily)
            if messages:
                # We append to the last message's content if it's text, or add a system message
                # Note: This is NOT saved to the permanent AgentPool history
                last_msg = messages[-1]
                if isinstance(last_msg.content, str):
                    last_msg.content += warning
                elif isinstance(last_msg.content, list):
                    from qwen_agent.llm.schema import ContentItem
                    last_msg.content.append(ContentItem(text=warning))


    @property
    def support_multimodal_input(self) -> bool:
        """
        Signal that the orchestrator itself can handle multimodal messages
        and should not have them stripped before reaching _run.
        """
        return True

    def _run(
        self,
        messages: List[Message],
        lang: str = 'en',
        knowledge: str = '',
        **kwargs,
    ) -> Iterator[List[Message]]:
        # Clear active stack only for the root turn to avoid state leakage
        if not kwargs.get('agent_instance_name'):
            self.agent_pool.active_stack.clear()
            
        instance = kwargs.get('agent_instance_name') or self.session_name
        self.session_name = instance # Track current instance name
        
        # Prepend knowledge like Assistant does
        messages = self._prepend_knowledge_prompt(
            messages=messages, lang=lang, knowledge=knowledge, **kwargs
        )
        
        # Using the base agent_type for the class metadata field keeps logs clean.
        logger_inst = self.agent_pool.get_logger(instance, self.agent_type)
        
        # Log the latest turn
        if messages:
            # Robust identity insertion into the system prompt regardless of line position
            m0 = messages[0]
            m0_role = m0.get('role') if isinstance(m0, dict) else getattr(m0, 'role', '')
            if m0_role == SYSTEM:
                m0_content = m0.get('content', '') if isinstance(m0, dict) else getattr(m0, 'content', '')
                if isinstance(m0_content, str) and instance:
                    import re
                    # 1. Update identity "You are [instance]."
                    pattern = rf"(?i)You are {self.agent_type}\."
                    if re.search(pattern, m0_content):
                        m0_content = re.sub(pattern, f"You are {instance}.", m0_content, count=1)
                    
                    # 2. Insert Session Metadata section (Stable)
                    if '## Session Metadata' not in m0_content:
                        meta_lines = [
                            "## Session Metadata",
                            f"- Supervisor: User",
                            f"- Log Path: {logger_inst.log_path}",
                            "Use your logs to recall details from turns that were compressed.\n"
                        ]
                        content_lines = m0_content.split('\n')
                        insert_pos = 2 if len(content_lines) > 1 and not content_lines[1].startswith("#") else 1
                        for i, ml in enumerate(meta_lines): content_lines.insert(insert_pos + i, ml)
                        m0_content = '\n'.join(content_lines)
                    
                    # 3. Inject available resources (Stable sort for caching)
                    if '--- CURRENT AVAILABLE RESOURCES' not in m0_content:
                        res_append = "\n\n--- CURRENT AVAILABLE RESOURCES (Auto-Injected) ---\n"
                        res_append += "\nAvailable Sub-Agents (call via call_agent):\n"
                        has_agents = False
                        for name in sorted(self.agent_pool.list_agents()):
                            if name.lower() != self.name.lower():
                                info = self.agent_pool.get_agent_info(name)
                                if info:
                                    res_append += f"- **{info['name']}**: {info['tagline']}\n"
                                    has_agents = True
                        if not has_agents: res_append += "- None currently available.\n"
                        
                        res_append += "\nEnabled Tools (can change per interaction):\n"
                        if self.function_map:
                            disabled_tools = self._get_disabled_tool_names()
                            for t_name in sorted(self.function_map.keys()):
                                if t_name in disabled_tools:
                                    continue
                                desc = getattr(self.function_map[t_name], 'description', 'No description provided')
                                res_append += f"- **{t_name}**: {desc}\n"
                        else: res_append += "- None currently enabled.\n"
                        m0_content += res_append
                    
                    # 4. Inject Argument Reuse instructions (Static version for caching)
                    if '### Advanced Feature: Argument Reuse' not in m0_content:
                        m0_content += (
                            "\n\n### Advanced Feature: Argument Reuse\n"
                            "To reuse a LARGE argument value (like full file content or path) from any previous successful tool call in this session, "
                            "use the exact placeholder: \"__USE_PREV_ARG__\". This saves tokens and processing time."
                        )
                    
                    if isinstance(m0, dict): m0['content'] = m0_content
                    else: m0.content = m0_content
        # Sync initial state if history is empty
        if not logger_inst.data["history"]:
            logger_inst.update_history(messages)
        elif not kwargs.get('agent_instance_name'):
            # Only log the last message for the main orchestrator session.
            # Sub-agents are already logged by _stream_sub_agent_call to avoid duplicates.
            logger_inst.log_message(messages[-1])

        # --- Check for manual commands ---
        last_msg = messages[-1] if messages else None
        last_role = last_msg.get('role') if isinstance(last_msg, dict) else getattr(last_msg, 'role', '')
        
        cmd_text = ""
        if last_msg:
            try:
                msg_obj = Message(**last_msg) if isinstance(last_msg, dict) else last_msg
                cmd_text = extract_text_from_message(msg_obj, add_upload_info=False).strip()
            except Exception:
                pass
        
        if last_role == USER and cmd_text:
            # --- /compress command ---
            if cmd_text.startswith('/compress'):
                parts = cmd_text.split()
                fraction = 0.5
                if len(parts) > 1:
                    try:
                        fraction = float(parts[1])
                    except ValueError:
                        pass
                
                compress_tool = self.function_map.get('compress_context')
                if compress_tool:
                    messages.pop() # Remove from history
                    
                    yield [Message(role=ASSISTANT, content=f"Generating context summary for {int(fraction*100)}% of history...")]
                    
                    params = json.dumps({
                        'fraction': fraction,
                        'justification': 'MANUAL USER COMMAND (Preview)'
                    })
                    
                    # Generate summary without applying
                    summary = compress_tool.call(
                        params,
                        messages=messages,
                        agent_instance_name=instance,
                        agent_obj=self,
                        dry_run=True  # Ensure it doesn't apply yet
                    )
                    
                    if summary and not summary.startswith("ERROR"):
                        description = f"Proposed Compression Summary ({int(fraction*100)}% of history)"
                        approved, reason = self.agent_pool.operation_manager.request_user_approval(
                            agent_name=instance,
                            tool_name='compress_context',
                            tool_args={'fraction': fraction, 'summary': summary},
                            description=description,
                            )
                        
                        if approved:
                            # Apply the compression
                            params = json.dumps({
                                'fraction': fraction,
                                'justification': 'MANUAL USER COMMAND (Approved)'
                            })
                            result = compress_tool.call(
                                params,
                                messages=messages,
                                agent_instance_name=instance,
                                agent_obj=self,
                                precomputed_summary=summary # Skip generation
                            )
                            yield [Message(role=ASSISTANT, content=f"Context compressed successfully.\nResult: {result}")]
                        else:
                            yield [Message(role=ASSISTANT, content=f"Context compression cancelled: {reason}")]
                    else:
                        yield [Message(role=ASSISTANT, content=f"Failed to generate summary: {summary}")]
                else:
                    yield [Message(role=ASSISTANT, content="Error: compress_context tool is not available.")]
                return

        # --- Custom FnCallAgent-style loop with streaming sub-agent support ---
        # messages[0] now contains all stabilized instructions, so caches will hit across turns.
        llm_messages = copy.deepcopy(messages)
            
        # Robustness: Read turn limit and auto-continue settings
        # These may be set on the instance by api_server.py or passed in kwargs
        max_turns = getattr(self, 'max_turns', None) or kwargs.get('max_turns') or 50
        self.auto_continue_enabled = getattr(self, 'auto_continue_enabled', None)
        if self.auto_continue_enabled is None:
            self.auto_continue_enabled = True # default
        
        num_llm_calls_available = max(int(max_turns), MAX_LLM_CALL_PER_RUN)
        response: List[Message] = []

        while num_llm_calls_available > 0:
            if self.agent_pool.stopped:
                logger.info(f"Agent {self.name} stopped by user.")
                yield response
                break
                
            num_llm_calls_available -= 1
    
            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
    
            # --- ASYNC MESSAGE INJECTION ---
            if hasattr(self.agent_pool, 'async_message_queue') and self.agent_pool.async_message_queue:
                while self.agent_pool.async_message_queue:
                    async_msg_text = self.agent_pool.async_message_queue.pop(0)
                    async_msg = Message(role=USER, content=f"[ASYNC INTERRUPTION]: {async_msg_text}")
                    messages.append(async_msg)
                    llm_messages.append(async_msg)
                    response.append(async_msg)
                    logger.info(f"Injected async user message into {self.name}: {async_msg_text}")
                yield response  # Update UI with the injected message
                
            # Inject warning if needed (only for the LLM call, doesn't affect saved history)
            self._inject_compression_warning(llm_messages)
            
            # DEBUG: Inspect message roles to find "Start with User" violations
            # msg_roles = [m.role for m in llm_messages]
            # logger.info(f"LLM Call Order: {msg_roles}")
    
            active_functions = self._get_active_functions()
            
            output_stream = self._call_llm(
                messages=llm_messages,
                functions=active_functions,
                extra_generate_cfg=extra_generate_cfg,
            )
    
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
    
            if not output:
                break
    
            # --- UPDATE HISTORY ---
            # We must ensure reasoning_content is visible to the LLM in the next turn.
            # Most providers don't support 'reasoning_content' in input history,
            # so we merge it into content using standard tags.
            history_output = []
            for msg in output:
                m = copy.deepcopy(msg)
                if m.get('reasoning_content'):
                    reasoning = f"<think>\n{m['reasoning_content']}\n</think>\n"
                    old_content = m.get(CONTENT)
                    if old_content is None:
                        m[CONTENT] = reasoning
                    elif isinstance(old_content, list):
                        # Prepend reasoning as a new text item if it's a list
                        m[CONTENT] = [ContentItem(text=reasoning)] + old_content
                    else:
                        m[CONTENT] = reasoning + str(old_content)
                history_output.append(m)

            response.extend(output)
            messages.extend(output)
            llm_messages.extend(history_output)
    
            # Log generated messages
            is_truncated = False
            for msg in output:
                logger_inst.log_message(msg)
                # Detection: finish_reason is often in extra
                if msg.extra and msg.extra.get('finish_reason') == 'length':
                    is_truncated = True
            
            # --- AUTO-CONTINUE ON LENGTH LIMIT ---
            if is_truncated and self.auto_continue_enabled:
                logger.info(f"Detected message truncation (length limit) for {self.name}. Auto-triggering continuation.")
                # Increment budget to allow the continuation
                num_llm_calls_available += 1
                # Inject a system prompt to continue
                cont_msg = Message(role=USER, content="[SYSTEM]: Your previous response was cut off by the length limit. Please continue exactly from where you left off, without repeating yourself.")
                llm_messages.append(cont_msg)
                # Yield a small hint to the UI
                yield response + output + [Message(role=ASSISTANT, content="... (Continuing output due to length limit)")]
    
            used_any_tool = False
            for out in output:
                use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                if not use_tool:
                    continue
    
                used_any_tool = True
                # Yield the tool call request immediately so UI sees "calling tool..."
                yield response
    
                if tool_name in self.STREAMING_TOOLS:
                    # ── Streaming sub-agent call ──
                    tool_result = yield from self._stream_sub_agent_call(
                        tool_name, tool_args, response, messages
                    )
                else:
                    # ── Normal synchronous tool ──
                    
                    # --- Handle __USE_PREV_ARG__ Placeholder Replacement ---
                    if isinstance(tool_args, str):
                        tool_args = tool_args.strip()
                        if tool_args:
                            try:
                                # Use relaxed json_loads to handle trailing commas or other LLM quirks
                                tool_args = json_loads(tool_args)
                            except Exception:
                                pass # Let _call_tool handle standard verification
                        else:
                            tool_args = {} # Guard against empty string arguments
                    
                    if isinstance(tool_args, dict):
                        # Use the current instance name as the scope for the last_tool_args cache
                        instance_scope = self.session_name
                        
                        # Resolve placeholders
                        placeholders_found = []
                        for arg_key, arg_val in tool_args.items():
                            if arg_val == "__USE_PREV_ARG__":
                                placeholders_found.append(arg_key)
                                
                        if placeholders_found:
                            # 1. Try tool-specific cache first
                            prev_args = self.agent_pool.last_tool_args.get(instance_scope, {}).get(tool_name)
                            
                            # 2. Fallback to global cache for common parameters like 'path'
                            global_args = self.agent_pool.last_tool_args.get(instance_scope, {}).get("__GLOBAL__", {})
                            
                            if not prev_args and not global_args:
                                tool_result = f"Error: Cannot use __USE_PREV_ARG__ for '{tool_name}' because no previous call to this tool was recorded for instance '{instance_scope}'."
                                # Skip tool execution if placeholder fails
                                skip_execution = True
                            else:
                                skip_execution = False
                                for arg_key in placeholders_found:
                                    # Prefer tool-specific, then global
                                    if prev_args and arg_key in prev_args:
                                        tool_args[arg_key] = prev_args[arg_key]
                                    elif arg_key in global_args:
                                        tool_args[arg_key] = global_args[arg_key]
                                    else:
                                        tool_result = f"Error: Cannot use __USE_PREV_ARG__ for argument '{arg_key}' because it was not found in previous calls (neither specific to '{tool_name}' nor globally)."
                                        skip_execution = True
                                        break
                        else:
                            skip_execution = False
                            
                        if not skip_execution:
                            call_kwargs = kwargs.copy()
                            if 'agent_instance_name' not in call_kwargs:
                                call_kwargs['agent_instance_name'] = self.session_name
                            
                            # Pass the agent itself so tools (like compress_context) can sync 
                            # back to its base system_message for persistence across turns.
                            call_kwargs['agent_obj'] = self
                                
                            try:
                                tool_result = self._call_tool(
                                    tool_name, tool_args, messages=llm_messages, 
                                    **call_kwargs
                                )
                            except Exception as e:
                                logger.error(f"Error calling tool {tool_name}: {e}")
                                tool_result = f"Error: {e}"
                                if "valid JSON" in str(e) and isinstance(tool_args, str):
                                    tool_result += f"\nYour arguments: {tool_args[:200]}..."
                            
                            # Caching: Save successful tool args for future reuse
                            # Note: We save them even if the tool returned an error string, 
                            # as long as the arguments themselves were theoretically valid.
                            if instance_scope not in self.agent_pool.last_tool_args:
                                self.agent_pool.last_tool_args[instance_scope] = {}
                            
                            # Local Tool Cache
                            self.agent_pool.last_tool_args[instance_scope][tool_name] = copy.deepcopy(tool_args)
                            
                            # Global Cache Fallback (for cross-tool reuse)
                            if "__GLOBAL__" not in self.agent_pool.last_tool_args[instance_scope]:
                                self.agent_pool.last_tool_args[instance_scope]["__GLOBAL__"] = {}
                            self.agent_pool.last_tool_args[instance_scope]["__GLOBAL__"].update(copy.deepcopy(tool_args))
                    else:
                        # Fallback for non-dict tool_args
                        call_kwargs = kwargs.copy()
                        if 'agent_instance_name' not in call_kwargs:
                            call_kwargs['agent_instance_name'] = self.session_name
                        call_kwargs['agent_obj'] = self
                        try:
                            tool_result = self._call_tool(
                                tool_name, tool_args, messages=llm_messages, 
                                **call_kwargs
                            )
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_name}: {e}")
                            tool_result = f"Error: {e}"
                            if "valid JSON" in str(e) and isinstance(tool_args, str):
                                tool_result += f"\nYour arguments: {tool_args[:200]}..."
    
                # --- Generic truncation: protect ALL tool results ---
                if isinstance(tool_result, str):
                    tool_result = self._truncate_tool_result(
                        tool_result, tool_name, llm_messages, self.session_name
                    )
    
                fn_msg = Message(
                    role=FUNCTION,
                    name=tool_name,
                    content=tool_result,
                    extra={
                        'function_id': out.extra.get('function_id', '1')
                        if out.extra else '1'
                    },
                )
                messages.append(fn_msg)
                llm_messages.append(fn_msg)
                response.append(fn_msg)
                
                # Log just the function result (LLM output already logged above)
                logger_inst.log_message(fn_msg)
                
                # --- ASYNC MESSAGE INJECTION (URGENT) ---
                if hasattr(self.agent_pool, 'async_message_queue') and self.agent_pool.async_message_queue:
                    while self.agent_pool.async_message_queue:
                        async_msg_text = self.agent_pool.async_message_queue.pop(0)
                        async_msg = Message(role=USER, content=f"[ASYNC INTERRUPTION]: {async_msg_text}")
                        messages.append(async_msg)
                        llm_messages.append(async_msg)
                        response.append(async_msg)
                        logger.info(f"Injected urgent async user message mid-tool-loop into {self.name}: {async_msg_text}")
                    yield response
                    break  # CRITICAL: Stop executing the rest of the batched tools!
                    
                yield response
    
            # Check if the turn is truly finished.
            has_real_content = any(out.get('content') and not out.get('content').startswith('<think>') for out in output)
            has_thinking = any(out.get('thought') or out.get('reasoning_content') for out in output)
            
            if not used_any_tool:
                if has_thinking and not has_real_content:
                    # It's a pure thinking turn. Continue to the next turn.
                    logger.info(f"Pure reasoning turn detected for {self.name}. Continuing to next turn.")
                    pass
                else:
                    # Final answer or real content provided, or no thinking at all.
                    break
    
            # No final update_history needed: all messages are logged individually
            # via log_message above (LLM output at line 502, fn results at line 542).
            
            # Expose the final context for the WebUI so it can detect if a compression 
            # occurred and mutated the history mid-turn.
            self.turn_final_messages = messages

        if num_llm_calls_available <= 0:
            logger.warning(f"Agent {self.name} reached turn limit. Stopping.")
            term_msg = Message(role=ASSISTANT, content="\n\n[SYSTEM: Turn limit reached. If the task is incomplete, please ask me to continue.]")
            response.append(term_msg)
            yield response

        yield response

    # ------------------------------------------------------------------ #
    #  Streaming sub-agent execution                                      #
    # ------------------------------------------------------------------ #

    def _stream_sub_agent_call(
        self,
        tool_name: str,
        tool_args: Union[str, dict],
        current_response: List[Message],
        manager_history: List[Message],
    ) -> Iterator[List[Message]]:
        """
        Generator that runs a sub-agent instance, yields intermediate results to the
        WebUI (so streaming is visible), and *returns* the final tool-result
        string via ``return``.  Called via ``yield from`` in ``_run``.
        """
        if isinstance(tool_args, str):
            try:
                args = json.loads(tool_args)
            except json.JSONDecodeError:
                return f'Error: Invalid JSON arguments: {tool_args}'
        else:
            args = tool_args

        instance_name = args.get('instance_name', '')
        agent_class = args.get('agent_class', '')

        # Prevent state corruption when an agent calls ITSELF recursively.
        # If the instance is already in the stack, cloning its state ensures
        # that the outer caller's 'conv' doesn't get polluted by inner messages, 
        # which causes out-of-order UI rendering when 'conv + resp' is joined.
        if instance_name in self.agent_pool.active_stack:
            original_instance = instance_name
            count = self.agent_pool.active_stack.count(instance_name)
            instance_name = f"{instance_name}_child{count}"
            
            # Clone base conversation and append the current turn's context
            base_conv = self.agent_pool.get_conversation(original_instance)
            clone_conv = copy.deepcopy(base_conv)
            if current_response:
                clone_conv.extend(copy.deepcopy(current_response))
            self.agent_pool.instance_conversations[instance_name] = clone_conv

        # 1. Resolve agent class and isolation
        existing_class = self.agent_pool.instance_classes.get(instance_name)
        
        # If the requested class is different from the existing one,
        # we MUST clear history to avoid confusing context mix-ups (stacking).
        if existing_class and agent_class and existing_class != agent_class:
            logger.info(f"Class mismatch for '{instance_name}': {existing_class} -> {agent_class}. Clearing history.")
            self.agent_pool.clear_conversation(instance_name)
            existing_class = None # Trigger fresh initialization
            
        if existing_class:
            agent_class = existing_class
        elif not agent_class:
            return f"Error: No active instance named '{instance_name}'. Provide agent_class to start one."
        
        # Register/Confirm instance class
        self.agent_pool.instance_classes[instance_name] = agent_class

        agent = self.agent_pool.get_agent(agent_class)
        if not agent:
            return (
                f"Error: Agent class '{agent_class}' not found. "
                f"Available classes: {self.agent_pool.list_agents()}"
            )

        if self.agent_pool.stopped:
            return f"Operation cancelled by user."
            
        # Prepare sub-agent logger
        logger_inst = self.agent_pool.get_logger(instance_name, agent_class)

        # Build the final system message FIRST (before conversation init)
        # so there's only ever ONE system prompt in the conversation and logs.
        base_sys = getattr(agent, 'base_system_message', agent.system_message)
        lines = base_sys.strip().split('\n')
        
        if lines:
            # 1. Update the first line: "You are [Role]." -> "You are [Instance]."
            if lines[0].startswith("You are") and f" {instance_name}" not in lines[0]:
                lines[0] = f"You are {instance_name}."
            
            # 2. Insert session metadata after the tagline (usually line 2)
            metadata_block = [
                "## Session Metadata",
                f"- Supervisor: {self.session_name}",
                f"- Log Path: {logger_inst.log_path}",
                "Use your logs to recall details from turns that were compressed.\n"
            ]
            
            # Insert after the tagline if line 2 exists and isn't a header, otherwise after line 1
            insert_pos = 2 if len(lines) > 1 and not lines[1].startswith("#") else 1
            for i, metadata_line in enumerate(metadata_block):
                lines.insert(insert_pos + i, metadata_line)
                
        final_sys_content = "\n".join(lines)
        agent.system_message = final_sys_content

        if tool_name == 'call_agent':
            # Persistent sessions: if instance exists, we just append to it.
            if instance_name not in self.agent_pool.instance_conversations:
                # New instance: use the FINAL system message (with metadata already integrated)
                self.agent_pool.instance_conversations[instance_name] = [
                    Message(role=SYSTEM, content=final_sys_content)
                ]
                # Record the initial state in the log
                logger_inst.update_history(self.agent_pool.instance_conversations[instance_name])
            else:
                # Existing instance: update the system message in-place with latest metadata
                conv_existing = self.agent_pool.instance_conversations[instance_name]
                if conv_existing and conv_existing[0].get(ROLE) == SYSTEM:
                    conv_existing[0].content = final_sys_content
                if not logger_inst.data["history"]:
                    logger_inst.update_history(conv_existing)
            
        task = args.get('task', '')
        context = args.get('context', '')
        
        caller_prefix = f"This is a message from {self.session_name}."
        if context:
            context = f"{caller_prefix}\n{context}"
        else:
            context = caller_prefix

        msg_text = f'Context: {context}\n\nTask: {task}\n\nPlease help with this task.'
        if not msg_text.strip():
            msg_text = "Please proceed with your task."

        # --- Resolve multimodal content (Images) ---
        sub_agent_msg_content = [{'text': msg_text}]
        
        seen_images = {}
        for msg in manager_history:
            content = msg.get(CONTENT)
            if isinstance(content, list):
                for item in content:
                    # Content item might be dict or object
                    item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                    item_value = item.get('value') if isinstance(item, dict) else getattr(item, 'value', None)
                    if item_type == IMAGE:
                        img_url = item_value
                        basename = get_basename_from_url(img_url)
                        seen_images[basename] = img_url
                        idx = len([v for k, v in seen_images.items() if not k.startswith("image_")]) - 1
                        seen_images[f"image_{idx}"] = img_url

        added_to_sub = set()
        for ref, img_url in seen_images.items():
            if ref in msg_text and img_url not in added_to_sub:
                sub_agent_msg_content.append({IMAGE: img_url})
                added_to_sub.add(img_url)
        
        if len(sub_agent_msg_content) == 1 and tool_name == 'call_agent':
            last_user_msg = next((m for m in reversed(manager_history) if m.get(ROLE) == USER), None)
            if last_user_msg:
                content = last_user_msg.get(CONTENT)
                if isinstance(content, list):
                    for item in content:
                        item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                        item_value = item.get('value') if isinstance(item, dict) else getattr(item, 'value', None)
                        if item_type == IMAGE and item_value not in added_to_sub:
                            sub_agent_msg_content.append({IMAGE: item_value})
                            added_to_sub.add(item_value)

        # streaming status is updated during the run

        # Initialize sub-agent chat for new turn
        # This block is from web_ui.py, not agent_orchestrator.py.
        # The instruction to reset self._last_active_sa, _last_stack_top, and _last_rendered_sa
        # applies to the agent_run method in web_ui.py, not here.
        # The provided code snippet for the change is a mix of files.
        # I will only apply the change to agent_orchestrator.py as per the first part of the instruction.
        # The second part of the instruction for web_ui.py cannot be applied here.

        conv = self.agent_pool.get_conversation(instance_name)
        user_msg = {ROLE: USER, CONTENT: sub_agent_msg_content}
        conv.append(user_msg)
        
        # Record user message in persistent log
        logger_inst.log_message(user_msg)
        
        # Track this call in the active stack for UI context switching
        self.agent_pool.active_stack.append(instance_name)

        # Initialize streaming state for the WebUI
        state = {
            'active': True,
            'agent_name': f"{instance_name} ({agent_class})",
            'messages': copy.deepcopy(conv),
        }
        # Overwrite any existing state for this instance
        self.agent_pool.sub_agent_state[instance_name] = state

        # Force an immediate yield so the WebUI detects the new active_stack entry
        # and switches the subagent window context immediately.
        yield current_response

        # ── Propagate disabled_tools to sub-agent's LLM ──
        # The API server only patches the orchestrator's llm.generate_cfg with
        # disabled_tools from the frontend. Sub-agents have their own LLM instances,
        # so we must copy the policy over before running them.
        # ── Propagate agent settings to sub-agent ──
        if hasattr(agent, 'max_turns') is False:
            agent.max_turns = getattr(self, 'max_turns', 50)
        if hasattr(agent, 'auto_continue_enabled') is False:
            agent.auto_continue_enabled = getattr(self, 'auto_continue_enabled', True)

        orchestrator_disabled = getattr(self.llm, 'generate_cfg', {}).get('disabled_tools')
        if orchestrator_disabled and hasattr(agent, 'llm') and agent.llm:
            if not hasattr(agent.llm, 'generate_cfg') or agent.llm.generate_cfg is None:
                agent.llm.generate_cfg = {}
            agent.llm.generate_cfg['disabled_tools'] = orchestrator_disabled

        # Run the sub-agent as a generator
        final_resp: list = []
        try:
            # ── Monkey-patch sub-agent's _call_llm to enforce compression ──
            if not hasattr(agent, '_original_call_llm'):
                agent._original_call_llm = agent._call_llm

                def hooked_call_llm(self_agent, messages: List[Message], **kwargs_llm):
                    # --- ASYNC MESSAGE INJECTION ---
                    if hasattr(self.agent_pool, 'async_message_queue') and self.agent_pool.async_message_queue:
                        while self.agent_pool.async_message_queue:
                            async_msg_text = self.agent_pool.async_message_queue.pop(0)
                            async_msg = Message(role=USER, content=f"[ASYNC INTERRUPTION]: {async_msg_text}")
                            messages.append(async_msg)
                            logger.info(f"Injected async user message into sub-agent {instance_name} via LLM hook: {async_msg_text}")

                    self._inject_compression_warning_for_agent(self_agent, instance_name, messages)
                    
                    # --- CALL LLM WITH AUTO-CONTINUE ---
                    # We wrap the generator to detect truncation (finish_reason='length')
                    last_output = []
                    for output in self_agent._original_call_llm(messages, **kwargs_llm):
                        last_output = output
                        yield output
                    
                    # Detect truncation
                    is_truncated = False
                    for msg in last_output:
                        if msg.extra and msg.extra.get('finish_reason') == 'length':
                            is_truncated = True
                            break
                    
                    # Read auto-continue setting from current orchestrator config
                    # Note: self is the OrchestratorAgent instance
                    auto_continue_enabled = getattr(self, 'auto_continue_enabled', True)
                    
                    if is_truncated and auto_continue_enabled:
                        logger.info(f"Sub-agent {instance_name} truncated by length limit. Auto-triggering continuation...")
                        cont_msg = Message(role=USER, content="[SYSTEM]: Your previous response was cut off by the token limit. Please continue exactly from where you left off, without repeating yourself.")
                        messages.append(cont_msg)
                        # Recursive call to continue the same LLM turn
                        yield from hooked_call_llm(self_agent, messages, **kwargs_llm)

                import types
                agent._call_llm = types.MethodType(hooked_call_llm, agent)
            
            # agent.run mutates the passed list, so we pass a copy to avoid double-appending
            # when we do state['messages'] = conv + resp
            run_conv = copy.deepcopy(conv)
            
            # Pass instance name through kwargs so tools (like compress_context) know who they are contextually
            for resp in agent.run(run_conv, agent_instance_name=instance_name):
                if self.agent_pool.stopped:
                    logger.info(f"Sub-agent {instance_name} interrupted by user stop signal.")
                    yield current_response
                    break
                    
                final_resp = resp

                # Update streaming state for WebUI
                state['messages'] = list(conv) + list(resp)
                yield current_response
                
                # Efficient logging: check if a tool call was just completed
                if resp and (resp[-1].get(ROLE) == FUNCTION or resp[-1].get('function_call')):
                    # Log full conversation snapshot on tool events
                    logger_inst.update_history(conv + resp)
                    
                    # Note: Context compression mutates the pool history object in-place,
                    # and 'conv' is a reference to that object, so it stays in sync automatically.

            if final_resp:
                # IMPORTANT: Update the persistent conversation instance with the FULL TURN result.
                # This ensures the next turn (or next 'call_agent') sees the tool results.
                conv.extend(final_resp)
                
                # Final log sync for the session turn
                logger_inst.update_history(conv)
                
                # Extraction logic: get only text blocks from the last successful turn 
                # to avoid repeating the whole task/context history in the manager's prompt.
                result_str = extract_sub_agent_feedback(final_resp, instance_name)
                return f"[{instance_name}'s output]:\n{result_str}"
            else:
                return f"[{instance_name}] finished with no output."

        except Exception as e:
            logger.error(f"Error in sub-agent {instance_name}: {str(e)}", exc_info=True)
            return f"Error executing sub-agent {instance_name}: {str(e)}"
        finally:
            # Clean up state
            state['active'] = False
            self.agent_pool.sub_agent_state[instance_name] = state
            
            # Remove from active stack (pop the most recent occurrence)
            for i in range(len(self.agent_pool.active_stack) - 1, -1, -1):
                if self.agent_pool.active_stack[i] == instance_name:
                    self.agent_pool.active_stack.pop(i)
                    break




# ─── Utility ───────────────────────────────────────────────────────────────────

def extract_sub_agent_feedback(messages: List[Dict], instance_name: str) -> str:
    """
    Extracts text output from sub-agent messages.
    Only includes text generated AFTER the last tool call ended.
    """
    last_tool_idx = -1
    for i, msg in enumerate(messages):
        if msg.get(ROLE) == FUNCTION or msg.get('function_call'):
            last_tool_idx = i

    relevant_msgs = messages[last_tool_idx + 1:] if last_tool_idx != -1 else messages

    collected_text = []
    for msg in relevant_msgs:
        if isinstance(msg, dict):
            msg_role = msg.get('role', '')
        else:
            msg_role = msg.role

        if msg_role == ASSISTANT:
            text = extract_text_from_message(msg, add_upload_info=False)
            if text:
                collected_text.append(text)

    result_str = "\n\n".join(collected_text).strip()

    if not result_str:
        if last_tool_idx != -1:
            return f"WARNING: Sub-agent {instance_name} performed tool calls but provided no final summary."
        return f"Sub-agent {instance_name} finished but provided no text output."

    return result_str


# ─── Backward-compatible re-exports ────────────────────────────────────────────
# These were extracted into their own modules during the restructure.
# Existing imports like `from agent_orchestrator import AgentPool` still work.

from agent_pool import AgentPool  # noqa: F401
from agent_logger import AgentInstanceLogger  # noqa: F401
from agent_factory import load_orchestrator_agent, load_sub_agent_with_tools  # noqa: F401


