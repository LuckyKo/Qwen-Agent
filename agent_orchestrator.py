"""
Agent Orchestrator - An agent that can call other agents dynamically
Each sub-agent has its own soul.md, context, and specialized job.

User Approval System:
- All mutating tool calls (file write/edit/delete/move/copy) block until the
  user approves or rejects via the WebUI.
- Read operations are free access.
"""

import copy
import os
import json
import logging
import datetime
import time
from qwen_agent.agents import Assistant
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)

from qwen_agent.llm.schema import (
    ASSISTANT,
    CONTENT,
    FILE,
    FUNCTION,
    IMAGE,
    ROLE,
    SYSTEM,
    USER,
    Message,
)
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools.base import BaseTool, register_tool
from soul_loader import create_agent_from_soul
from operation_manager import OperationManager
from qwen_agent.utils.utils import (
    extract_text_from_message,
    get_basename_from_url,
    merge_generate_cfgs,
)
from qwen_agent.tools.custom import (
    CallAgent,
    ReadFile,
    ViewImage,
    WriteFile,
    EditFile,
    ListDir,
    Grep,
    DeleteFile,
    CopyFile,
    MoveFile,
    DismissAgent,
    ListAgents,
    ShellCmd,
)
from qwen_agent.tools.code_interpreter import CodeInterpreter
from qwen_agent.tools.custom.compression_tools import CompressContext


class AgentPool:
    """Manages a pool of specialized sub-agents."""
    
    def __init__(self, llm_cfg: dict, agents_dir: str = 'agents'):
        self.llm_cfg = llm_cfg
        self.agents_dir = Path(agents_dir)
        # Agent templates (loaded by class name)
        self.agents: Dict[str, Assistant] = {}
        self.agent_configs: Dict[str, dict] = {}
        
        # Initialize OperationManager for blocking approvals
        from operation_manager import OperationManager
        self.operation_manager = OperationManager(agent_pool=self)
        
        # Persistent conversation histories for each named instance
        self.instance_conversations: Dict[str, List] = {}
        # Mapping of instance_name to its agent_class
        self.instance_classes: Dict[str, str] = {}
        
        # Mapping of instance_name to its AgentInstanceLogger
        self.instance_loggers: Dict[str, 'AgentInstanceLogger'] = {}
        
        # Live streaming state for WebUI (updated during sub-agent execution)
        self.sub_agent_state: Dict[str, dict] = {}
        
        # List of instance_names currently in an active call (stack for recursion)
        self.active_stack: List[str] = []
        
        # Explicit stop flag for cancellation
        self.stopped = False
        
        # Auto-load all agents from the agents directory
        self._discover_agents()
    
    def get_logger(self, instance_name: str, agent_class: str) -> 'AgentInstanceLogger':
        """Get or create a logger for an agent instance."""
        if instance_name not in self.instance_loggers:
            # Ensure workspace/logs exists
            log_dir = Path('workspace/logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.instance_loggers[instance_name] = AgentInstanceLogger(
                agent_class=agent_class,
                instance_name=instance_name,
                log_dir=str(log_dir)
            )
        return self.instance_loggers[instance_name]
    
    def _discover_agents(self):
        """Find and load all agent configurations from the agents directory."""
        if not self.agents_dir.exists():
            self.agents_dir.mkdir(exist_ok=True)
            # Create a default example agent
            self._create_example_agent()
        
        # Load all *_soul.md files
        for soul_file in self.agents_dir.glob('*_soul.md'):
            agent_name = soul_file.name.replace('_soul.md', '')
            try:
                self.load_agent(agent_name)
                print(f"[OK] Loaded agent: {agent_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load agent {agent_name}: {e}")
    
    def _create_example_agent(self):
        """Create an example sub-agent."""
        soul_path = self.agents_dir / 'researcher_soul.md'
        
        soul_content = """name: Researcher
tagline: Deep research specialist

identity:
  role: Academic and technical research expert
  background: |
    You specialize in deep research, analysis, and synthesizing complex information.
    You're methodical, thorough, and love diving into technical details.
  personality_traits:
    - Analytical and detail-oriented
    - Patient and systematic
    - Loves citing sources and evidence
    - Asks clarifying questions

communication:
  tone: Professional, precise, academic
  style_notes:
    - Always cite sources when using web_search
    - Break down complex topics step by step
    - Use technical terms when appropriate
    - Summarize key findings clearly

capabilities:
  tools:
    - web_search
    - visit_website
  
  skills:
    - Literature review
    - Technical analysis
    - Fact verification
    - Source evaluation

rules:
  - Always verify information from multiple sources
  - Cite your sources explicitly
  - Distinguish between facts and opinions
  - Admit uncertainty when evidence is weak
"""
        soul_path.write_text(soul_content)
    
    def load_agent(self, agent_name: str) -> Assistant:
        """Load or reload a specific agent with file tools."""
        soul_path = self.agents_dir / f'{agent_name}_soul.md'
        
        if not soul_path.exists():
            raise FileNotFoundError(f"No soul.md found for agent: {agent_name}")
        
        # Load agent with file tools
        agent = load_sub_agent_with_tools(self, agent_name, self.llm_cfg)
        
        self.agents[agent_name] = agent
        self.agent_configs[agent_name] = agent.agent_configs.get(agent_name, {})
        
        return agent
    
    def get_agent(self, agent_name: str) -> Optional[Assistant]:
        """Get an agent by name."""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.agents.keys())
    
    def get_conversation(self, instance_name: str) -> List:
        """Get or create persistent conversation history for an agent instance."""
        if instance_name not in self.instance_conversations:
            self.instance_conversations[instance_name] = []
        return self.instance_conversations[instance_name]

    def clear_conversation(self, instance_name: str):
        """Clear an agent instance's conversation history."""
        self.instance_conversations.pop(instance_name, None)
        self.instance_classes.pop(instance_name, None)
        self.instance_loggers.pop(instance_name, None)
        self.sub_agent_state.pop(instance_name, None)

    def reset(self):
        """Full reset of all sub-agent instances and persistent data."""
        self.instance_conversations.clear()
        self.instance_classes.clear()
        self.instance_loggers.clear()
        self.sub_agent_state.clear()
        self.stopped = False
        logger.info("AgentPool reset — all instances and loggers cleared.")
    
    def _apply_context_compression(self, agent_name: str, summary: str, fraction: float, agent_obj: Optional[Assistant] = None):
        """
        Actually replace the oldest messages in an agent's history with a summary.
        Called by OperationManager after approval.
        """
        history = self.get_conversation(agent_name)
        if not history:
            return
            
        # Keep the first message if it's SYSTEM
        system_msg = None
        start_idx = 0
        first_role = history[0].get('role') if isinstance(history[0], dict) else getattr(history[0], 'role', '')
        if first_role == SYSTEM:
            system_msg = history[0]
            start_idx = 1
            
        messages_to_compress = history[start_idx:]
        # Ensure we remove at least 1 message if possible
        num_to_remove = max(1, int(len(messages_to_compress) * fraction))
        
        # ADJUSTMENT: Ensure the first remaining message is a USER message.
        # Specifically, we scan forward from num_to_remove to find the NEXT USER message.
        # However, we must NEVER remove the very last message in the history (which is usually 
        # the current turn's prompt or assistant/function message).
        found_user = False
        temp_remove = num_to_remove
        while temp_remove < len(messages_to_compress):
            next_msg = messages_to_compress[temp_remove]
            role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', '')
            if role == USER:
                found_user = True
                num_to_remove = temp_remove
                break
            temp_remove += 1
            
        # If we didn't find a USER message by scanning forward, we MUST scan BACKWARD 
        # from our target to ensure the history we keep starts with a USER message.
        if not found_user:
            temp_remove = num_to_remove - 1
            while temp_remove >= 0:
                next_msg = messages_to_compress[temp_remove]
                role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', '')
                if role == USER:
                    found_user = True
                    num_to_remove = temp_remove
                    break
                temp_remove -= 1
        
        # If we STILL found no USER message (which should be impossible), 
        # don't remove anything - better to be full than 400 error.
        if not found_user:
            logger.warning(f"Compression for {agent_name} could not find a USER message to start with. Skipping compression.")
            return
            
        if num_to_remove <= 0:
            return
            
        # Create the summary text
        summary_text = f"\n\n--- CONTEXT COMPRESSED ({int(fraction*100)}% of history summarized) ---\n\nSummary of previous context:\n{summary}\n\n--- END SUMMARY ---"
        
        # New history: [System (with summary appended)] + [Remaining Messages]
        new_history = []
        if system_msg:
            # Merge summary INTO the existing system message to avoid a second SYSTEM message
            if isinstance(system_msg, dict):
                merged_content = system_msg.get('content', '') + summary_text
                merged_msg = {**system_msg, 'content': merged_content}
            else:
                merged_content = system_msg.content + summary_text
                merged_msg = Message(role=system_msg.role, content=merged_content)
            
            new_history.append(merged_msg)
            
            # Sync back to the agent's base system_message so it persists across turns
            # (since web_ui.py strips historical system messages to avoid duplicates)
            if agent_obj and hasattr(agent_obj, 'system_message'):
                agent_obj.system_message = merged_content
        else:
            # No system message exists — insert summary as a USER message to stay compliant
            # Note: In this case, persistence is handled by the USER message itself being in history
            new_history.append({'role': USER, 'content': summary_text})
        new_history.extend(messages_to_compress[num_to_remove:])
        
        # Modify list in-place so active references (like 'conv' in _stream_sub_agent_call) remain valid!
        history.clear()
        history.extend(new_history)
        
        # Reset the logger's internal tracking to this new baseline.
        # The additive update_history() method can't handle a rewritten history —
        # it would re-append all remaining messages as "new". A hard reset writes
        # a COMPRESSION_RESET marker and reinitializes from the compressed state.
        if agent_name in self.instance_loggers:
            self.instance_loggers[agent_name].reset_history(new_history)
            
        logger.info(f"Compressed context for agent '{agent_name}'. Removed {num_to_remove} messages.")
    
    def get_agent_info(self, agent_name: str) -> Optional[dict]:
        """Get info about a specific agent."""
        config = self.agent_configs.get(agent_name)
        if not config:
            return None
        
        return {
            'name': config.get('name', agent_name),
            'tagline': config.get('tagline', ''),
            'tools': config.get('capabilities', {}).get('tools', []),
            'description': config.get('identity', {}).get('background', ''),
        }



# ─── Sub-agent function schemas ────────────────────────────────────────────────
# These are NOT called via _call_tool; OrchestratorAgent._run intercepts them
# and handles them as streaming generators. They exist only so the LLM sees
# them in the function list.

CALL_AGENT_SCHEMA = {
    'name': 'call_agent',
    'description': (
        'Delegate a task to a specialized sub-agent. '
        'If the instance_name already exists, the session continues with the existing context. '
        'Otherwise, a new session is started using the specified agent_class.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'agent_class': {
                'type': 'string',
                'description': 'The class of agent to call (e.g. "coder", "researcher", "writer")'
            },
            'instance_name': {
                'type': 'string',
                'description': 'A unique name for this agent instance. Use this to continue the session later.'
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
        self.session_name: str = "MainSession"
        from qwen_agent.utils.tokenization_qwen import count_tokens
        self._count_tokens = count_tokens

    def _get_history_tokens(self, messages: List[Message]) -> int:
        """Calculate total tokens in a message list."""
        total = 0
        for msg in messages:
            content = extract_text_from_message(msg, add_upload_info=False)
            total += self._count_tokens(content)
        return total

    def _inject_compression_warning(self, messages: List[Message]):
        """Inject a warning if context is getting full."""
        # Get actual max tokens (detected from API or default)
        max_tokens = self.llm.generate_cfg.get('max_input_tokens', 58000)
        current_tokens = self._get_history_tokens(messages)
        
        if current_tokens > max_tokens * 0.85:
            usage_pct = (current_tokens / max_tokens) * 100
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
        # Prepend knowledge like Assistant does
        messages = self._prepend_knowledge_prompt(
            messages=messages, lang=lang, knowledge=knowledge, **kwargs
        )
        
        # Identity formatting: "[Role] [Instance/SessionName]"
        instance = kwargs.get('agent_instance_name') or self.session_name
        full_name = self.name
        if instance:
            if not full_name.lower().endswith(instance.lower()):
                full_name = f"{self.agent_type} {instance}"
        
        self.name = full_name # Update for UI display
        
        # Prepare appropriate logger (Main Session or Sub-Agent Instance)
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
                    # Look for "You are [Type]." anywhere in the prompt
                    # This handles both cases: 
                    # 1. Start of prompt (standard Assistant)
                    # 2. After [IDENTITY] block (sub-agent instance call)
                    pattern = rf"(?i)You are {self.agent_type}\."
                    new_intro = f"You are {self.agent_type} {instance}."
                    
                    if re.search(pattern, m0_content):
                        m0_content = re.sub(pattern, new_intro, m0_content, count=1)
                    
                    # Insert Session Metadata section if not already present
                    if '## Session Metadata' not in m0_content:
                        meta_lines = [
                            "## Session Metadata",
                            f"- Supervisor: User",
                            f"- Log Path: {logger_inst.log_path}",
                            "Use your logs to recall details from turns that were compressed.\n"
                        ]
                        content_lines = m0_content.split('\n')
                        insert_pos = 2 if len(content_lines) > 1 and not content_lines[1].startswith("#") else 1
                        for i, ml in enumerate(meta_lines):
                            content_lines.insert(insert_pos + i, ml)
                        m0_content = '\n'.join(content_lines)
                    
                    if isinstance(m0, dict):
                        m0['content'] = m0_content
                    else:
                        m0.content = m0_content

            # Sync initial state if history is empty
            if not logger_inst.data["history"]:
                logger_inst.update_history(messages)
            elif not kwargs.get('agent_instance_name'):
                # Only log the last message for the main orchestrator session.
                # Sub-agents are already logged by _stream_sub_agent_call to avoid duplicates.
                logger_inst.log_message(messages[-1])

        # --- Custom FnCallAgent-style loop with streaming sub-agent support ---
        llm_messages = copy.deepcopy(messages)
        
        # --- Dynamically inject available sub-agents and tools ---
        if llm_messages and llm_messages[0].role == SYSTEM:
            # We want to modify the system message specifically for this run without
            # altering the base system message permanently or the persistent log.
            
            system_append = "\n\n--- CURRENT AVAILABLE RESOURCES (Auto-Injected) ---\n"
            
            # 1. Inject available sub-agents
            system_append += "\nAvailable Sub-Agents (call via call_agent):\n"
            has_agents = False
            for name in self.agent_pool.list_agents():
                if name.lower() != self.name.lower(): # Don't list self
                    info = self.agent_pool.get_agent_info(name)
                    if info:
                        system_append += f"- **{info['name']}**: {info['tagline']}\n"
                        has_agents = True
            if not has_agents:
                system_append += "- None currently available.\n"
                
            # 2. Inject available tools
            system_append += "\nEnabled Tools (can change per interaction):\n"
            if self.function_map:
                for t_name, t_obj in self.function_map.items():
                    desc = getattr(t_obj, 'description', 'No description provided')
                    system_append += f"- **{t_name}**: {desc}\n"
            else:
                system_append += "- None currently enabled.\n"
                
            # Modify the content of the first message (which is the SYSTEM message)
            llm_messages[0].content += system_append
            
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
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

            # Inject warning if needed (only for the LLM call, doesn't affect saved history)
            self._inject_compression_warning(llm_messages)
            
            # DEBUG: Inspect message roles to find "Start with User" violations
            msg_roles = [m.role for m in llm_messages]
            logger.info(f"LLM Call Order: {msg_roles}")

            output_stream = self._call_llm(
                messages=llm_messages,
                functions=[func.function for func in self.function_map.values()],
                extra_generate_cfg=extra_generate_cfg,
            )

            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output

            if not output:
                break

            response.extend(output)
            messages.extend(output)
            llm_messages.extend(output)

            # Log generated messages
            for msg in output:
                logger_inst.log_message(msg)

            used_any_tool = False
            for out in output:
                use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                if not use_tool:
                    continue

                used_any_tool = True

                if tool_name in self.STREAMING_TOOLS:
                    # ── Streaming sub-agent call ──
                    tool_result = yield from self._stream_sub_agent_call(
                        tool_name, tool_args, response, messages
                    )
                else:
                    # ── Normal synchronous tool ──
                    call_kwargs = kwargs.copy()
                    if 'agent_instance_name' not in call_kwargs:
                        call_kwargs['agent_instance_name'] = self.session_name
                    
                    # Pass the agent itself so tools (like compress_context) can sync 
                    # back to its base system_message for persistence across turns.
                    call_kwargs['agent_obj'] = self
                        
                    tool_result = self._call_tool(
                        tool_name, tool_args, messages=messages, 
                        **call_kwargs
                    )
                    
                    if tool_name == 'compress_context':
                        # Sync llm_messages with the newly compressed history in 'messages'.
                        # This stops the LLM from re-evaluating the huge context mid-turn.
                        llm_messages = copy.deepcopy(messages)
                        logger.info(f"Synced llm_messages for {self.name} after context compression.")

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
                
                yield response

            if not used_any_tool:
                break

        # No final update_history needed: all messages are logged individually
        # via log_message above (LLM output at line 502, fn results at line 542).
        
        # Expose the final context for the WebUI so it can detect if a compression 
        # occurred and mutated the history mid-turn.
        self.turn_final_messages = messages

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

        # 1. Resolve agent class and instance
        if not agent_class:
            agent_class = self.agent_pool.instance_classes.get(instance_name)
            if not agent_class:
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
            # 1. Update the first line: "You are [Role]." -> "You are [Role] [Instance]."
            if lines[0].startswith("You are") and f" {instance_name}" not in lines[0]:
                lines[0] = lines[0].replace(".", f" {instance_name}.", 1)
            
            # 2. Insert session metadata after the tagline (usually line 2)
            metadata_block = [
                "## Session Metadata",
                f"- Supervisor: {self.name} ({self.__class__.__name__})",
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
        
        caller_prefix = f"This is a message from {self.name}."
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
        self.agent_pool.sub_agent_state[instance_name] = state

        # Run the sub-agent as a generator
        final_resp: list = []
        try:
            # Pass instance name through kwargs so tools (like compress_context) know who they are contextually
            for resp in agent.run(conv, agent_instance_name=instance_name):
                if self.agent_pool.stopped:
                    logger.info(f"Sub-agent {instance_name} interrupted by user stop signal.")
                    yield current_response
                    break
                    
                final_resp = resp
                
                # Update streaming state for WebUI
                state['messages'] = conv + resp
                yield current_response
                
                # Efficient logging: check if a tool call was just completed
                if resp and (resp[-1].get(ROLE) == FUNCTION or resp[-1].get('function_call')):
                    # Log full conversation snapshot on tool events
                    logger_inst.update_history(conv + resp)
                    
                    # Check if context compression occurred during this tool call.
                    # The `AgentPool.get_conversation` list will have shrunk.
                    pool_history = self.agent_pool.get_conversation(instance_name)
                    if pool_history and len(pool_history) < len(conv):
                        logger.info(f"Context compression detected in sub-agent {instance_name}. Syncing orchestrator history.")
                        
                        # The `resp` array holds the NEW messages from this current active turn.
                        # The `conv` array holds the OLD history from before this turn started.
                        # Since the tool compressed the AgentPool, we just need to make `conv` match
                        # the newly compressed history list, minus the currently active `resp` elements
                        # (which are usually appended at the very end of the pool history by the tool).
                        
                        # Actually, our compression tool replaces the pool history with the new summary + old messages.
                        # Since `resp` hasn't been added to the pool yet, `pool_history` IS exactly what `conv` should be!
                        conv.clear()
                        conv.extend(pool_history)

            
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


# ─── Agent loading ─────────────────────────────────────────────────────────────

def load_orchestrator_agent(agent_pool: AgentPool, llm_cfg: dict) -> Assistant:
    """
    Load the orchestrator as an OrchestratorAgent with soul.md configuration
    and streaming sub-agent support.
    """
    soul_path = agent_pool.agents_dir / 'orchestrator_soul.md'

    if soul_path.exists():
        from soul_loader import load_soul, build_system_prompt
        config = load_soul(str(soul_path))
        system_prompt = build_system_prompt(config)
    else:
        config = {}
        system_prompt = _default_orchestrator_prompt(agent_pool)

    # Create OrchestratorAgent directly
    # Identity formatting (Orchestrator SessionName) is handled in _run
    raw_name = config.get('name', 'Orchestrator')
    
    agent = OrchestratorAgent(
        agent_pool=agent_pool,
        llm=llm_cfg,
        name=raw_name,
        agent_type="Orchestrator",
        description=config.get('tagline', 'Supervisor agent that coordinates sub-agents'),
        system_message=system_prompt,
        function_list=[],
    )

    # Store config for agent selector UI
    agent.agent_configs = {config.get('name', 'orchestrator'): config}

    # ── Register sub-agent tools (intercepted in _run, not _call_tool) ──
    agent.function_map['call_agent'] = _SubAgentFunctionProxy(CALL_AGENT_SCHEMA)
    agent.function_map['dismiss_agent'] = DismissAgent(agent_pool=agent_pool)
    agent.function_map['list_agents'] = ListAgents(agent_pool=agent_pool)

    # ── File tools (reads are free, writes block for user approval) ──
    read_tool = ReadFile()
    read_tool.agent_pool = agent_pool
    agent.function_map['read_file'] = read_tool

    view_tool = ViewImage()
    view_tool.agent_pool = agent_pool
    agent.function_map['view_image'] = view_tool

    agent.function_map['list_dir'] = ListDir()
    agent.function_map['list_dir'].agent_pool = agent_pool

    agent.function_map['grep'] = Grep()
    agent.function_map['grep'].agent_pool = agent_pool

    # ── Write/mutating file tools (block for user approval) ──
    write_tool = WriteFile()
    write_tool.agent_pool = agent_pool
    write_tool.agent_name = 'orchestrator'
    agent.function_map['write_file'] = write_tool

    edit_tool = EditFile()
    edit_tool.agent_pool = agent_pool
    edit_tool.agent_name = 'orchestrator'
    agent.function_map['edit_file'] = edit_tool

    delete_tool = DeleteFile()
    delete_tool.agent_pool = agent_pool
    delete_tool.agent_name = 'orchestrator'
    agent.function_map['delete_file'] = delete_tool

    copy_tool = CopyFile()
    copy_tool.agent_pool = agent_pool
    copy_tool.agent_name = 'orchestrator'
    agent.function_map['copy_file'] = copy_tool

    move_tool = MoveFile()
    move_tool.agent_pool = agent_pool
    move_tool.agent_name = 'orchestrator'
    agent.function_map['move_file'] = move_tool

    # ── Context compression ──
    compress_tool = CompressContext()
    compress_tool.agent_pool = agent_pool
    compress_tool.agent_name = 'orchestrator'
    agent.function_map['compress_context'] = compress_tool

    # ── Shell Execution ──
    shell_tool = ShellCmd()
    shell_tool.agent_pool = agent_pool
    shell_tool.agent_name = 'orchestrator'
    agent.function_map['shell_cmd'] = shell_tool

    return agent


def _default_orchestrator_prompt(agent_pool: AgentPool) -> str:
    """Fallback system prompt when no soul.md exists."""
    prompt = """You are a supervisor agent that coordinates with specialized sub-agents.

Your role:
1. Understand the user's request
2. Determine if you need help from a specialized sub-agent
3. Use call_agent to delegate tasks to the right expert
4. Use continue_with_agent to send follow-ups to agents with existing context
5. Use dismiss_agent when you're done with a sub-agent's context
6. Synthesize responses from sub-agents and present to the user

Available sub-agents:
"""
    for name in agent_pool.list_agents():
        info = agent_pool.get_agent_info(name)
        if info:
            prompt += f"\n- **{info['name']}**: {info['tagline']}"

    prompt += """

Sub-Agent Conversation Tools:
- call_agent: Work with a specialized sub-agent. If the instance_name already exists, the session continues.
- dismiss_agent: Clear a sub-agent's conversation context
"""
    return prompt


def create_orchestrator(
    llm_cfg: dict,
    agent_pool: AgentPool,
    system_instruction: str = None
) -> Assistant:
    """
    Create a supervisor/orchestrator agent that can delegate to sub-agents.
    The orchestrator has manager privileges for file operations.
    """

    if not system_instruction:
        system_instruction = """You are a supervisor agent that coordinates with specialized sub-agents.

Your role:
1. Understand the user's request
2. Determine if you need help from a specialized sub-agent
3. Use the call_agent tool to delegate tasks to the right expert
4. Synthesize responses from sub-agents and present to the user

Available sub-agents:
"""

        # Add info about available agents
        for agent_name in agent_pool.list_agents():
            info = agent_pool.get_agent_info(agent_name)
            if info:
                system_instruction += f"""
- **{info['name']}**: {info['tagline']}
  Tools: {', '.join(info['tools'])}
  Expertise: {info['description'][:100]}...
"""

    system_instruction += """
Guidelines:
- Call sub-agents when you need specialized expertise
- Provide clear context and task descriptions
- You can call multiple agents for complex tasks
- Always synthesize and summarize results for the user

File Operations (You have direct access):
- read_file: Read any file (free access). Large files are paginated.
- view_image: View an image file (free access). Returns the image for you to analyze.
- list_dir: List directory contents
- grep: Search for text patterns in files

User Approval System:
- All mutating operations (file write, edit, delete, move, copy) require user approval.
- The user will see a prompt and can approve or reject each operation.
- If rejected, you'll receive the user's reason. Adjust your approach accordingly.
- dismiss_agent: Clear a sub-agent's conversation history
"""

    # Create the orchestrator agent with manager tools
    orchestrator = Assistant(
        llm=llm_cfg,
        name='Orchestrator',
        description='Supervisor agent that coordinates with specialized sub-agents',
        system_message=system_instruction,
        function_list=['call_agent']
    )

    # Register the tools manually (not via framework registry)
    orchestrator.function_map['call_agent'] = CallAgent(agent_pool=agent_pool)

    view_tool = ViewImage()
    view_tool.agent_pool = agent_pool
    orchestrator.function_map['view_image'] = view_tool

    compress_tool = CompressContext()
    compress_tool.agent_pool = agent_pool
    compress_tool.agent_name = 'orchestrator'
    orchestrator.function_map['compress_context'] = compress_tool

    return orchestrator


def load_sub_agent_with_tools(agent_pool: AgentPool, agent_name: str, llm_cfg: dict) -> Assistant:
    """
    Load a sub-agent from soul.md and give it file operation tools and the ability to spawn sub-agents recursively.
    """
    soul_path = agent_pool.agents_dir / f'{agent_name}_soul.md'
    if not soul_path.exists():
        raise FileNotFoundError(f"No soul.md found for agent: {agent_name}")
    
    # Load the agent as an OrchestratorAgent so it can spawn sub-agents
    agent, config = create_agent_from_soul(
        llm_cfg, 
        str(soul_path), 
        agent_class=OrchestratorAgent, 
        agent_pool=agent_pool,
        role_name=agent_name # e.g. 'writer', 'researcher'
    )
    
    # Add recursive sub-agent tools (intercepted in _run)
    agent.function_map['call_agent'] = _SubAgentFunctionProxy(CALL_AGENT_SCHEMA)
    agent.function_map['dismiss_agent'] = DismissAgent(agent_pool=agent_pool)
    agent.function_map['list_agents'] = ListAgents(agent_pool=agent_pool)
    
    # Add file tools to the agent (instantiate directly, then set attrs)
    read_tool = ReadFile()
    read_tool.agent_pool = agent_pool
    agent.function_map['read_file'] = read_tool
    
    view_tool = ViewImage()
    view_tool.agent_pool = agent_pool
    agent.function_map['view_image'] = view_tool
    
    write_tool = WriteFile()
    write_tool.agent_pool = agent_pool
    write_tool.agent_name = agent_name
    agent.function_map['write_file'] = write_tool
    
    # Add Python Sandbox tool (Code Interpreter)
    # The Code Interpreter operates inside a Docker sandbox and mounts the workspace
    try:
        code_tool = CodeInterpreter(cfg={'work_dir': str(agent_pool.operation_manager.base_dir)})
        agent.function_map['code_interpreter'] = code_tool
    except Exception as e:
        logger.warning(f"Failed to load CodeInterpreter for agent {agent_name}: {e}")
        print(f"[WARNING] 🐳 CodeInterpreter disabled for '{agent_name}': {e}")
    
    edit_tool = EditFile()
    edit_tool.agent_pool = agent_pool
    edit_tool.agent_name = agent_name
    agent.function_map['edit_file'] = edit_tool
    
    list_tool = ListDir()
    list_tool.agent_pool = agent_pool
    agent.function_map['list_dir'] = list_tool
    
    grep_tool = Grep()
    grep_tool.agent_pool = agent_pool
    agent.function_map['grep'] = grep_tool
    
    delete_tool = DeleteFile()
    delete_tool.agent_pool = agent_pool
    delete_tool.agent_name = agent_name
    agent.function_map['delete_file'] = delete_tool
    
    copy_tool = CopyFile()
    copy_tool.agent_pool = agent_pool
    copy_tool.agent_name = agent_name
    agent.function_map['copy_file'] = copy_tool
    
    move_tool = MoveFile()
    move_tool.agent_pool = agent_pool
    move_tool.agent_name = agent_name
    agent.function_map['move_file'] = move_tool
    
    compress_tool = CompressContext()
    compress_tool.agent_pool = agent_pool
    compress_tool.agent_name = agent_name
    agent.function_map['compress_context'] = compress_tool
    
    shell_tool = ShellCmd()
    shell_tool.agent_pool = agent_pool
    shell_tool.agent_name = agent_name
    agent.function_map['shell_cmd'] = shell_tool
    
    # Inform the agent about the user-approval workflow
    agent.system_message += """
    
User Approval System:
- All mutating operations (file write, edit, delete, move, copy) require explicit user approval.
- When you call a tool like write_file or edit_file, the user will see a prompt and can approve or reject.
- If rejected, you'll receive the user's reason. Adjust your approach accordingly.
- Read operations (read_file, list_dir, grep, view_image) are free access.
"""
    
    return agent
def extract_sub_agent_feedback(messages: List[Dict], instance_name: str) -> str:
    """
    Extracts text output from sub-agent messages.
    Refined logic:
    1. Only include text generated AFTER the last tool call ended.
    2. If no tool calls were made, include all text from the beginning.
    3. If tool calls were made but no text follows, return a warning for the manager.
    """
    last_tool_idx = -1
    for i, msg in enumerate(messages):
        # Mark index of last message that involved a tool
        if msg.get(ROLE) == FUNCTION or msg.get('function_call'):
            last_tool_idx = i

    relevant_msgs = messages[last_tool_idx + 1:] if last_tool_idx != -1 else messages
    
    collected_text = []
    for msg in relevant_msgs:
        if isinstance(msg, dict):
            # Safe dot access imitation if msg is a dict
            msg_content = msg.get('content', '')
            msg_role = msg.get('role', '')
        else:
            msg_content = msg.content
            msg_role = msg.role

        if msg_role == ASSISTANT:
            text = extract_text_from_message(msg, add_upload_info=False)
            if text:
                collected_text.append(text)
    
    result_str = "\n\n".join(collected_text).strip()
    
    if not result_str:
        if last_tool_idx != -1:
            return f"WARNING: Sub-agent {instance_name} performed tool calls but provided no final summary or conclusion after the last operation."
        else:
            return f"Sub-agent {instance_name} finished but provided no text output."
            
    return result_str


class AgentInstanceLogger:
    """Handles persistent logging for an agent instance."""
    
    def __init__(self, agent_class: str, instance_name: str, log_dir: str):
        self.agent_class = agent_class
        self.instance_name = instance_name
        self.start_time = datetime.datetime.now()
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_class}_{instance_name}_{timestamp}.jsonl"
        self.log_path = os.path.join(log_dir, filename)
        
        self.data = {
            "metadata": {
                "agent_class": agent_class,
                "instance_name": instance_name,
                "start_timestamp": self.start_time.isoformat(),
            },
            "history": []
        }
        self._initial_save()

    def _format_message(self, message: Union[Dict, Any]) -> Dict:
        """Ensure message is a dict and has a timestamp."""
        if hasattr(message, 'model_dump'):  # For Pydantic-based Message
            msg_dict = message.model_dump()
        elif hasattr(message, 'to_dict'):
            msg_dict = message.to_dict()
        elif isinstance(message, dict):
            msg_dict = copy.deepcopy(message)
        else:
            # Fallback for generic objects or Message dataclass
            msg_dict = {}
            for k in ['role', 'content', 'name', 'function_call', 'extra']:
                if hasattr(message, k):
                    val = getattr(message, k)
                    if val is not None:
                        msg_dict[k] = val
            if not msg_dict and isinstance(message, str):
                msg_dict = {'role': 'unknown', 'content': message}
        
        # Add timestamp if missing
        if 'timestamp' not in msg_dict:
            msg_dict['timestamp'] = datetime.datetime.now().isoformat()
        
        return msg_dict

    def _append_line(self, data: Dict):
        """Append a single JSON line to the log file."""
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to append to agent log {self.log_path}: {e}")

    def _initial_save(self):
        """Write metadata as the first line."""
        self._append_line({"metadata": self.data["metadata"]})

    def log_message(self, message: Any):
        """Append a single message to history and file."""
        formatted_msg = self._format_message(message)
        self.data["history"].append(formatted_msg)
        self._append_line(formatted_msg)

    def update_history(self, history: List[Any]):
        """
        Additive sync for persistent logs (JSONL). 
        Only appends new messages found in `history` that aren't in the log yet.
        """
        old_history = self.data["history"]
        last_match_idx = -1  # Index in old_history
        
        for msg in history:
            formatted = self._format_message(msg)
            
            # Look for this message in the log AFTER the last matched message
            found = False
            # Check a reasonable range to find a match (e.g. up to 10 messages ahead)
            start_search = last_match_idx + 1
            for j in range(start_search, len(old_history)):
                potential_match = old_history[j]
                if potential_match.get('role') == formatted.get('role') and \
                   potential_match.get('content') == formatted.get('content'):
                    # Found a match!
                    last_match_idx = j
                    found = True
                    break
            
            if not found:
                # This is a truly new message — append it!
                old_history.append(formatted)
                last_match_idx = len(old_history) - 1
                self._append_line(formatted)

    def reset_history(self, new_history: List[Any]):
        """
        Update internal tracking after a compression event.
        
        The JSONL log file is APPEND-ONLY and preserves the full uncompressed
        history. This method only:
        1. Writes a compression marker so readers know compression happened
        2. Resets the internal data["history"] to the new compressed baseline
           so that subsequent update_history() calls match correctly
        
        It does NOT re-write compressed messages to the file.
        """
        import datetime as _dt
        # Write a visible marker so log readers know compression happened here
        self._append_line({
            "event": "COMPRESSION",
            "timestamp": _dt.datetime.now().isoformat(),
            "new_message_count": len(new_history),
            "message": "Context was compressed. Messages above are the full history. The agent now sees a compressed version."
        })
        
        # Reset internal tracking to the compressed baseline.
        # This is critical: update_history() does sequential matching against
        # self.data["history"]. After compression the in-memory history changed,
        # so we must update our tracking to match, otherwise it will re-append
        # all the remaining messages as "new".
        self.data["history"] = [self._format_message(msg) for msg in new_history]

