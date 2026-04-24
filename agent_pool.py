"""
Agent Pool — Manages a pool of specialized sub-agents.

Handles agent discovery, loading, instance lifecycle, conversation persistence,
context compression, and streaming state for the WebUI.
"""

import copy
import json
import os
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qwen_agent.agents import Assistant
from qwen_agent.log import logger
from qwen_agent.llm.schema import (
    ASSISTANT, CONTENT, FUNCTION, ROLE, SYSTEM, USER, Message,
)
from qwen_agent.utils.utils import extract_text_from_message

from agent_logger import AgentInstanceLogger


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
        self.instance_loggers: Dict[str, AgentInstanceLogger] = {}
        
        # Live streaming state for WebUI (updated during sub-agent execution)
        self.sub_agent_state: Dict[str, dict] = {}
        
        # List of instance_names currently in an active call (stack for recursion)
        self.active_stack: List[str] = []
        
        # Caching for tool arguments to support __USE_PREV_ARG__
        # self.last_tool_args[instance_name][tool_name] = {arg_name: actual_value}
        self.last_tool_args: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Explicit stop flag for cancellation
        self.stopped = False
        
        # Async message queue for injecting user messages mid-generation
        self.async_message_queue: List[str] = []
        
        # Auto-load all agents from the agents directory
        self._discover_agents()
    
    def get_logger(self, instance_name: str, agent_class: str, base_metadata: Optional[Dict] = None) -> AgentInstanceLogger:
        """Get or create a logger for an agent instance."""
        if instance_name not in self.instance_loggers:
            # Ensure workspace/logs exists
            log_dir = Path('workspace/logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.instance_loggers[instance_name] = AgentInstanceLogger(
                agent_class=agent_class,
                instance_name=instance_name,
                log_dir=str(log_dir),
                base_metadata=base_metadata
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
        from agent_factory import load_sub_agent_with_tools
        
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
        self.active_stack.clear()
        self.last_tool_args.clear()
        logger.info("AgentPool reset — all instances and loggers cleared.")
    
    def load_session_from_log(self, log_input: str, target_instance: Optional[str] = None) -> str:
        """
        Load session history from a log entry (JSON string) or a log file path.
        Returns a status message.
        """
        log_input = log_input.strip()
        if not log_input:
            return "Error: Empty log input."

        messages = []
        metadata = {}
        
        # Try as file path first
        potential_path = Path(log_input)
        if potential_path.exists() and potential_path.is_file():
            try:
                with open(potential_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if "metadata" in item:
                                metadata.update(item["metadata"])
                            else:
                                messages.append(item)
                        except json.JSONDecodeError:
                            continue
                log_source = f"file '{potential_path.name}'"
            except Exception as e:
                return f"Error reading log file: {e}"
        else:
            # Try as JSON (single line or block)
            try:
                # Handle potential multiple JSON objects in one block (JSONL style)
                lines = log_input.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if "metadata" in item:
                            metadata.update(item["metadata"])
                        elif isinstance(item, list): # Full history block
                            messages.extend(item)
                        else:
                            messages.append(item)
                    except json.JSONDecodeError:
                        # Maybe it's a single large JSON block
                        if len(lines) == 1:
                            raise # Re-raise to try full-block parse
                        continue
                log_source = "JSON input"
            except json.JSONDecodeError:
                # Try parsing the whole thing as one JSON block
                try:
                    item = json.loads(log_input)
                    if isinstance(item, list):
                        messages = item
                    elif isinstance(item, dict) and "history" in item:
                        messages = item["history"]
                        if "metadata" in item:
                            metadata.update(item["metadata"])
                    else:
                        messages = [item]
                    log_source = "JSON block"
                except json.JSONDecodeError:
                    return "Error: Input is neither a valid file path nor a valid JSON."

        if not messages:
            return "Error: No valid messages found in log input."

        # Determine instance and class
        instance_name = target_instance or metadata.get("instance_name") or "RecoveredSession"
        agent_class = metadata.get("agent_class") or "Orchestrator"

        # Filter out event markers and ensure role/content exist
        cleaned_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if "event" in msg: # Skip COMPRESSION markers
                continue
            if ROLE in msg and CONTENT in msg:
                cleaned_messages.append(msg)

        if not cleaned_messages:
            return "Error: No valid conversation messages found."

        # Restore to pool
        self.instance_conversations[instance_name] = cleaned_messages
        self.instance_classes[instance_name] = agent_class
        
        # Proactively clear any existing logger for this instance so get_logger creates a fresh one
        self.instance_loggers.pop(instance_name, None)
        
        # Initialize a new logger for the continued session
        self.instance_loggers[instance_name] = self.get_logger(
            instance_name=instance_name,
            agent_class=agent_class,
            base_metadata=metadata
        )
        # Sync the loaded history to the NEW log file so it's persistent
        self.instance_loggers[instance_name].update_history(cleaned_messages)

        return f"Successfully loaded {len(cleaned_messages)} messages for instance '{instance_name}' ({agent_class}) from {log_source}."
    
    def _apply_context_compression(self, agent_name: str, summary: str, fraction: float, agent_obj=None):
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
        
        from qwen_agent.utils.tokenization_qwen import count_tokens
        
        # Calculate total tokens to find the actual fraction of content to compress
        total_tokens = 0
        token_counts = []
        for msg in messages_to_compress:
            tokens = agent_obj._count_message_tokens(msg) if agent_obj and hasattr(agent_obj, '_count_message_tokens') else 0
            if not tokens:
                # Fallback if agent_obj isn't provided
                from qwen_agent.utils.tokenization_qwen import count_tokens as qwen_count
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    function_call = msg.get('function_call')
                    if role == ASSISTANT and function_call:
                        tokens = qwen_count(f'{function_call}')
                    else:
                        content = extract_text_from_message(Message(**msg), add_upload_info=True)
                        tokens = qwen_count(content)
                else:
                    if msg.role == ASSISTANT and msg.function_call:
                        tokens = qwen_count(f'{msg.function_call}')
                    else:
                        content = extract_text_from_message(msg, add_upload_info=True)
                        tokens = qwen_count(content)
                        
            token_counts.append(tokens)
            total_tokens += tokens
            
        target_tokens = int(total_tokens * fraction)
        
        tokens_seen = 0
        num_to_remove = 0
        for count in token_counts:
            tokens_seen += count
            num_to_remove += 1
            if tokens_seen >= target_tokens and num_to_remove < len(messages_to_compress) - 1:
                break
                
        # Ensure we remove at least 1 message if possible
        num_to_remove = max(1, num_to_remove)
        
        # ADJUSTMENT: Ensure the first remaining message is a safe boundary.
        # Specifically, we scan forward from num_to_remove to find a message that is NOT a FUNCTION return.
        # However, we must NEVER remove the very last message in the history.
        found_safe = False
        temp_remove = num_to_remove
        while temp_remove < len(messages_to_compress):
            next_msg = messages_to_compress[temp_remove]
            role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', '')
            if role != FUNCTION:
                found_safe = True
                num_to_remove = temp_remove
                break
            temp_remove += 1
            
        # If we didn't find a safe message forward, scan BACKWARD.
        if not found_safe:
            temp_remove = num_to_remove - 1
            while temp_remove >= 0:
                next_msg = messages_to_compress[temp_remove]
                role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', '')
                if role != FUNCTION:
                    found_safe = True
                    num_to_remove = temp_remove
                    break
                temp_remove -= 1
        
        # If STILL none found, don't remove anything to avoid crashes.
        if not found_safe:
            logger.warning(f"Compression for {agent_name} could not find a safe boundary to start with. Skipping compression.")
            return
            
        if num_to_remove <= 0:
            return
            
        # Create the summary text
        summary_text = f"\n\n--- CONTEXT COMPRESSED ({int(fraction*100)}% of history summarized) ---\n\nSummary of previous context:\n{summary}\n\n--- END SUMMARY ---"
        
        # New history: [System (if any)] + [User (with summary)] + [Remaining Messages]
        new_history = []
        is_dict = isinstance(system_msg, dict) if system_msg else isinstance(messages_to_compress[0], dict)
        
        if system_msg:
            new_history.append(system_msg)
            
        if is_dict:
            new_history.append({'role': USER, 'content': str(summary_text)})
        else:
            new_history.append(Message(role=USER, content=str(summary_text)))
            
        new_history.extend(messages_to_compress[num_to_remove:])
        
        # Modify list in-place so active references (like 'conv' in _stream_sub_agent_call) remain valid!
        history.clear()
        history.extend(new_history)
        
        # Reset the logger's internal tracking to this new baseline.
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
