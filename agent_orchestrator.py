"""
Agent Orchestrator - An agent that can call other agents dynamically
Each sub-agent has its own soul.md, context, and specialized job.

Operation Approval System:
- Sub-agents request operations with context
- Manager can approve, reject, or suggest modifications
- Supports back-and-forth conversation
- Maintains conversation state for efficient context switching
"""

import copy
import os
import json
from qwen_agent.agents import Assistant
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
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
from operation_manager import OperationManager, OperationType
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
    ApproveOperation,
    RejectOperation,
    AskAgent,
    RespondToManager,
    ListPendingOperations,
    DismissAgent,
)


class AgentPool:
    """Manages a pool of specialized sub-agents."""
    
    def __init__(self, llm_cfg: dict, agents_dir: str = 'agents'):
        self.llm_cfg = llm_cfg
        self.agents_dir = Path(agents_dir)
        self.agents: Dict[str, Assistant] = {}
        self.agent_configs: Dict[str, dict] = {}
        
        # Operation manager for approval-based operations
        self.operation_manager = OperationManager('workspace')
        
        # Persistent conversation histories for each sub-agent
        self.agent_conversations: Dict[str, List] = {}
        
        # Live streaming state for WebUI (updated during sub-agent execution)
        self.sub_agent_state: Dict[str, dict] = {}
        
        # Auto-load all agents from the agents directory
        self._discover_agents()
    
    def _discover_agents(self):
        """Find and load all agent configurations from the agents directory."""
        if not self.agents_dir.exists():
            self.agents_dir.mkdir(exist_ok=True)
            # Create a default example agent
            self._create_example_agent()
        
        # Load all soul.md files
        for soul_file in self.agents_dir.glob('*/soul.md'):
            agent_name = soul_file.parent.name
            try:
                self.load_agent(agent_name)
                print(f"[OK] Loaded agent: {agent_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load agent {agent_name}: {e}")
    
    def _create_example_agent(self):
        """Create an example sub-agent."""
        example_dir = self.agents_dir / 'researcher'
        example_dir.mkdir(exist_ok=True)
        
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
        (example_dir / 'soul.md').write_text(soul_content)
    
    def load_agent(self, agent_name: str) -> Assistant:
        """Load or reload a specific agent with file tools."""
        soul_path = self.agents_dir / agent_name / 'soul.md'
        
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
    
    def get_conversation(self, agent_name: str) -> List:
        """Get or create persistent conversation history for an agent."""
        if agent_name not in self.agent_conversations:
            self.agent_conversations[agent_name] = []
        return self.agent_conversations[agent_name]
    
    def clear_conversation(self, agent_name: str):
        """Clear an agent's conversation history."""
        self.agent_conversations[agent_name] = []
        self.sub_agent_state.pop(agent_name, None)
    
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
        'Start a NEW conversation with a sub-agent. '
        'Clears any previous conversation context. '
        'Use this to delegate a fresh task.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'agent_name': {
                'type': 'string',
                'description': 'Name of the sub-agent (e.g. "coder", "researcher", "writer")'
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
        'required': ['agent_name', 'task'],
    },
}

CONTINUE_WITH_AGENT_SCHEMA = {
    'name': 'continue_with_agent',
    'description': (
        'Send a follow-up message to a sub-agent that already has conversation context. '
        'The sub-agent remembers everything from previous call_agent / continue_with_agent calls.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'agent_name': {
                'type': 'string',
                'description': 'Name of the sub-agent to continue with'
            },
            'message': {
                'type': 'string',
                'description': 'The follow-up message or instruction'
            },
        },
        'required': ['agent_name', 'message'],
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

    def __init__(self, agent_pool: AgentPool, **kwargs):
        super().__init__(**kwargs)
        self.agent_pool = agent_pool

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
        
        # --- Dynamically inject available sub-agents and tools ---
        if messages and messages[0].role == SYSTEM:
            # We want to modify the system message specifically for this run without
            # altering the base system message permanently.
            
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
            messages[0].content += system_append
        # --- Custom FnCallAgent-style loop with streaming sub-agent support ---
        messages = copy.deepcopy(messages)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response: List[Message] = []

        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']

            output_stream = self._call_llm(
                messages=messages,
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
                    tool_result = self._call_tool(
                        tool_name, tool_args, messages=messages, **kwargs
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
                response.append(fn_msg)
                yield response

            if not used_any_tool:
                break

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
        Generator that runs a sub-agent, yields intermediate results to the
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

        agent_name = args.get('agent_name', '')
        agent = self.agent_pool.get_agent(agent_name)
        if not agent:
            return (
                f"Error: Agent '{agent_name}' not found. "
                f"Available: {self.agent_pool.list_agents()}"
            )

        # Determine the user message for the sub-agent
        if tool_name == 'call_agent':
            # Fresh conversation — clear old context
            self.agent_pool.clear_conversation(agent_name)
            task = args.get('task', '')
            context = args.get('context', '')
            msg_text = (
                f'Context: {context}\n\nTask: {task}\n\nPlease help with this task.'
                if context
                else task
            )
        else:  # continue_with_agent
            msg_text = args.get('message', '')

        # --- Resolve multimodal content (Images) ---
        # If the task refers to images like [image_0], pass them to the sub-agent
        sub_agent_msg_content = [{'text': msg_text}]
        
        # Extract images from the manager's history to fulfill potential references
        seen_images = {}
        for msg in manager_history:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == IMAGE:
                        img_url = item.value
                        basename = get_basename_from_url(img_url)
                        seen_images[basename] = img_url
                        # Also support numeric indexing if the model uses it
                        idx = len([v for k, v in seen_images.items() if not k.startswith("image_")]) - 1
                        seen_images[f"image_{idx}"] = img_url

        # Check if the message text mentions any seen images or just pass all from last msg?
        # A robust way is to pass images that are likely relevant. 
        # For simplicity, we'll look for strings matching basenames or [image_N]
        added_to_sub = set()
        for ref, img_url in seen_images.items():
            if ref in msg_text and img_url not in added_to_sub:
                sub_agent_msg_content.append({IMAGE: img_url})
                added_to_sub.add(img_url)
        
        # Fallback: if no specific refs found but this is a call_agent, 
        # maybe pass images from the very last user message if they exist
        if len(sub_agent_msg_content) == 1 and tool_name == 'call_agent':
            last_user_msg = next((m for m in reversed(manager_history) if m.role == USER), None)
            if last_user_msg and isinstance(last_user_msg.content, list):
                for item in last_user_msg.content:
                    if item.type == IMAGE and item.value not in added_to_sub:
                        sub_agent_msg_content.append({IMAGE: item.value})
                        added_to_sub.add(item.value)

        conv = self.agent_pool.get_conversation(agent_name)
        conv.append({'role': 'user', 'content': sub_agent_msg_content})

        # Initialize streaming state for the WebUI
        state = {
            'active': True,
            'agent_name': agent_name,
            'messages': copy.deepcopy(conv),  # Initial conversation (User message)
        }
        self.agent_pool.sub_agent_state[agent_name] = state

        # Run the sub-agent as a generator
        final_resp: list = []
        try:
            for resp in agent.run(conv):
                final_resp = resp
                
                # Update streaming state with current conversation + latest assistant response
                if resp:
                    state['messages'] = conv + resp
                else:
                    state['messages'] = conv

                # Yield current_response unchanged — the WebUI reads 
                # sub_agent_state from the agent_pool for the sub-panel.
                yield current_response

        except Exception as e:
            state['active'] = False
            return f'Error calling agent {agent_name}: {e}'

        # Mark streaming complete
        state['active'] = False

        # Save sub-agent response to persistent conversation
        if final_resp:
            conv.extend(final_resp)
            result_content = final_resp[-1].get('content', '')
            if isinstance(result_content, list):
                # Handle content that's a list of ContentItem
                result_content = ' '.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in result_content
                )
            return f"[{agent_name}'s response]:\n{result_content or 'No content'}"

        return f"[{agent_name}]: No response generated"


# ─── Agent loading ─────────────────────────────────────────────────────────────

def load_orchestrator_agent(agent_pool: AgentPool, llm_cfg: dict) -> Assistant:
    """
    Load the orchestrator as an OrchestratorAgent with soul.md configuration
    and streaming sub-agent support.
    """
    soul_path = agent_pool.agents_dir / 'orchestrator' / 'soul.md'

    if soul_path.exists():
        from soul_loader import load_soul, build_system_prompt
        config = load_soul(str(soul_path))
        system_prompt = build_system_prompt(config)
    else:
        config = {}
        system_prompt = _default_orchestrator_prompt(agent_pool)

    # Create OrchestratorAgent directly
    agent = OrchestratorAgent(
        agent_pool=agent_pool,
        llm=llm_cfg,
        name=config.get('name', 'Orchestrator'),
        description=config.get('tagline', 'Supervisor agent that coordinates sub-agents'),
        system_message=system_prompt,
        function_list=[],
    )

    # Store config for agent selector UI
    agent.agent_configs = {config.get('name', 'orchestrator'): config}

    # ── Register sub-agent tools (intercepted in _run, not _call_tool) ──
    agent.function_map['call_agent'] = _SubAgentFunctionProxy(CALL_AGENT_SCHEMA)
    agent.function_map['continue_with_agent'] = _SubAgentFunctionProxy(CONTINUE_WITH_AGENT_SCHEMA)
    agent.function_map['dismiss_agent'] = DismissAgent(agent_pool=agent_pool)

    # ── Manager operation tools ──
    agent.function_map['approve_operation'] = ApproveOperation()
    agent.function_map['approve_operation'].agent_pool = agent_pool
    agent.function_map['approve_operation'].agent_name = 'orchestrator'

    agent.function_map['reject_operation'] = RejectOperation()
    agent.function_map['reject_operation'].agent_pool = agent_pool
    agent.function_map['reject_operation'].agent_name = 'orchestrator'

    agent.function_map['ask_agent'] = AskAgent()
    agent.function_map['ask_agent'].agent_pool = agent_pool
    agent.function_map['ask_agent'].agent_name = 'orchestrator'

    agent.function_map['list_pending_operations'] = ListPendingOperations(agent_pool=agent_pool)

    # ── File tools ──
    read_tool = ReadFile()
    read_tool.agent_pool = agent_pool
    agent.function_map['read_file'] = read_tool

    agent.function_map['list_dir'] = ListDir()
    agent.function_map['list_dir'].agent_pool = agent_pool

    agent.function_map['grep'] = Grep()
    agent.function_map['grep'].agent_pool = agent_pool

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
- call_agent: Start a NEW conversation with a sub-agent (clears old context)
- continue_with_agent: Send follow-up to a sub-agent using existing context
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
5. Manage operation approvals - use list_pending_operations to see requests, approve_operation to approve/reject

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
- Monitor pending operation requests and approve/reject as needed

File Operations (You have direct access):
- read_file: Read any file (free access). Large files are paginated.
- view_image: View an image file (free access). Returns the image for you to analyze.
- list_dir: List directory contents
- grep: Search for text patterns in files

Operation Management:
- You have manager privileges to approve/reject operations
- You can ask agents for clarification before deciding
- You can suggest modifications to operations
- Sub-agents can respond to your questions via conversation

Available Manager Tools:
- list_pending_operations: See all pending requests
- approve_operation: Approve with optional modifications
- reject_operation: Reject with reason
- ask_agent: Ask for clarification
- dismiss_agent: Clear a sub-agent's conversation history
"""

    # Create the orchestrator agent with manager tools
    orchestrator = Assistant(
        llm=llm_cfg,
        name='Orchestrator',
        description='Supervisor agent that coordinates with specialized sub-agents',
        system_message=system_instruction,
        function_list=['call_agent']  # Only use registered tools via function_list
    )

    # Register the tools manually (not via framework registry)
    orchestrator.function_map['call_agent'] = CallAgent(agent_pool=agent_pool)
    orchestrator.function_map['approve_operation'] = ApproveOperation()
    orchestrator.function_map['approve_operation'].agent_pool = agent_pool
    orchestrator.function_map['approve_operation'].agent_name = 'orchestrator'
    
    orchestrator.function_map['reject_operation'] = RejectOperation()
    orchestrator.function_map['reject_operation'].agent_pool = agent_pool
    orchestrator.function_map['reject_operation'].agent_name = 'orchestrator'
    
    orchestrator.function_map['ask_agent'] = AskAgent()
    orchestrator.function_map['ask_agent'].agent_pool = agent_pool
    orchestrator.function_map['ask_agent'].agent_name = 'orchestrator'

    orchestrator.function_map['list_pending_operations'] = ListPendingOperations()
    orchestrator.function_map['list_pending_operations'].agent_pool = agent_pool

    view_tool = ViewImage()
    view_tool.agent_pool = agent_pool
    orchestrator.function_map['view_image'] = view_tool

    return orchestrator


def load_sub_agent_with_tools(agent_pool: AgentPool, agent_name: str, llm_cfg: dict) -> Assistant:
    """
    Load a sub-agent from soul.md and give it file operation tools and the ability to spawn sub-agents recursively.
    """
    soul_path = agent_pool.agents_dir / agent_name / 'soul.md'
    if not soul_path.exists():
        raise FileNotFoundError(f"No soul.md found for agent: {agent_name}")
    
    # Load the agent as an OrchestratorAgent so it can spawn sub-agents
    agent, config = create_agent_from_soul(
        llm_cfg, 
        str(soul_path), 
        agent_class=OrchestratorAgent, 
        agent_pool=agent_pool
    )
    
    # Add recursive sub-agent tools (intercepted in _run)
    agent.function_map['call_agent'] = _SubAgentFunctionProxy(CALL_AGENT_SCHEMA)
    agent.function_map['continue_with_agent'] = _SubAgentFunctionProxy(CONTINUE_WITH_AGENT_SCHEMA)
    agent.function_map['dismiss_agent'] = DismissAgent(agent_pool=agent_pool)
    
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
    
    respond_tool = RespondToManager()
    respond_tool.agent_pool = agent_pool
    respond_tool.agent_name = agent_name
    agent.function_map['respond_to_manager'] = respond_tool
    
    # Update the agent's system message to mention abstract capabilities without heavily repeating specific tools 
    # (since the dynamic tool list handles the exact names and descriptions)
    agent.system_message += """
    
Operation Approval Workflow:
- If your request needs approval (like editing files you don't own), you'll get a request_id instead of immediate execution.
- The Manager may ask questions via the `ask_agent` tool.
- Use `respond_to_manager` to answer and continue the conversation.
- Manager can approve with modifications or suggest alternatives.
"""
    
    return agent
