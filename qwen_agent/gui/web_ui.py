# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import pprint
import re
from typing import List, Optional, Union

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.tools import TOOL_REGISTRY
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import ASSISTANT, AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, SYSTEM, USER, VIDEO, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback


class WebUI:
    """A Common chatbot application for agent."""

    def __init__(self, agent: Union[Agent, MultiAgentHub, List[Agent]], chatbot_config: Optional[dict] = None):
        """
        Initialization the chatbot.

        Args:
            agent: The agent or a list of agents,
                supports various types of agents such as Assistant, GroupChat, Router, etc.
            chatbot_config: The chatbot configuration.
                Set the configuration as {'user.name': '', 'user.avatar': '', 'agent.avatar': '', 'input.placeholder': '', 'prompt.suggestions': []}.
        """
        chatbot_config = chatbot_config or {}

        if isinstance(agent, MultiAgentHub):
            self.agent_list = [agent for agent in agent.nonuser_agents]
            self.agent_hub = agent
        elif isinstance(agent, list):
            self.agent_list = agent
            self.agent_hub = None
        else:
            self.agent_list = [agent]
            self.agent_hub = None

        user_name = chatbot_config.get('user.name', 'user')
        self._last_active_sa = None

        self.user_config = {
            'name': user_name,
            'avatar': chatbot_config.get(
                'user.avatar',
                get_avatar_image(user_name),
            ),
        }

        self.agent_config_list = [{
            'name': agent.name,
            'avatar': chatbot_config.get(
                'agent.avatar',
                get_avatar_image(agent.name),
            ),
            'description': agent.description or "I'm a helpful assistant.",
        } for agent in self.agent_list]

        self.input_placeholder = chatbot_config.get('input.placeholder', '跟我聊聊吧～')
        self.prompt_suggestions = chatbot_config.get('prompt.suggestions', [])
        self.verbose = chatbot_config.get('verbose', False)
        
        # Store original function maps for tool toggling
        self.original_function_maps = {}
        for i, agent in enumerate(self.agent_list):
            if hasattr(agent, 'function_map') and agent.function_map:
                self.original_function_maps[i] = dict(agent.function_map)
        
        # Build a master list of ALL tools across all agents
        self.all_available_tools = set()
        for i, agent in enumerate(self.agent_list):
            if hasattr(agent, 'function_map') and agent.function_map:
                self.all_available_tools.update(agent.function_map.keys())
        
        # Add explicitly available tools from config
        config_available_tools = chatbot_config.get('available_tools', [])
        if config_available_tools:
            self.all_available_tools.update(config_available_tools)
            
        self.all_available_tools = sorted(list(self.all_available_tools))

    """
    Run the chatbot.

    Args:
        messages: The chat history.
    """

    def run(self,
            messages: List[Message] = None,
            share: bool = False,
            server_name: str = None,
            server_port: int = None,
            concurrency_limit: int = 10,
            enable_mention: bool = False,
            **kwargs):
        self.run_kwargs = kwargs

        from qwen_agent.gui.gradio_dep import gr, mgr, ms

        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        with gr.Blocks(
                css=os.path.join(os.path.dirname(__file__), 'assets/appBot.css'),
                theme=customTheme,
        ) as demo:
            history = gr.State([])
            sub_agent_history = gr.State([])  # Track sub-agent conversations
            
            with ms.Application():
                with gr.Row(elem_classes='container'):
                    with gr.Column(scale=3):  # Main chat - slightly smaller
                        chatbot = mgr.Chatbot(value=convert_history_to_chatbot(messages=messages),
                                              avatar_images=[
                                                  self.user_config,
                                                  self.agent_config_list,
                                              ],
                                              height=700,
                                              avatar_image_width=80,
                                              flushing=False,
                                              show_copy_button=True,
                                              label='Main Chat',
                                              latex_delimiters=[{
                                                  'left': '\\(',
                                                  'right': '\\)',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{equation}',
                                                  'right': '\\end{equation}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{align}',
                                                  'right': '\\end{align}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{alignat}',
                                                  'right': '\\end{alignat}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{gather}',
                                                  'right': '\\end{gather}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{CD}',
                                                  'right': '\\end{CD}',
                                                  'display': True
                                              }, {
                                                  'left': '\\[',
                                                  'right': '\\]',
                                                  'display': True
                                              }])

                        input = mgr.MultimodalInput(placeholder=self.input_placeholder,)
                        with gr.Row():
                            stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                            retry_btn = gr.Button("🔄 Retry", variant="secondary")
                            reset_btn = gr.Button("🗑️ Reset Chat", variant="danger")
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath"
                        )
                    
                    # Sub-agent conversation panel (visible when manager delegates)
                    with gr.Column(scale=2, visible=(len(self.agent_list) > 1)) as sub_agent_panel:
                        sub_chatbot = mgr.Chatbot(
                            value=[],
                            label='🔄 Sub-Agent Activity',
                            height=700,
                            avatar_image_width=60,
                            flushing=False,
                            show_copy_button=True,
                        )
                        sub_agent_status = gr.Textbox(
                            label='Status',
                            value='Ready',
                            interactive=False,
                            lines=2
                        )

                    with gr.Column(scale=1):
                        if len(self.agent_list) > 1:
                            agent_selector = gr.Dropdown(
                                [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                                label='Agents',
                                info='Agent',
                                value=0,
                                interactive=True,
                            )
                        else:
                            agent_selector = gr.State(0)

                        session_name = gr.Textbox(
                            label='Session Name',
                            placeholder='Enter session identifier (e.g. CoderSession)',
                            value='MainSession',
                            interactive=True,
                            elem_id="session_name_input"
                        )

                        agent_info_block = self._create_agent_info_block()

                        if self.prompt_suggestions:
                            gr.Examples(
                                label='Suggestions',
                                examples=self.prompt_suggestions,
                                inputs=[input],
                            )
                        
                        # --- User Approval Panel (polls for blocking approvals) ---
                        with gr.Accordion("🛡️ Pending Approvals", open=True, visible=False, elem_id="approval_panel") as approval_accordion:
                            approval_id_list = gr.Dropdown(
                                label="Select Request",
                                choices=[],
                                interactive=True
                            )
                            approval_details = gr.Markdown("No pending requests.")
                            with gr.Row():
                                approve_btn = gr.Button("✅ Approve", variant="primary")
                                reject_btn = gr.Button("❌ Reject", variant="stop")
                            reject_reason_input = gr.Textbox(
                                label="Rejection reason (required)",
                                placeholder="Why are you rejecting this operation?",
                                visible=False,
                                lines=2,
                            )
                            # Control settings for approvals
                            timeout_toggle = gr.Checkbox(label="Enable 5-minute AFK auto-reject timeout (if unchecked, it will wait forever)", value=True)
                            
                            # Timer to poll for pending approvals every 1 second
                            approval_timer = gr.Timer(value=1, active=True)
                        
                        agent_plugins_block = self._create_agent_plugins_block()

                        # Add event handler for tool toggle
                        if len(self.agent_list) > 1:
                            agent_plugins_block.change(
                                fn=self.toggle_tools,
                                inputs=[agent_plugins_block, agent_selector],
                                outputs=[agent_plugins_block],
                                queue=False,
                            )
                        else:
                            # Single agent case - use constant 0 for agent index
                            agent_plugins_block.change(
                                fn=self.toggle_tools,
                                inputs=[agent_plugins_block],
                                outputs=[agent_plugins_block],
                                queue=False,
                            )

                    if len(self.agent_list) > 1:
                        agent_selector.change(
                            fn=self.change_agent,
                            inputs=[agent_selector],
                            outputs=[agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )

                    input_promise = input.submit(
                        fn=self.add_text,
                        inputs=[input, audio_input, chatbot, history],
                        outputs=[input, audio_input, chatbot, history],
                        queue=True,
                    )

                    if len(self.agent_list) > 1 and enable_mention:
                        input_promise = input_promise.then(
                            self.add_mention,
                            [chatbot, agent_selector],
                            [chatbot, agent_selector],
                            queue=True,
                        ).then(
                            self.agent_run,
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                            queue=True,
                        )
                    elif len(self.agent_list) > 1:
                        # Multiple agents but mention disabled - still pass agent_selector
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status, session_name],
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                            queue=True,
                        )
                    else:
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history, session_name],
                            [chatbot, history],
                            queue=True,
                        )

                    input_promise.then(self.flushed, None, [input])

                    # --- Event Handlers for Approvals ---
                    # Timer-based polling for pending approvals
                    approval_timer.tick(
                        fn=self._update_approval_list,
                        outputs=[approval_id_list, approval_details, approval_accordion]
                    )

                    approval_id_list.change(
                        fn=self._update_approval_details,
                        inputs=[approval_id_list],
                        outputs=[approval_details, reject_reason_input]
                    )

                    approve_btn.click(
                        fn=self._handle_approve,
                        inputs=[approval_id_list],
                        outputs=[approval_id_list, approval_details, approval_accordion]
                    )

                    reject_btn.click(
                        fn=self._handle_reject,
                        inputs=[approval_id_list, reject_reason_input],
                        outputs=[approval_id_list, approval_details, approval_accordion, reject_reason_input]
                    )
                    
                    timeout_toggle.change(
                        fn=self._handle_timeout_toggle,
                        inputs=[timeout_toggle],
                        outputs=None
                    )

                    # --- Event Handlers for Stop/Retry/Reset ---
                    stop_btn.click(
                        fn=self.stop_chat,
                        inputs=[agent_selector],
                        outputs=None,
                        cancels=[input_promise],
                        queue=False
                    )
                    
                    retry_promise = retry_btn.click(
                        fn=self.retry_chat,
                        inputs=[chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                        outputs=[chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                    )
                    
                    reset_btn.click(
                        fn=self.reset_chat,
                        inputs=[agent_selector],
                        outputs=[chatbot, history, sub_chatbot, sub_agent_status],
                        queue=False,
                    )

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share,
                                                                       server_name=server_name,
                                                                       server_port=server_port)

    def _sanitize_content(self, text: str) -> str:
        """Prevent Gradio crash by disabling links to local directories."""
        if not text or not isinstance(text, str):
            return text
            
        def is_dir(path):
            try:
                # Remove common local prefixes
                clean_path = path
                if clean_path.startswith('file://'):
                    clean_path = clean_path[7:]
                
                # Strip leading slash on Windows if it's like /C:/
                if os.name == 'nt' and clean_path.startswith('/') and len(clean_path) > 2 and clean_path[2] == ':':
                    clean_path = clean_path[1:]

                return os.path.isdir(clean_path)
            except:
                return False

        # 1. Handle explicit markdown links [label](path)
        def replace_dir_link(match):
            label = match.group(1)
            path = match.group(2)
            if is_dir(path):
                return f"📂 **{label}** (Directory: `{path}`)"
            return match.group(0)

        # Regex for markdown links: [label](path)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_dir_link, text)
        
        # 2. Handle raw absolute paths that might be auto-linkified by ModelScope
        def replace_raw_dir(match):
            path = match.group(0)
            # Basic punctuation cleanup (don't include trailing dots/commas in the path check)
            punct = ""
            while path and path[-1] in '.,;:!?)]':
                punct = path[-1] + punct
                path = path[:-1]
                
            if is_dir(path):
                return f"`{path}`{punct}"
            return match.group(0)

        # Regex for Windows paths (C:\...) or Unix-style absolute paths ( /... )
        raw_path_pattern = r'(?:[a-zA-Z]:\\[^\s"\'<>|]+|/(?:[^/\s"\'<>|]+/)+[^\s"\'<>|]*)'
        sanitized = re.sub(raw_path_pattern, replace_raw_dir, text)
        
        return sanitized

    def change_agent(self, agent_selector):
        # Restore original function map when switching agents
        if agent_selector in self.original_function_maps:
            self.agent_list[agent_selector].function_map = self.original_function_maps[agent_selector]
        yield agent_selector, self._create_agent_info_block(agent_selector), self._get_agent_tools_update(
            agent_selector)

    def toggle_tools(self, selected_tools, agent_selector=None):
        """Update the agent's available tools based on user selection with cross-agent discovery."""
        if agent_selector is None:
            agent_selector = 0
        
        agent = self.agent_list[agent_selector]
        if hasattr(agent, 'function_map'):
            new_map = {}
            for t_name in selected_tools:
                # 1. Try to restore from this agent's original map
                if agent_selector in self.original_function_maps and t_name in self.original_function_maps[agent_selector]:
                    new_map[t_name] = self.original_function_maps[agent_selector][t_name]
                # 2. Try to discover from other agents
                else:
                    discovered_tool = None
                    for other_idx, other_map in self.original_function_maps.items():
                        if t_name in other_map:
                            # Found it! Copy the tool instance
                            discovered_tool = copy.copy(other_map[t_name])
                            
                            # Reconfigure for current agent if it has an agent_name attribute
                            if hasattr(discovered_tool, 'agent_name'):
                                discovered_tool.agent_name = agent.name
                            break
                    
                    if discovered_tool:
                        new_map[t_name] = discovered_tool
                    # 3. Last resort: Instantiate from registry
                    elif t_name in TOOL_REGISTRY:
                        try:
                            # Create new instance of the tool
                            tool_class = TOOL_REGISTRY[t_name]
                            new_tool = tool_class()
                            
                            # Configure for current agent if it has an agent_name attribute
                            if hasattr(new_tool, 'agent_name'):
                                new_tool.agent_name = agent.name
                            
                            new_map[t_name] = new_tool
                        except Exception as e:
                            logger.error(f"Failed to dynamically instantiate tool {t_name}: {e}")
            
            # If no tools selected, Qwen framework usually wants all restored or empty?
            # User expectation: if I uncheck everything, agent has no tools.
            # But if I check one, it should have ONLY that one.
            agent.function_map = new_map
            
        yield self._get_agent_tools_update(agent_selector)

    def add_text(self, _input, _audio_input, _chatbot, _history):
        _history.append({
            ROLE: USER,
            CONTENT: [{
                'text': _input.text
            }],
        })

        if self.user_config[NAME]:
            _history[-1][NAME] = self.user_config[NAME]
        
        # if got audio from microphone, append it to the multimodal inputs
        if _audio_input:
            from qwen_agent.gui.gradio_dep import gr, mgr, ms
            audio_input_file = gr.data_classes.FileData(path=_audio_input, mime_type="audio/wav")
            _input.files.append(audio_input_file)

        if _input.files:
            for file in _input.files:
                if file.mime_type.startswith('image/'):
                    _history[-1][CONTENT].append({IMAGE: 'file://' + file.path})
                elif file.mime_type.startswith('audio/'):
                    _history[-1][CONTENT].append({AUDIO: 'file://' + file.path})
                elif file.mime_type.startswith('video/'):
                    _history[-1][CONTENT].append({VIDEO: 'file://' + file.path})
                else:
                    _history[-1][CONTENT].append({FILE: file.path})

        _chatbot.append([_input, None])

        from qwen_agent.gui.gradio_dep import gr

        yield gr.update(interactive=False, value=None), None, _chatbot, _history

    def add_mention(self, _chatbot, _agent_selector):
        if len(self.agent_list) == 1:
            yield _chatbot, _agent_selector

        query = _chatbot[-1][0].text
        match = re.search(r'@\w+\b', query)
        if match:
            _agent_selector = self._get_agent_index_by_name(match.group()[1:])

        agent_name = self.agent_list[_agent_selector].name

        if ('@' + agent_name) not in query and self.agent_hub is None:
            _chatbot[-1][0].text = '@' + agent_name + ' ' + query

        yield _chatbot, _agent_selector

    def reset_chat(self, _agent_selector=None):
        """Reset the conversation state."""
        # Clear main history and chatbot
        _history = []
        _chatbot = []
        _sub_chatbot = []
        _sub_status = "Ready"

        # If we have an OrchestratorAgent, clearing the internal history is important
        if _agent_selector is not None:
            agent = self.agent_list[_agent_selector]
            _agent_pool = getattr(agent, 'agent_pool', None)
            if _agent_pool:
                # Full reset of all sub-agent instances and loggers in the pool
                _agent_pool.reset()

        return _chatbot, _history, _sub_chatbot, _sub_status

    def stop_chat(self, _agent_selector=None):
        """Signal all agents to stop execution."""
        agent_runner = self.agent_list[_agent_selector or 0]
        if self.agent_hub:
            agent_runner = self.agent_hub
            
        _agent_pool = getattr(agent_runner, 'agent_pool', None)
        if _agent_pool:
            logger.info("STOP button clicked. Signalling cancellation.")
            _agent_pool.stopped = True
        return None

    def retry_chat(self, _chatbot, _history, _agent_selector=None, _sub_chatbot=None, _sub_status=None):
        """Remove last response and re-run."""
        if not _history or len(_history) < 2:
            yield _chatbot, _history, _agent_selector, _sub_chatbot, _sub_status
            return

        # Remove the last ASSISTANT message and any trailing metadata
        # History is typically [{role: user, content: ...}, {role: assistant, content: ...}]
        if _history[-1][ROLE] == ASSISTANT:
            _history.pop()
        
        # Remove the last bubble from chatbot if it was an assistant response
        if _chatbot and _chatbot[-1][1] is not None:
            _chatbot.pop()

        # Re-run the generation
        yield from self.agent_run(_chatbot, _history, _agent_selector, _sub_chatbot, _sub_status)

    def agent_run(self, _chatbot, _history, _agent_selector=None, _sub_chatbot=None, _sub_status=None, _session_name="MainSession"):
        if self.verbose:
            logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

        # Capture expected structure at the start to ensure stable yield lengths
        has_sub = len(self.agent_list) > 1 or self.agent_hub is not None
        has_selector = len(self.agent_list) > 1
        
        num_input_bubbles = len(_chatbot) - 1
        num_output_bubbles = 1
        _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

        agent_runner = self.agent_list[_agent_selector or 0]
        if self.agent_hub:
            agent_runner = self.agent_hub
        
        # Apply session name if it's an Orchestrator
        if hasattr(agent_runner, 'session_name'):
            agent_runner.session_name = _session_name
        
        # Reset stop flag for new run
        _agent_pool = getattr(agent_runner, 'agent_pool', None)
        if _agent_pool:
            _agent_pool.stopped = False
            if hasattr(_agent_pool, 'sub_agent_state'):
                _agent_pool.sub_agent_state.clear()
        
        # Initialize sub-agent chat for new turn
        if has_sub:
            _sub_chatbot = []
            _sub_chatbot.append([None, f"🚀 {agent_runner.name} is working..."])
            if _sub_status:
                _sub_status = "Agent thinking..."
        
        # Track previously processed response count
        _prev_rsp_count = 0
        agent_index = _agent_selector or 0
        
        responses = []
        from qwen_agent.gui.gradio_dep import gr
        
        main_label = f"Main Chat: {agent_runner.__class__.__name__}"
        if hasattr(agent_runner, 'session_name'):
            main_label += f" ({agent_runner.session_name})"
        elif hasattr(agent_runner, 'name'):
            main_label += f" ({agent_runner.name})"
            
        sub_label_base = "🔄 Sub-Agent Activity"
        sub_label = sub_label_base

        import time
        last_yield_time = 0
        yield_interval = 0.1  # 10Hz throttle

        try:
            for responses in (agent_runner.run(_history, **self.run_kwargs) if hasattr(agent_runner, "run") else []):
                if not responses:
                    continue
                if responses[-1][CONTENT] == PENDING_USER_INPUT:
                    logger.info('Interrupted. Waiting for user input!')
                    if has_sub:
                        _sub_chatbot.append([None, "⏳ Waiting for your input..."])
                    break

                display_responses = convert_fncall_to_text(responses)
                if not display_responses:
                    continue
                if display_responses[-1][CONTENT] is None:
                    continue

                # Update sub-agent panel from agent_pool streaming state
                if has_sub:
                    # Try to read live streaming state from OrchestratorAgent's agent_pool
                    _agent_pool = getattr(agent_runner, 'agent_pool', None)
                    if _agent_pool and hasattr(_agent_pool, 'sub_agent_state'):
                        # Identify the sub-agent to display using the active_stack (recursive safe)
                        display_name = None
                        active_stack = getattr(_agent_pool, 'active_stack', [])
                        
                        if active_stack:
                            # Use the top of the stack (deepest active agent)
                            display_name = active_stack[-1]
                            self._last_active_sa = display_name
                        else:
                            # If no one is active according to stack, check if anyone is still marked active in state
                            # (Safety fallback)
                            for sa_name, sa_state in _agent_pool.sub_agent_state.items():
                                if sa_state.get('active'):
                                    display_name = sa_name
                                    self._last_active_sa = sa_name
                                    break
                        
                        # Fallback to last active if none are currently running
                        if not display_name:
                            display_name = self._last_active_sa

                        if display_name and display_name in _agent_pool.sub_agent_state:
                            sa_state = _agent_pool.sub_agent_state[display_name]
                            sub_label = f"Sub-Agent: {sa_state.get('agent_name', display_name)}"
                            messages = sa_state.get('messages', [])
                            
                            new_sub_chatbot = []
                            if messages:
                                formatted_msgs = convert_fncall_to_text(messages)
                                new_sub_chatbot.append([f"🤝 Full context for **{display_name}**", None])
                                
                                current_pair = [None, None]
                                for msg in formatted_msgs:
                                    role = msg.get('role')
                                    content = msg.get('content')
                                    if role == USER:
                                        if current_pair[0] is not None:
                                            new_sub_chatbot.append(current_pair)
                                            current_pair = [None, None]
                                        current_pair[0] = content
                                    elif role == ASSISTANT:
                                        current_pair[1] = content
                                        new_sub_chatbot.append(current_pair)
                                        current_pair = [None, None]
                                
                                if current_pair[0] is not None or current_pair[1] is not None:
                                    new_sub_chatbot.append(current_pair)
                            
                            if sa_state.get('active') and _sub_status:
                                _sub_status = f"{display_name} is responding..."
                            
                            _sub_chatbot = new_sub_chatbot
                        else:
                            sub_label = sub_label_base
                    else:
                        # Fallback for non-OrchestratorAgent
                        new_responses = responses[_prev_rsp_count:]
                        for rsp in new_responses:
                            role = rsp.get('role', '')
                            fn_call = rsp.get('function_call')
                            if role == 'assistant' and fn_call:
                                tool_name = fn_call.get('name', 'tool') if isinstance(fn_call, dict) else getattr(fn_call, 'name', 'tool')
                                _sub_chatbot.append([f"🔧 {tool_name}", "Calling..."])
                            elif role == 'function':
                                tool_name = rsp.get('name', 'tool')
                                result_preview = str(rsp.get('content', ''))[:200]
                                for i in range(len(_sub_chatbot) - 1, -1, -1):
                                    if _sub_chatbot[i][0] and f"🔧 {tool_name}" in _sub_chatbot[i][0]:
                                        _sub_chatbot[i] = [f"✅ {tool_name}", result_preview or "Done"]
                                        break

                    _prev_rsp_count = len(responses)

                while len(display_responses) > num_output_bubbles:
                    _chatbot.append([None, None])
                    _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]
                    num_output_bubbles += 1

                assert num_output_bubbles == len(display_responses)
                assert num_input_bubbles + num_output_bubbles == len(_chatbot)

                for i, rsp in enumerate(display_responses):
                    agent_index = self._get_agent_index_by_name(rsp[NAME])
                    sanitized_content = self._sanitize_content(rsp[CONTENT])
                    _chatbot[num_input_bubbles + i][1][agent_index] = sanitized_content

                if has_selector:
                    _agent_selector = agent_index

                # Sanitize sub-chatbot content
                if has_sub:
                    for bubble in _sub_chatbot:
                        if bubble[0]:
                            bubble[0] = self._sanitize_content(bubble[0])
                        if bubble[1]:
                            bubble[1] = self._sanitize_content(bubble[1])

                # Throttled Yield based on initial parameters
                current_time = time.time()
                if current_time - last_yield_time > yield_interval:
                    last_yield_time = current_time
                    if has_sub:
                        if has_selector:
                            yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history, _agent_selector, gr.update(value=copy.deepcopy(_sub_chatbot), label=sub_label), _sub_status
                        else:
                            yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history, gr.update(value=copy.deepcopy(_sub_chatbot), label=sub_label), _sub_status
                    else:
                        if has_selector:
                            yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history, _agent_selector
                        else:
                            yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"⚠️ **Model or Service Error:**\n```\n{str(e)}\n```\n\n*Please check your LLM configuration or ensure the model is loaded.*"
            if _chatbot and not getattr(_chatbot[-1][1][-1] if isinstance(_chatbot[-1][1], list) else (_chatbot[-1][1] or ''), 'strip', lambda: '')():
                # Replace empty bubble
                if isinstance(_chatbot[-1][1], list):
                    _chatbot[-1][1][-1] = error_msg
                else:
                    _chatbot[-1][1] = error_msg
            else:
                _chatbot.append((None, error_msg))
            
            if has_sub:
                if has_selector:
                    yield gr.update(value=_chatbot, label=main_label), _history, _agent_selector, gr.update(value=_sub_chatbot, label="⚠️ Error"), "Failed"
                else:
                    yield gr.update(value=_chatbot, label=main_label), _history, gr.update(value=_sub_chatbot, label="⚠️ Error"), "Failed"
            else:
                if has_selector:
                    yield gr.update(value=_chatbot, label=main_label), _history, _agent_selector
                else:
                    yield gr.update(value=_chatbot, label=main_label), _history

        if responses:
            _history.extend([res for res in responses if res[CONTENT] != PENDING_USER_INPUT])

        # Check if the Orchestrator's internal messages array was compressed mid-turn
        if hasattr(agent_runner, 'turn_final_messages') and agent_runner.turn_final_messages:
            if len(agent_runner.turn_final_messages) < len(_history):
                logger.info("Orchestrator history was compressed. Syncing WebUI history.")
                _history.clear()
                for res in agent_runner.turn_final_messages:
                    msg = res.model_dump() if not isinstance(res, dict) else res
                    # Strip system messages — agent.py:run() dynamically prepends
                    # the system message on every turn, and the compression summary
                    # is already merged into the AgentPool history's system message.
                    if msg.get(ROLE) == SYSTEM:
                        continue
                    _history.append(msg)
            # Clear so it doesn't carry over
            agent_runner.turn_final_messages = None
        # Final update to sub-agent chat
        if has_sub:
            _sub_chatbot.append([None, f"✅ {agent_runner.name} completed!"])
            if _sub_status:
                _sub_status = "Ready"

        # Stable Final Yield
        if has_sub:
            if has_selector:
                yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history, _agent_selector, gr.update(value=copy.deepcopy(_sub_chatbot), label=sub_label_base), _sub_status
            else:
                yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history, gr.update(value=copy.deepcopy(_sub_chatbot), label=sub_label_base), _sub_status
        else:
            if has_selector:
                yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history, _agent_selector
            else:
                yield gr.update(value=copy.deepcopy(_chatbot), label=main_label), _history

        if self.verbose:
            logger.info('agent_run response:\n' + pprint.pformat(responses, indent=2))

    def flushed(self):
        from qwen_agent.gui.gradio_dep import gr

        return gr.update(interactive=True)

    def _get_agent_index_by_name(self, agent_name):
        if agent_name is None:
            return 0

        try:
            agent_name = agent_name.strip()
            for i, agent in enumerate(self.agent_list):
                if agent.name == agent_name:
                    return i
            return 0
        except Exception:
            print_traceback()
            return 0

    def _create_agent_info_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_config_interactive = self.agent_config_list[agent_index]

        return gr.HTML(
            format_cover_html(
                bot_name=agent_config_interactive['name'],
                bot_description=agent_config_interactive['description'],
                bot_avatar=agent_config_interactive['avatar'],
            ))

    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_interactive = self.agent_list[agent_index]

        # Show ALL available tools across all agents, not just current agent's tools
        # This allows users to freely enable/disable any tool for any agent
        all_tools = self.all_available_tools
        
        # Get currently enabled tools for this agent
        if hasattr(agent_interactive, 'function_map'):
            enabled_tools = list(agent_interactive.function_map.keys())
        else:
            enabled_tools = []
        
        return gr.CheckboxGroup(
            label='Tools (enable/disable freely)',
            value=enabled_tools,
            choices=all_tools,  # Show all tools from all agents
            interactive=True,
        )

    def _get_agent_tools_update(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr
        agent_interactive = self.agent_list[agent_index]
        all_tools = self.all_available_tools
        if hasattr(agent_interactive, 'function_map'):
            enabled_tools = list(agent_interactive.function_map.keys())
        else:
            enabled_tools = []
        return gr.update(value=enabled_tools, choices=all_tools)

    # --- User Approval Helper Methods ---

    def _get_operation_manager(self):
        """Find the OperationManager from the agent pool."""
        for agent in self.agent_list:
            if hasattr(agent, 'agent_pool') and agent.agent_pool:
                return agent.agent_pool.operation_manager
        return None

    def _update_approval_list(self):
        """Poll for pending approvals and update the UI."""
        from qwen_agent.gui.gradio_dep import gr
        manager = self._get_operation_manager()
        if not manager:
            return gr.update(choices=[], value=None), "No approval system.", gr.update(visible=False)

        pending = manager.list_pending_approvals()
        choices = [
            (f"⏳ {op['agent_name']}: {op['tool_name']} - {op['description']}", op['request_id'])
            for op in pending
        ]

        if not choices:
            return gr.update(choices=[], value=None), "No pending requests.", gr.update(visible=False, open=False)

        # Auto-select first if only one
        first_id = choices[0][1] if len(choices) == 1 else None
        return gr.update(choices=choices, value=first_id), gr.update(), gr.update(visible=True, open=True)

    def _update_approval_details(self, request_id):
        """Show details for a specific pending approval."""
        from qwen_agent.gui.gradio_dep import gr
        if not request_id:
            return "Select a request to see details.", gr.update(visible=False)

        manager = self._get_operation_manager()
        if not manager:
            return "Error: OperationManager not found.", gr.update(visible=False)

        # Find the pending approval
        pending = manager.list_pending_approvals()
        req = None
        for p in pending:
            if p['request_id'] == request_id:
                req = p
                break

        if not req:
            return f"Request {request_id} not found or already resolved.", gr.update(visible=False)

        details = f"### 🛡️ Approval Required\n\n"
        details += f"**Agent:** `{req['agent_name']}`\n\n"
        details += f"**Tool:** `{req['tool_name']}`\n\n"
        details += f"**Description:** {req['description']}\n\n"
        details += "**Parameters:**\n"
        details += f"```json\n{json.dumps(req['tool_args'], indent=2)}\n```\n"

        return details, gr.update(visible=True)

    def _handle_approve(self, request_id):
        """User approves a pending operation."""
        from qwen_agent.gui.gradio_dep import gr
        if not request_id:
            return gr.update(), "No request selected.", gr.update()

        manager = self._get_operation_manager()
        if not manager:
            return gr.update(), "Error: OperationManager not found.", gr.update()

        result = manager.user_approve(request_id)

        # Refresh the list
        return self._update_approval_list()

    def _handle_reject(self, request_id, reason):
        """User rejects a pending operation with a reason."""
        from qwen_agent.gui.gradio_dep import gr
        if not request_id:
            return gr.update(), "No request selected.", gr.update(), gr.update()

        if not reason or not reason.strip():
            return gr.update(), "⚠️ **Please provide a reason for rejection.**", gr.update(), gr.update()

        manager = self._get_operation_manager()
        if not manager:
            return gr.update(), "Error: OperationManager not found.", gr.update(), gr.update()

        result = manager.user_reject(request_id, reason.strip())

        # Refresh the list and clear the reason input
        choices_update, details_update, accordion_update = self._update_approval_list()
        return choices_update, details_update, accordion_update, gr.update(value="", visible=False)

    def _handle_timeout_toggle(self, enabled):
        """Update the AFK timeout setting in OperationManager."""
        manager = self._get_operation_manager()
        if manager:
            manager.enable_timeout = enabled
