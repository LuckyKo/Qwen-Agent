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
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import ASSISTANT, AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, USER, VIDEO, Message
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
                                info='选择一个Agent',
                                value=0,
                                interactive=True,
                            )

                        agent_info_block = self._create_agent_info_block()

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

                        if self.prompt_suggestions:
                            gr.Examples(
                                label='推荐对话',
                                examples=self.prompt_suggestions,
                                inputs=[input],
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
                        queue=False,
                    )

                    if len(self.agent_list) > 1 and enable_mention:
                        input_promise = input_promise.then(
                            self.add_mention,
                            [chatbot, agent_selector],
                            [chatbot, agent_selector],
                        ).then(
                            self.agent_run,
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                        )
                    elif len(self.agent_list) > 1:
                        # Multiple agents but mention disabled - still pass agent_selector
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                            [chatbot, history, agent_selector, sub_chatbot, sub_agent_status],
                        )
                    else:
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history],
                            [chatbot, history],
                        )

                    input_promise.then(self.flushed, None, [input])

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share,
                                                                       server_name=server_name,
                                                                       server_port=server_port)

    def change_agent(self, agent_selector):
        # Restore original function map when switching agents
        if agent_selector in self.original_function_maps:
            self.agent_list[agent_selector].function_map = self.original_function_maps[agent_selector]
        yield agent_selector, self._create_agent_info_block(agent_selector), self._create_agent_plugins_block(
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
            
            # If no tools selected, Qwen framework usually wants all restored or empty?
            # User expectation: if I uncheck everything, agent has no tools.
            # But if I check one, it should have ONLY that one.
            agent.function_map = new_map
            
        yield self._create_agent_plugins_block(agent_selector)

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

    def agent_run(self, _chatbot, _history, _agent_selector=None, _sub_chatbot=None, _sub_status=None):
        if self.verbose:
            logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

        num_input_bubbles = len(_chatbot) - 1
        num_output_bubbles = 1
        _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

        agent_runner = self.agent_list[_agent_selector or 0]
        if self.agent_hub:
            agent_runner = self.agent_hub
        
        # Initialize sub-agent chat if we have a sub-chatbot
        if _sub_chatbot is not None:
            _sub_chatbot = _sub_chatbot or []
            _sub_chatbot.append([None, f"🚀 {agent_runner.name} is working..."])
            if _sub_status:
                _sub_status = "Agent thinking..."
        
        # Track previously processed response count and sub-chatbot length 
        # to avoid re-processing old messages and spamming Gradio with unchanged state
        _prev_rsp_count = 0
        _prev_sub_len = len(_sub_chatbot) if _sub_chatbot is not None else 0
        
        responses = []
        for responses in agent_runner.run(_history, **self.run_kwargs):
            if not responses:
                continue
            if responses[-1][CONTENT] == PENDING_USER_INPUT:
                logger.info('Interrupted. Waiting for user input!')
                if _sub_chatbot is not None:
                    _sub_chatbot.append([None, "⏳ Waiting for your input..."])
                break

            display_responses = convert_fncall_to_text(responses)
            if not display_responses:
                continue
            if display_responses[-1][CONTENT] is None:
                continue

            # Update sub-agent panel from agent_pool streaming state
            if _sub_chatbot is not None:
                # Try to read live streaming state from OrchestratorAgent's agent_pool
                _agent_pool = getattr(agent_runner, 'agent_pool', None)
                if _agent_pool and hasattr(_agent_pool, 'sub_agent_state'):
                    new_sub_chatbot = []
                    
                    # Find the deeply active sub-agent (the last one marked active, or the most recent)
                    active_sa_name = None
                    active_sa_state = None
                    for sa_name, sa_state in _agent_pool.sub_agent_state.items():
                        if sa_state.get('active'):
                            active_sa_name = sa_name
                            active_sa_state = sa_state
                    
                    # If none are currently 'active' (e.g. they just finished), show the very last one
                    if not active_sa_name and _agent_pool.sub_agent_state:
                        active_sa_name, active_sa_state = list(_agent_pool.sub_agent_state.items())[-1]
                        
                    if active_sa_name and active_sa_state:
                        messages = active_sa_state.get('messages', [])
                        if messages:
                            # Apply identical formatting as main chat
                            formatted_msgs = convert_fncall_to_text(messages)
                            
                            # Add a header for the specific sub-agent
                            new_sub_chatbot.append([f"🤝 Conversation with **{active_sa_name}**", None])
                            
                            # Convert messages to Chatbot [[user, assistant], ...] format
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
                            
                            if _sub_status and active_sa_state.get('active'):
                                _sub_status = f"{active_sa_name} is responding..."

                    _sub_chatbot = new_sub_chatbot
                else:
                    # Fallback for non-OrchestratorAgent: show main agent tool calls
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
                # Create a new chat bubble
                _chatbot.append([None, None])
                _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]
                num_output_bubbles += 1

            assert num_output_bubbles == len(display_responses)
            assert num_input_bubbles + num_output_bubbles == len(_chatbot)

            for i, rsp in enumerate(display_responses):
                agent_index = self._get_agent_index_by_name(rsp[NAME])
                _chatbot[num_input_bubbles + i][1][agent_index] = rsp[CONTENT]

            if len(self.agent_list) > 1:
                _agent_selector = agent_index

            # Yield with sub-chatbot updates
            if _sub_chatbot is not None:
                if _agent_selector is not None:
                    yield _chatbot, _history, _agent_selector, _sub_chatbot, _sub_status
                else:
                    yield _chatbot, _history, _sub_chatbot, _sub_status
            else:
                if _agent_selector is not None:
                    yield _chatbot, _history, _agent_selector
                else:
                    yield _chatbot, _history

        if responses:
            _history.extend([res for res in responses if res[CONTENT] != PENDING_USER_INPUT])

        # Final update to sub-agent chat
        if _sub_chatbot is not None:
            _sub_chatbot.append([None, f"✅ {agent_runner.name} completed!"])
            if _sub_status:
                _sub_status = "Ready"

        if _sub_chatbot is not None:
            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector, _sub_chatbot, _sub_status
            else:
                yield _chatbot, _history, _sub_chatbot, _sub_status
        else:
            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:
                yield _chatbot, _history

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
