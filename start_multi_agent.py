"""
Multi-Agent Orchestrator Demo
A supervisor agent that can delegate tasks to specialized sub-agents.

Each sub-agent has its own:
- soul.md (personality & behavior)
- Specialized tools
- Domain expertise

File Permission System:
- Reads: Free access
- Writes to new/owned files: Auto-approved  
- Writes to others' files: Requires manager approval

The Orchestrator is also a launchable agent with its own soul.md!
"""

import json
import requests
from bs4 import BeautifulSoup
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from agent_orchestrator import AgentPool, load_orchestrator_agent

# Import built-in tools from qwen_agent
from qwen_agent.tools import (
    image_gen,
    web_extractor,
    storage,
    simple_doc_parser,
    doc_parser,
    extract_doc_vocabulary,
    amap_weather,
    code_interpreter,
)

# Register web tools globally
@register_tool('ddg_search', allow_overwrite=True)
class DDGSearch(BaseTool):
    name = 'ddg_search'
    description = 'Search for information from the internet using DuckDuckGo (No API key required).'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'The search query'
            }
        },
        'required': ['query'],
    }
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params['query']
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f'https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}'
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.select('.result')[:5]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                url_elem = result.select_one('.result__url')
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    url_text = url_elem.get_text(strip=True) if url_elem else ''
                    results.append(f'Title: {title}\nSnippet: {snippet}\nURL: {url_text}')
            if results:
                return '\n\n'.join(results)
            return 'No results found.'
        except Exception as e:
            return f'Search failed: {str(e)}'

# Configure for LM Studio (using a vision-capable model)
llm_cfg = {
    'model': 'whatever_is_on',  # or your preferred vision model
    'model_server': 'http://localhost:1234/v1',
    'api_key': 'EMPTY',
    'model_type': 'qwenvl_oai',  # Force multimodal support
    'max_input_tokens': 32768,   # Custom context window limit (override detection)
}

# Define default tools for each agent type
# All tools will be available in UI, but only these will be enabled by default
DEFAULT_TOOLS = {
    'orchestrator': [
        'call_agent', 'continue_with_agent', 'dismiss_agent',
        'approve_operation', 'reject_operation', 'ask_agent', 'respond_to_manager',
        'list_pending_operations', 'compress_context', 'write_file', 'edit_file', 'delete_file', 'copy_file', 'move_file', 'read_file', 'view_image', 'list_dir', 'grep',
        'ddg_search', 'web_extractor'
    ],
    'coder': [
        'call_agent', 'continue_with_agent', 'dismiss_agent',
        'approve_operation', 'reject_operation', 'ask_agent', 'respond_to_manager',
        'read_file', 'view_image', 'compress_context', 'write_file', 'edit_file', 'delete_file', 'copy_file', 'move_file', 'list_dir', 'grep', 'code_interpreter',
        'ddg_search', 'web_extractor'
    ],
    'researcher': [
        'call_agent', 'continue_with_agent', 'dismiss_agent',
        'approve_operation', 'reject_operation', 'ask_agent', 'respond_to_manager',
        'read_file', 'view_image', 'compress_context', 'write_file', 'edit_file', 'delete_file', 'copy_file', 'move_file', 'list_dir', 'grep', 'code_interpreter',
        'ddg_search', 'web_extractor',
        'doc_parser', 'simple_doc_parser', 'extract_doc_vocabulary'
    ],
    'writer': [
        'call_agent', 'continue_with_agent', 'dismiss_agent',
        'approve_operation', 'reject_operation', 'ask_agent', 'respond_to_manager',
        'read_file', 'view_image', 'compress_context', 'write_file', 'edit_file', 'list_dir',
        'ddg_search', 'web_extractor',
        'doc_parser', 'simple_doc_parser'
    ],
}

# Tools available to ALL agents (shown in UI but disabled by default)
ALL_BUILTIN_TOOLS = [
    'image_gen', 'storage', 'amap_weather', 'code_interpreter', 'python_executor',
    'delete_file', 'respond_to_manager'  # These are powerful - enable manually
]



if __name__ == '__main__':
    print("Initializing Agent Orchestrator...")
    print("=" * 50)

    # Create agent pool (auto-loads all agents from /agents directory)
    agent_pool = AgentPool(llm_cfg, 'agents')

    # Add tools to all agents based on their role
    for agent_name in agent_pool.list_agents():
        agent = agent_pool.get_agent(agent_name)
        if agent:
            # Get default tools for this agent type
            default_tools = DEFAULT_TOOLS.get(agent_name, DEFAULT_TOOLS['writer'])
            
            # Always add web tools
            agent.function_map['ddg_search'] = DDGSearch()
            
            # Add default built-in tools
            if 'image_gen' in default_tools:
                try:
                    agent.function_map['image_gen'] = image_gen.ImageGen(llm_cfg=llm_cfg)
                except Exception:
                    pass
            
            if 'web_extractor' in default_tools:
                agent.function_map['web_extractor'] = web_extractor.WebExtractor()
            
            if 'storage' in default_tools:
                agent.function_map['storage'] = storage.Storage()
            
            if 'simple_doc_parser' in default_tools:
                agent.function_map['simple_doc_parser'] = simple_doc_parser.SimpleDocParser()
            
            if 'doc_parser' in default_tools:
                agent.function_map['doc_parser'] = doc_parser.DocParser()
            
            if 'extract_doc_vocabulary' in default_tools:
                agent.function_map['extract_doc_vocabulary'] = extract_doc_vocabulary.ExtractDocVocabulary()
            
            if 'amap_weather' in default_tools:
                try:
                    agent.function_map['amap_weather'] = amap_weather.AmapWeather()
                except Exception:
                    pass
            
            if 'code_interpreter' in default_tools:
                try:
                    agent.function_map['code_interpreter'] = code_interpreter.CodeInterpreter(cfg={'work_dir': 'workspace'})
                except Exception:
                    pass
            
            if 'python_executor' in default_tools:
                try:
                    agent.function_map['python_executor'] = python_executor.PythonExecutor()
                except Exception:
                    pass

    # Load orchestrator with its default tools
    orchestrator = load_orchestrator_agent(agent_pool, llm_cfg)

    # Add web tools to orchestrator
    orchestrator.function_map['ddg_search'] = DDGSearch()

    # Add orchestrator's default built-in tools
    default_orch_tools = DEFAULT_TOOLS['orchestrator']
    if 'image_gen' in default_orch_tools:
        try:
            orchestrator.function_map['image_gen'] = image_gen.ImageGen(llm_cfg=llm_cfg)
        except Exception:
            pass

    if 'web_extractor' in default_orch_tools:
        orchestrator.function_map['web_extractor'] = web_extractor.WebExtractor()

    if 'storage' in default_orch_tools:
        orchestrator.function_map['storage'] = storage.Storage()

    if 'simple_doc_parser' in default_orch_tools:
        orchestrator.function_map['simple_doc_parser'] = simple_doc_parser.SimpleDocParser()

    if 'doc_parser' in default_orch_tools:
        orchestrator.function_map['doc_parser'] = doc_parser.DocParser()

    if 'extract_doc_vocabulary' in default_orch_tools:
        orchestrator.function_map['extract_doc_vocabulary'] = extract_doc_vocabulary.ExtractDocVocabulary()

    if 'amap_weather' in default_orch_tools:
        try:
            orchestrator.function_map['amap_weather'] = amap_weather.AmapWeather()
        except Exception:
            pass

    if 'code_interpreter' in default_orch_tools:
        try:
            orchestrator.function_map['code_interpreter'] = code_interpreter.CodeInterpreter(cfg={'work_dir': 'workspace'})
        except Exception:
            pass

    if 'python_executor' in default_orch_tools:
        try:
            orchestrator.function_map['python_executor'] = python_executor.PythonExecutor()
        except Exception:
            pass

    # Also load all sub-agents for the agent selector
    all_agents = [orchestrator]
    for agent_name in agent_pool.list_agents():
        if agent_name != 'orchestrator':  # Don't duplicate
            sub_agent = agent_pool.get_agent(agent_name)
            if sub_agent:
                all_agents.append(sub_agent)

    print(f"[OK] Available agents: {[a.name for a in all_agents]}")
    print("=" * 50)

    # Configure UI - now the orchestrator is one of the agents!
    chatbot_config = {
        'input.placeholder': 'Ask me anything! Multi-agent system with approval workflow...',
        'prompt.suggestions': [
            'Have the coder create a Python script',
            'Research quantum computing and write a report',
            'List all pending operations',
            'Show me the workspace files',
        ],
        'user.name': 'You',
        'available_tools': ALL_BUILTIN_TOOLS,
    }

    # Launch WebUI with orchestrator as the main agent
    # Users can also switch between agents via the dropdown
    print("\n[OK] Orchestrator ready! Launching WebUI...")
    print("Open your browser to http://127.0.0.1:7860")
    print("=" * 50)

    WebUI(all_agents, chatbot_config=chatbot_config).run()
