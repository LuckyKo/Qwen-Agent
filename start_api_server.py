"""
Multi-Agent API Server — Entry Point

Same agent initialization as start_multi_agent.py, but launches the
WebSocket/REST API server instead of Gradio.

Usage:
    python start_api_server.py
    Open http://127.0.0.1:8765 in your browser.
"""

import json
import requests
from bs4 import BeautifulSoup
from qwen_agent.tools.base import BaseTool, register_tool
from agent_orchestrator import AgentPool, load_orchestrator_agent

from qwen_agent.tools import (
    image_gen,
    web_extractor,
    storage,
    simple_doc_parser,
    doc_parser,
    extract_doc_vocabulary,
    code_interpreter,
)


# ── Reuse DDGSearch from start_multi_agent ────────────────────────────────────
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


# ── Configuration (same as start_multi_agent.py) ──────────────────────────────
llm_cfg = {
    'model': 'whatever_is_on',
    'model_server': 'http://localhost:1234/v1',
    'api_key': 'EMPTY',
    'model_type': 'qwenvl_oai',
    'max_input_tokens': 65536,
}

DEFAULT_TOOLS = {
    'orchestrator': [
        'call_agent', 'dismiss_agent', 'list_agents',
        'compress_context', 'write_file', 'edit_file', 'delete_file', 'copy_file', 'move_file', 'read_file', 'view_image', 'list_dir', 'grep',
        'ddg_search', 'web_extractor', 'storage', 'retrieval'
    ],
    'coder': [
        'call_agent', 'list_agents',
        'read_file', 'view_image', 'compress_context', 'write_file', 'edit_file', 'delete_file', 'copy_file', 'move_file', 'list_dir', 'grep', 'code_interpreter', 'shell_cmd',
        'ddg_search', 'web_extractor', 'storage'
    ],
    'researcher': [
        'call_agent', 'list_agents',
        'read_file', 'view_image', 'compress_context', 'write_file', 'edit_file', 'delete_file', 'copy_file', 'move_file', 'list_dir', 'grep', 'code_interpreter',
        'ddg_search', 'web_extractor', 'storage', 'retrieval',
        'doc_parser', 'simple_doc_parser', 'extract_doc_vocabulary'
    ],
    'writer': [
        'call_agent', 'list_agents',
        'read_file', 'view_image', 'compress_context', 'write_file', 'edit_file', 'list_dir',
        'ddg_search', 'web_extractor', 'storage',
        'doc_parser', 'simple_doc_parser'
    ],
    'reviewer': [
        'call_agent', 'list_agents',
        'read_file', 'view_image', 'compress_context', 'list_dir', 'grep', 'code_interpreter',
    ],
}


def initialize_agents():
    """Set up agents, pool, and config. Returns (all_agents, agent_pool, chatbot_config)."""
    print("Initializing Agent Orchestrator (API Server)...")
    print("=" * 50)

    agent_pool = AgentPool(llm_cfg, 'agents')

    # Add tools to all agents based on their role
    for agent_name in agent_pool.list_agents():
        agent = agent_pool.get_agent(agent_name)
        if agent:
            default_tools = DEFAULT_TOOLS.get(agent_name, DEFAULT_TOOLS['writer'])

            agent.function_map['ddg_search'] = DDGSearch()

            if 'image_gen' in default_tools:
                try:
                    agent.function_map['image_gen'] = image_gen.ImageGen(llm_cfg=llm_cfg)
                except Exception:
                    pass

            if 'web_extractor' in default_tools:
                agent.function_map['web_extractor'] = web_extractor.WebExtractor(cfg={'work_dir': 'workspace'})

            if 'storage' in default_tools:
                agent.function_map['storage'] = storage.Storage()

            if 'retrieval' in default_tools:
                from qwen_agent.tools import retrieval
                agent.function_map['retrieval'] = retrieval.Retrieval(cfg={'work_dir': 'workspace'})

            if 'simple_doc_parser' in default_tools:
                agent.function_map['simple_doc_parser'] = simple_doc_parser.SimpleDocParser(cfg={'work_dir': 'workspace'})

            if 'doc_parser' in default_tools:
                agent.function_map['doc_parser'] = doc_parser.DocParser(cfg={'work_dir': 'workspace'})

            if 'extract_doc_vocabulary' in default_tools:
                agent.function_map['extract_doc_vocabulary'] = extract_doc_vocabulary.ExtractDocVocabulary(cfg={'work_dir': 'workspace'})

            if 'code_interpreter' in default_tools:
                try:
                    agent.function_map['code_interpreter'] = code_interpreter.CodeInterpreter(cfg={'work_dir': 'workspace'})
                except Exception:
                    pass

            if 'python_executor' in default_tools:
                try:
                    from qwen_agent.tools import python_executor
                    agent.function_map['python_executor'] = python_executor.PythonExecutor(cfg={'work_dir': 'workspace'})
                except Exception:
                    pass

    # Load orchestrator
    orchestrator = load_orchestrator_agent(agent_pool, llm_cfg)

    orchestrator.function_map['ddg_search'] = DDGSearch()

    default_orch_tools = DEFAULT_TOOLS['orchestrator']
    if 'web_extractor' in default_orch_tools:
        orchestrator.function_map['web_extractor'] = web_extractor.WebExtractor(cfg={'work_dir': 'workspace'})
    if 'storage' in default_orch_tools:
        orchestrator.function_map['storage'] = storage.Storage()
    if 'retrieval' in default_orch_tools:
        from qwen_agent.tools import retrieval
        orchestrator.function_map['retrieval'] = retrieval.Retrieval(cfg={'work_dir': 'workspace'})

    if 'simple_doc_parser' in default_orch_tools:
        orchestrator.function_map['simple_doc_parser'] = simple_doc_parser.SimpleDocParser(cfg={'work_dir': 'workspace'})
    if 'doc_parser' in default_orch_tools:
        orchestrator.function_map['doc_parser'] = doc_parser.DocParser(cfg={'work_dir': 'workspace'})
    if 'extract_doc_vocabulary' in default_orch_tools:
        orchestrator.function_map['extract_doc_vocabulary'] = extract_doc_vocabulary.ExtractDocVocabulary(cfg={'work_dir': 'workspace'})
    if 'code_interpreter' in default_orch_tools:
        try:
            orchestrator.function_map['code_interpreter'] = code_interpreter.CodeInterpreter(cfg={'work_dir': 'workspace'})
        except Exception:
            pass
    if 'python_executor' in default_orch_tools:
        try:
            from qwen_agent.tools import python_executor
            orchestrator.function_map['python_executor'] = python_executor.PythonExecutor(cfg={'work_dir': 'workspace'})
        except Exception:
            pass

    all_agents = [orchestrator]
    for agent_name in agent_pool.list_agents():
        if agent_name != 'orchestrator':
            sub_agent = agent_pool.get_agent(agent_name)
            if sub_agent:
                all_agents.append(sub_agent)

    print(f"[OK] Available agents: {[a.name for a in all_agents]}")
    print("=" * 50)

    chatbot_config = {
        'session_name': 'Maine',
        'verbose': False,
    }

    return all_agents, agent_pool, chatbot_config


if __name__ == '__main__':
    import sys

    all_agents, agent_pool, chatbot_config = initialize_agents()

    # Set up async terminal input (same as start_multi_agent.py)
    import threading
    def async_input_listener():
        while True:
            try:
                msg = sys.stdin.readline().strip()
                if msg:
                    agent_pool.async_message_queue.append(msg)
                    print(f"\n[QUEUED] '{msg}' will be injected on next turn.")
            except Exception:
                break
    threading.Thread(target=async_input_listener, daemon=True).start()

    # Create and launch the API server
    from api_server import create_app
    import uvicorn

    app = create_app(all_agents, agent_pool, chatbot_config)

    port = 8765
    print(f"\n[OK] API Server ready!")
    print(f"    -> Open http://127.0.0.1:{port} in your browser")
    print(f"    -> WebSocket at ws://127.0.0.1:{port}/ws/chat")
    print(f"    -> REST API at http://127.0.0.1:{port}/api/")
    print(f"\n[TIP] Type in this terminal to inject messages into the active agent.")
    print("=" * 50)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
