import json
import copy
from typing import List, Optional, Dict, Any
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import Message
from operation_manager import OperationType

@register_tool('call_agent', allow_overwrite=True)
class CallAgent(BaseTool):
    """Tool to call other specialized agents for help."""
    
    name = 'call_agent'
    description = 'Delegate a task to a specialized sub-agent. Use this when you need expertise from a specific domain.'
    parameters = {
        'type': 'object',
        'properties': {
            'agent_name': {
                'type': 'string',
                'description': 'Name of the sub-agent to call (e.g., "researcher", "coder", "writer")'
            },
            'task': {
                'type': 'string',
                'description': 'The specific task or question to delegate to the sub-agent'
            },
            'context': {
                'type': 'string',
                'description': 'Any relevant context or background information the sub-agent needs'
            }
        },
        'required': ['agent_name', 'task'],
    }
    
    def __init__(self, agent_pool=None, **kwargs):
        super().__init__(**kwargs)
        self.agent_pool = agent_pool
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        agent_name = params['agent_name']
        task = params['task']
        context = params.get('context', '')
        
        agent = self.agent_pool.get_agent(agent_name)
        if not agent:
            return f"Error: Agent '{agent_name}' not found. Available agents: {self.agent_pool.list_agents()}"
        
        messages = []
        if context:
            messages.append({
                'role': 'user',
                'content': f"Context: {context}\n\nTask: {task}\n\nPlease help with this task using your specialized expertise."
            })
        else:
            messages.append({'role': 'user', 'content': task})
        
        try:
            response = []
            for resp in agent.run(messages=messages):
                response = resp
            
            if response:
                result = response[-1].get('content', 'No response from agent')
                return f"[{agent_name}'s response]:\n{result}"
            return f"[{agent_name}]: No response generated"
        except Exception as e:
            return f"Error calling agent {agent_name}: {str(e)}"

class ApproveOperation(BaseTool):
    """Approve a pending operation, optionally with modifications (manager only)."""
    
    name = 'approve_operation'
    description = 'Approve a pending operation. Can include modifications to the parameters.'
    parameters = {
        'type': 'object',
        'properties': {
            'request_id': {
                'type': 'string',
                'description': 'The request ID to approve'
            },
            'response': {
                'type': 'string',
                'description': 'Approval message or conditions'
            },
            'modifications': {
                'type': 'object',
                'description': 'Optional modifications to apply to the operation',
                'properties': {},
                'additionalProperties': True
            }
        },
        'required': ['request_id'],
    }
    
    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name', 'orchestrator')
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        request_id = params['request_id']
        response = params.get('response', 'Approved')
        modifications = params.get('modifications')
        
        if not self.agent_pool:
            return "ERROR: agent_pool not set"
        
        return self.agent_pool.operation_manager.approve_operation(
            request_id, self.agent_name, response, modifications
        )

class RejectOperation(BaseTool):
    """Reject a pending operation with reason (manager only)."""
    
    name = 'reject_operation'
    description = 'Reject a pending operation. Must provide a reason.'
    parameters = {
        'type': 'object',
        'properties': {
            'request_id': {
                'type': 'string',
                'description': 'The request ID to reject'
            },
            'reason': {
                'type': 'string',
                'description': 'Reason for rejection'
            }
        },
        'required': ['request_id', 'reason'],
    }
    
    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name', 'orchestrator')
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        request_id = params['request_id']
        reason = params['reason']
        
        if not self.agent_pool:
            return "ERROR: agent_pool not set"
        
        return self.agent_pool.operation_manager.reject_operation(request_id, self.agent_name, reason)

class AskAgent(BaseTool):
    """Ask a sub-agent for clarification about their operation request (manager only)."""
    
    name = 'ask_agent'
    description = 'Ask a sub-agent for more information about their pending operation request.'
    parameters = {
        'type': 'object',
        'properties': {
            'request_id': {
                'type': 'string',
                'description': 'The request ID to ask about'
            },
            'question': {
                'type': 'string',
                'description': 'Question for the sub-agent'
            },
            'suggestions': {
                'type': 'string',
                'description': 'Optional suggestions for how to modify the request'
            }
        },
        'required': ['request_id', 'question'],
    }
    
    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name', 'orchestrator')
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        request_id = params['request_id']
        question = params['question']
        suggestions = params.get('suggestions', '')
        
        if not self.agent_pool:
            return "ERROR: agent_pool not set"
        
        return self.agent_pool.operation_manager.request_more_info(
            request_id, self.agent_name, question, suggestions
        )

class RespondToManager(BaseTool):
    """Respond to manager's questions about your operation request (sub-agents only)."""
    
    name = 'respond_to_manager'
    description = 'Respond to manager\'s questions about your pending operation request.'
    parameters = {
        'type': 'object',
        'properties': {
            'request_id': {
                'type': 'string',
                'description': 'The request ID you\'re responding to'
            },
            'response': {
                'type': 'string',
                'description': 'Your response to the manager\'s question'
            }
        },
        'required': ['request_id', 'response'],
    }
    
    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name')
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        request_id = params['request_id']
        response = params['response']
        
        return self.agent_pool.operation_manager.agent_respond_to_manager(
            request_id, self.agent_name, response
        )

class ListPendingOperations(BaseTool):
    """List all pending operation requests awaiting approval."""
    
    name = 'list_pending_operations'
    description = 'List all pending operation requests that need manager approval.'
    parameters = {
        'type': 'object',
        'properties': {
            'agent_name': {
                'type': 'string',
                'description': 'Filter by agent name (optional)'
            }
        },
        'required': [],
    }
    
    def __init__(self, agent_pool=None, **kwargs):
        super().__init__(**kwargs)
        self.agent_pool = agent_pool
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        agent_name = params.get('agent_name')
        
        requests = self.agent_pool.operation_manager.list_pending_operations(agent_name)
        
        if not requests:
            return "No pending operation requests."
        
        result = "Pending operation requests:\n\n"
        for req in requests:
            result += f"- **{req['request_id']}**: Agent `{req['agent_name']}` wants to `{req['operation_type']}` - {req['description']}\n"
            result += f"  Justification: {req['justification']}\n"
            if req['conversation_count'] > 0:
                result += f"  Messages: {req['conversation_count']} (has conversation)\n"
        
        return result

class DismissAgent(BaseTool):
    """Clear a sub-agent's conversation history (manager tool)."""

    name = 'dismiss_agent'
    description = (
        "End a sub-agent's current task and clear its conversation context. "
        "Use when you're done with a sub-agent and don't need its context anymore."
    )
    parameters = {
        'type': 'object',
        'properties': {
            'agent_name': {
                'type': 'string',
                'description': 'Name of the sub-agent to dismiss'
            },
        },
        'required': ['agent_name'],
    }

    def __init__(self, agent_pool=None, **kwargs):
        super().__init__(**kwargs)
        self.agent_pool = agent_pool

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        agent_name = params['agent_name']

        if agent_name not in self.agent_pool.agents:
            return f"Error: Agent '{agent_name}' not found. Available: {self.agent_pool.list_agents()}"

        had_context = bool(self.agent_pool.agent_conversations.get(agent_name))
        self.agent_pool.clear_conversation(agent_name)

        if had_context:
            return f"Agent '{agent_name}' dismissed — conversation context cleared."
        return f"Agent '{agent_name}' had no active conversation. Nothing to clear."
