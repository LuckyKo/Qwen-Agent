import json
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import SYSTEM, USER, Message
from operation_manager import OperationType

@register_tool('compress_context', allow_overwrite=True)
class CompressContext(BaseTool):
    """Tool to propose a summary of the oldest part of the conversation to free up context space."""
    
    name = 'compress_context'
    description = (
        'Propose summarizing the oldest part of the conversation history. '
        'This creates a pending operation for the manager to approve. '
        'Once approved, the specified fraction of the history (e.g. 0.2 for 20%) '
        'is replaced by a concise summary, preserving the narrative while freeing up tokens.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'fraction': {
                'type': 'number',
                'description': 'The fraction of history to summarize (e.g. 0.2 for 20%, 0.5 for 50%). Max 0.5.',
                'minimum': 0.1,
                'maximum': 0.5
            },
            'justification': {
                'type': 'string',
                'description': 'Why compression is needed now (e.g. "Context threshold reached")'
            }
        },
        'required': ['fraction', 'justification'],
    }
    
    def __init__(self, agent_pool=None, agent_name=None, **kwargs):
        super().__init__(**kwargs)
        self.agent_pool = agent_pool
        self.agent_name = agent_name
    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        fraction = min(params.get('fraction', 0.2), 0.5)
        justification = params.get('justification', 'Context management')
        
        if not self.agent_pool:
            return "ERROR: agent_pool not connected to tool"
            
        agent_name = self.agent_name or 'orchestrator'
        history = self.agent_pool.get_conversation(agent_name)
        
        if not history:
            return "ERROR: No conversation history to compress."
            
        # Keep SYSTEM message if it exists
        start_idx = 0
        if history[0].get('role') == SYSTEM:
            start_idx = 1
            
        messages_to_compress = history[start_idx:]
        num_to_summarize = int(len(messages_to_compress) * fraction)
        
        if num_to_summarize <= 0:
            return "ERROR: Conversation history too short for requested compression fraction."
            
        target_messages = messages_to_compress[:num_to_summarize]
        
        # Format the messages for the summary prompt
        history_text = ""
        for msg in target_messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            if isinstance(content, list):
                # Simple extraction for summary
                content = " ".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in content])
            history_text += f"{role}: {content}\n\n"
            
        # Call LLM to generate summary
        # We use the agent_pool's llm_cfg to create a temporary LLM for summarization
        from qwen_agent.llm import get_chat_model
        llm = get_chat_model(self.agent_pool.llm_cfg)
        
        summary_prompt = (
            "Summarize the following part of our conversation history. "
            "Focus on key decisions, important facts, discovered context, and the current state of tasks. "
            "Remain concise but comprehensive enough so that future turns can proceed without the original messages.\n\n"
            f"--- START HISTORY ---\n{history_text}\n--- END HISTORY ---\n\n"
            "Summary:"
        )
        
        summary = ""
        try:
            responses = list(llm.chat([Message(role=USER, content=summary_prompt)]))
            if responses:
                summary = responses[-1].content
        except Exception as e:
            return f"ERROR: Failed to generate summary: {str(e)}"
            
        if not summary:
            return "ERROR: LLM failed to generate a summary."
            
        # Create the pending operation
        description = f"Compress {int(fraction*100)}% of history for agent '{agent_name}'"
        result = self.agent_pool.operation_manager.request_operation(
            agent_name=agent_name,
            operation_type=OperationType.CONTEXT_COMPRESSION.value,
            parameters={
                'agent_name': agent_name,
                'summary': summary,
                'fraction': fraction
            },
            description=description,
            context=f"Proposed Summary:\n{summary}",
            justification=justification
        )
        
        return result
