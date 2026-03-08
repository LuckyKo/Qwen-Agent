import json
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import SYSTEM, USER, Message

@register_tool('compress_context', allow_overwrite=True)
class CompressContext(BaseTool):
    """Tool to propose a summary of the oldest part of the conversation to free up context space."""
    
    name = 'compress_context'
    description = (
        'Summarize the oldest part of the conversation history to free up context space. '
        'The specified fraction of the history (e.g. 0.2 for 20%) '
        'is replaced by a concise summary, preserving the narrative while freeing up tokens.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'fraction': {
                'type': 'number',
                'description': 'The fraction of history to summarize (e.g. 0.2 for 20%, 0.5 for 50%). Max 0.8.',
                'minimum': 0.1,
                'maximum': 0.8
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
        fraction = min(params.get('fraction', 0.2), 0.8)
        justification = params.get('justification', 'Context management')
        
        if not self.agent_pool:
            return "ERROR: agent_pool not connected to tool"
            
        # Prioritize instance name passed via kwargs (from Agent._call_tool)
        agent_name = kwargs.get('agent_instance_name') or self.agent_name or 'orchestrator'
        
        # Use current messages from kwargs if available to catch the very latest context,
        # otherwise fallback to the persistent pool.
        history = kwargs.get('messages')
        if not history:
            history = self.agent_pool.get_conversation(agent_name)
        
        if not history:
            return "ERROR: No conversation history to compress."
            
        # Handle both dicts (from pool) and Message objects (from kwargs)
        start_idx = 0
        first_msg = history[0]
        first_role = first_msg.get('role') if isinstance(first_msg, dict) else getattr(first_msg, 'role', '')
        
        if first_role == SYSTEM:
            start_idx = 1
            
        messages_to_compress = history[start_idx:]
        
        if len(messages_to_compress) < 3:
            return "ERROR: Conversation history too short to safely compress (need at least 3 messages)."
            
        # Ensure we compress at least 1 message if possible, ignoring strict fractions for short arrays
        num_to_summarize = max(1, int(len(messages_to_compress) * fraction))
            
        target_messages = messages_to_compress[:num_to_summarize]
        
        # Format the messages for the summary prompt
        history_text = ""
        for msg in target_messages:
            role = msg.get('role', 'unknown').upper() if isinstance(msg, dict) else getattr(msg, 'role', 'unknown').upper()
            content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
            if isinstance(content, list):
                content = " ".join([item.get('text', '') if isinstance(item, dict) else getattr(item, 'text', str(item)) for item in content])
            history_text += f"{role}: {content}\n\n"
            
        # Call LLM to generate summary
        from qwen_agent.llm import get_chat_model
        llm = get_chat_model(self.agent_pool.llm_cfg)
        
        summary_prompt = (
            "You are a memory compression assistant. Your task is to summarize the following conversation history.\n"
            "Focus strictly on key decisions, important facts, established context, and the current state of tasks.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the summary. Do not include introductory or concluding remarks (e.g. 'Here is a summary').\n"
            "2. Do not include meta-commentary or thinking process.\n"
            "3. Remain concise but comprehensive enough so that future turns can proceed without the original messages.\n\n"
            f"--- START HISTORY ---\n{history_text}\n--- END HISTORY ---\n\n"
            "Summary:"
        )
        
        summary = ""
        try:
            responses = list(llm.chat([Message(role=USER, content=summary_prompt)]))
            if responses and responses[-1]:
                summary = responses[-1][0].content
                
                # Cleanup common LM Studio meta-commentary
                import re
                summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
                
                # Strip conversational filler prefixes
                prefixes = ["here is a summary", "here is the summary", "summary:", "in summary,", "here's a summary", "**summary**:"]
                lower_summary = summary.lower()
                for prefix in prefixes:
                    if lower_summary.startswith(prefix):
                        summary = summary[len(prefix):].strip()
                        summary = summary.lstrip(':\n \t')
                        lower_summary = summary.lower() # update for next check
                        
        except Exception as e:
            return f"ERROR: Failed to generate summary: {str(e)}"
            
        if not summary:
            return "ERROR: LLM failed to generate a summary."
            
        # Context compression is an internal operation — apply directly
        try:
            agent_obj = kwargs.get('agent_obj')
            self.agent_pool.operation_manager.apply_context_compression(
                agent_name=agent_name,
                summary=summary,
                fraction=fraction,
                agent_obj=agent_obj,
            )
            
            # ALSO explicitly compress the active messages list sent by the LLM
            # (which is a deepcopy in FnCallAgent and won't automatically reflect the AgentPool changes)
            if kwargs.get('messages'):
                active_msgs = kwargs['messages']
                start_idx_active = 0
                
                # Check if first message is SYSTEM
                first_msg = active_msgs[0]
                first_role = first_msg.get('role') if isinstance(first_msg, dict) else getattr(first_msg, 'role', '')
                if first_role == SYSTEM:
                    start_idx_active = 1
                    
                messages_to_compress_active = active_msgs[start_idx_active:]
                num_to_remove_active = max(1, int(len(messages_to_compress_active) * fraction))

                # ADJUSTMENT: Ensure the first remaining message is a USER message.
                # Specifically, we scan forward from num_to_remove_active to find the NEXT USER message.
                # However, we must NEVER remove the very last message in the history (which is usually 
                # the current turn's prompt or assistant/function message).
                found_user = False
                temp_remove = num_to_remove_active
                while temp_remove < len(messages_to_compress_active):
                    next_msg = messages_to_compress_active[temp_remove]
                    role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', '')
                    if role == USER:
                        found_user = True
                        num_to_remove_active = temp_remove
                        break
                    temp_remove += 1
                
                # If we didn't find a USER message by scanning forward, we MUST scan BACKWARD 
                # from our target to ensure the history we keep starts with a USER message.
                if not found_user:
                    temp_remove = num_to_remove_active - 1
                    while temp_remove >= 0:
                        next_msg = messages_to_compress_active[temp_remove]
                        role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', '')
                        if role == USER:
                            found_user = True
                            num_to_remove_active = temp_remove
                            break
                        temp_remove -= 1
                
                # If we STILL found no USER message (which should be impossible), 
                # don't remove anything - better to be full than 400 error.
                if not found_user:
                    num_to_remove_active = 0

                if num_to_remove_active > 0:
                    summary_content = f"\n\n--- CONTEXT COMPRESSED ({int(fraction*100)}% of history summarized) ---\n\nSummary of previous context:\n{summary}\n\n--- END SUMMARY ---"
                    
                    new_active = []
                    if start_idx_active == 1:
                        # Merge summary INTO the existing system message to avoid a second SYSTEM message
                        first = active_msgs[0]
                        if isinstance(first, dict):
                            merged = {**first, 'content': (first.get('content', '') or '') + summary_content}
                        else:
                            merged = Message(role=first.role, content=(first.content or '') + summary_content)
                        new_active.append(merged)
                    else:
                        # No system message — insert summary as USER to stay API-compliant
                        if isinstance(active_msgs[0], dict):
                            new_active.append({'role': USER, 'content': summary_content})
                        else:
                            new_active.append(Message(role=USER, content=summary_content))
                    new_active.extend(messages_to_compress_active[num_to_remove_active:])
                    
                    # Mutate directly so the caller's reference is updated
                    active_msgs.clear()
                    active_msgs.extend(new_active)

            return (
                f"Context compressed: {int(fraction*100)}% of older history for agent '{agent_name}' "
                f"has been summarized and removed from your context window.\n\nSummary:\n{summary}"
            )
        except Exception as e:
            return f"ERROR: Compression failed: {str(e)}"
