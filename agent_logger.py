"""
Agent Instance Logger — Persistent JSONL logging for agent sessions.

Each agent instance gets its own timestamped log file that records
metadata and all messages (including tool calls and results).
"""

import copy
import json
import os
import datetime
from typing import Any, Dict, List, Optional, Union

from qwen_agent.log import logger


class AgentInstanceLogger:
    """Handles persistent logging for an agent instance."""
    
    def __init__(self, agent_class: str, instance_name: str, log_dir: str, base_metadata: Optional[Dict] = None):
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
                "current_log_path": self.log_path,
            },
            "history": []
        }
        
        # Merge base metadata if provided (e.g. from a loaded session)
        if base_metadata:
            for k, v in base_metadata.items():
                if k not in self.data["metadata"]:
                    self.data["metadata"][k] = v
                elif k == "original_log_path":
                     # Carry over origin if it exists, or set it if we're the first continuation
                     self.data["metadata"][k] = v
            # If we don't have an original_log_path yet and we are continuing, set it
            if "original_log_path" not in self.data["metadata"] and "current_log_path" in base_metadata:
                self.data["metadata"]["original_log_path"] = base_metadata["current_log_path"]
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
