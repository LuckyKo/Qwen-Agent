"""
Operation Manager - Generic approval system for agent operations
- Sub-agents request operations with context
- Manager can approve, reject, or suggest modifications
- Supports back-and-forth conversation between manager and sub-agents
- Maintains conversation state for efficient context switching
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


class OperationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"  # Manager suggested changes
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class OperationType(Enum):
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_DELETE = "file_delete"
    API_CALL = "api_call"
    CODE_EXECUTE = "code_execute"
    EXTERNAL_TOOL = "external_tool"
    CUSTOM = "custom"


@dataclass
class OperationRequest:
    """Represents a request from a sub-agent to perform an operation."""
    request_id: str
    agent_name: str
    operation_type: str
    description: str
    parameters: Dict[str, Any]
    context: str = ""
    justification: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    manager_response: Optional[str] = None
    manager_suggestions: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'agent_name': self.agent_name,
            'operation_type': self.operation_type,
            'description': self.description,
            'parameters': self.parameters,
            'context': self.context,
            'justification': self.justification,
            'timestamp': self.timestamp,
            'status': self.status,
            'manager_response': self.manager_response,
            'manager_suggestions': self.manager_suggestions,
            'conversation_history': self.conversation_history,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OperationRequest':
        return cls(**data)


@dataclass
class AgentConversation:
    """Maintains conversation state for a sub-agent."""
    agent_name: str
    messages: List[Dict] = field(default_factory=list)
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'agent_name': self.agent_name,
            'messages': self.messages,
            'last_active': self.last_active,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConversation':
        return cls(**data)


class OperationManager:
    """
    Manages operations requiring approval and facilitates communication
    between sub-agents and the manager/orchestrator.
    """
    
    def __init__(self, base_dir: str = 'workspace'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Pending operations awaiting approval
        self.operations: Dict[str, OperationRequest] = {}
        
        # Conversation state for each agent
        # Allows efficient context switching without reprocessing
        self.agent_conversations: Dict[str, AgentConversation] = {}
        
        # Operation history (for audit/review)
        self.operation_history: List[OperationRequest] = []
        
        # Auto-approve rules (can be configured by manager)
        self.auto_approve_rules: Dict[str, bool] = {
            'new_file': True,  # Auto-approve new file creation
            'own_file_edit': True,  # Auto-approve editing own files
            'small_file': True,  # Auto-approve files < 10KB
        }
        
        # File ownership tracking
        self.file_ownership: Dict[str, str] = {}
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"op_{uuid.uuid4().hex[:8]}"
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path to be within the base directory (security)."""
        # Prevent path traversal attacks
        try:
            resolved = (self.base_dir / path).resolve()
            if not str(resolved).startswith(str(self.base_dir.resolve())):
                raise ValueError(f"Path '{path}' is outside the allowed directory")
            return resolved
        except Exception:
            # Fallback for paths that don't exist yet but are within base_dir
            return self.base_dir / path
    
    def list_directory(self, path: str = ".") -> str:
        """List contents of a directory."""
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"Directory not found: {path}"
            if not resolved.is_dir():
                return f"Not a directory: {path}"
            
            result = f"Contents of {path}/:\n\n"
            dirs = []
            files = []
            for item in resolved.iterdir():
                if item.is_dir():
                    dirs.append(item.name)
                else:
                    files.append(item.name)
            
            if dirs:
                result += "📁 Directories:\n"
                for d in sorted(dirs):
                    result += f"  📂 {d}/\n"
            
            if files:
                result += "\n📄 Files:\n"
                for f in sorted(files):
                    try:
                        size = (resolved / f).stat().st_size
                        size_str = f"{size:,} bytes" if size > 1000 else f"{size} bytes"
                    except:
                        size_str = "?"
                    result += f"  📝 {f} ({size_str})\n"
            
            if not dirs and not files:
                result += "  (empty directory)"
            
            return result
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def grep(self, pattern: str, path: str = ".", include: str = "*") -> str:
        """Search for text pattern in files."""
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"Directory not found: {path}"
            
            results = []
            pattern_re = re.compile(pattern, re.IGNORECASE)
            
            for file_path in resolved.rglob(include):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if pattern_re.search(line):
                                rel_path = file_path.relative_to(self.base_dir)
                                results.append(f"{rel_path}:{line_num}: {line.strip()}")
                    except:
                        continue
            
            if not results:
                return f"No matches found for pattern '{pattern}' in {path}/**/{include}"
            
            return f"Found {len(results)} matches for '{pattern}':\n\n" + '\n'.join(results[:50])
        except Exception as e:
            return f"Error searching: {str(e)}"
    
    def request_operation(
        self,
        agent_name: str,
        operation_type: str,
        parameters: Dict[str, Any],
        description: str = "",
        context: str = "",
        justification: str = ""
    ) -> str:
        """
        Sub-agent requests permission for an operation.
        
        Returns:
            request_id if pending approval, or immediate result if auto-approved
        """
        # Check auto-approve rules
        if self._should_auto_approve(agent_name, operation_type, parameters):
            return f"AUTO_APPROVED: {description}"
        
        # Create operation request
        request_id = self._generate_request_id()
        
        request = OperationRequest(
            request_id=request_id,
            agent_name=agent_name,
            operation_type=operation_type,
            parameters=parameters,
            description=description,
            context=context,
            justification=justification,
        )
        
        self.operations[request_id] = request
        
        # Return formatted pending message
        return self._format_pending_response(request)
    
    def _should_auto_approve(self, agent_name: str, operation_type: str, parameters: dict) -> bool:
        """Check if operation should be auto-approved based on rules."""
        # Example: Auto-approve new file creation
        if operation_type == OperationType.FILE_WRITE.value:
            path = parameters.get('path', '')
            if not self._file_exists(path):
                return self.auto_approve_rules.get('new_file', True)
        
        # Example: Auto-approve editing own files
        if operation_type == OperationType.FILE_EDIT.value:
            path = parameters.get('path', '')
            owner = self.file_ownership.get(path)
            if owner == agent_name:
                return self.auto_approve_rules.get('own_file_edit', True)
        
        return False
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists."""
        try:
            resolved = (self.base_dir / path).resolve()
            return resolved.exists() and resolved.is_file()
        except:
            return False
    
    def _format_pending_response(self, request: OperationRequest) -> str:
        """Format a response indicating the request is pending approval."""
        return f"""PENDING_APPROVAL: Request ID `{request.request_id}`

Operation: {request.operation_type}
Description: {request.description}
Agent: {request.agent_name}

Your request is awaiting manager approval.
The manager will review and may:
- Approve immediately
- Reject with explanation
- Suggest modifications

Use request_id {request.request_id} to check status or respond to manager questions."""
    
    def approve_operation(
        self,
        request_id: str,
        approver_name: str,
        response: str = "Approved",
        modifications: Optional[Dict] = None
    ) -> str:
        """
        Manager approves an operation, optionally with modifications.
        
        Args:
            modifications: If provided, the operation parameters should be modified
        """
        if request_id not in self.operations:
            return f"ERROR: Request ID '{request_id}' not found"
        
        request = self.operations.pop(request_id)
        
        if modifications:
            request.status = OperationStatus.MODIFIED.value
            request.manager_suggestions = json.dumps(modifications)
            request.parameters.update(modifications)
            message = f"APPROVED_WITH_CHANGES: Manager '{approver_name}' approved with modifications"
        else:
            request.status = OperationStatus.APPROVED.value
            message = f"APPROVED: Manager '{approver_name}' approved"
        
        request.manager_response = response
        
        # Track ownership for file operations
        if request.operation_type in [OperationType.FILE_WRITE.value, OperationType.FILE_EDIT.value]:
            path = request.parameters.get('path', '')
            self.file_ownership[path] = request.agent_name
        
        # Add to history
        self.operation_history.append(request)
        
        # Notify agent via conversation
        self._add_to_agent_conversation(
            request.agent_name,
            {
                'role': 'system',
                'content': f"Your request {request_id} was {message}.\nManager: {response}"
            }
        )
        
        return f"{message}. Request ID: {request_id}"
    
    def reject_operation(
        self,
        request_id: str,
        approver_name: str,
        reason: str
    ) -> str:
        """Manager rejects an operation with reason."""
        if request_id not in self.operations:
            return f"ERROR: Request ID '{request_id}' not found"
        
        request = self.operations.pop(request_id)
        request.status = OperationStatus.REJECTED.value
        request.manager_response = reason
        
        # Add to history
        self.operation_history.append(request)
        
        # Notify agent
        self._add_to_agent_conversation(
            request.agent_name,
            {
                'role': 'system',
                'content': f"Your request {request_id} was REJECTED by manager '{approver_name}'.\nReason: {reason}"
            }
        )
        
        return f"REJECTED: Manager '{approver_name}' rejected request {request_id}. Reason: {reason}"
    
    def request_more_info(
        self,
        request_id: str,
        manager_name: str,
        question: str,
        suggestions: Optional[str] = None
    ) -> str:
        """
        Manager asks sub-agent for more information before deciding.
        This starts a conversation thread.
        """
        if request_id not in self.operations:
            return f"ERROR: Request ID '{request_id}' not found"
        
        request = self.operations[request_id]
        
        # Add manager's question to conversation
        conversation_entry = {
            'role': 'manager',
            'name': manager_name,
            'content': question,
            'suggestions': suggestions,
            'timestamp': datetime.now().isoformat(),
        }
        
        request.conversation_history.append(conversation_entry)
        
        # Also add to agent's conversation state
        self._add_to_agent_conversation(
            request.agent_name,
            {
                'role': 'system',
                'content': f"Manager '{manager_name}' has questions about your request {request_id}:\n\n{question}\n\nSuggestions: {suggestions or 'None'}"
            }
        )
        
        return f"QUESTION_ADDED: Manager '{manager_name}' asked for clarification on request {request_id}"
    
    def agent_respond_to_manager(
        self,
        request_id: str,
        agent_name: str,
        response: str
    ) -> str:
        """
        Sub-agent responds to manager's questions.
        """
        if request_id not in self.operations:
            return f"ERROR: Request ID '{request_id}' not found or already resolved"
        
        request = self.operations[request_id]
        
        if request.agent_name != agent_name:
            return f"ERROR: This request belongs to agent '{request.agent_name}'"
        
        # Add agent's response to conversation
        conversation_entry = {
            'role': 'agent',
            'name': agent_name,
            'content': response,
            'timestamp': datetime.now().isoformat(),
        }
        
        request.conversation_history.append(conversation_entry)
        
        return f"RESPONSE_ADDED: Your response to request {request_id} has been sent to the manager"
    
    def _add_to_agent_conversation(self, agent_name: str, message: Dict):
        """Add a message to an agent's conversation state."""
        if agent_name not in self.agent_conversations:
            self.agent_conversations[agent_name] = AgentConversation(agent_name=agent_name)
        
        conv = self.agent_conversations[agent_name]
        conv.messages.append(message)
        conv.last_active = datetime.now().isoformat()
    
    def get_agent_conversation(self, agent_name: str, limit: int = 50) -> List[Dict]:
        """
        Get an agent's conversation history for context loading.
        This allows LM Studio to load state without reprocessing.
        """
        if agent_name not in self.agent_conversations:
            return []
        
        conv = self.agent_conversations[agent_name]
        return conv.messages[-limit:]
    
    def clear_agent_conversation(self, agent_name: str):
        """Clear an agent's conversation state (free up memory)."""
        if agent_name in self.agent_conversations:
            # Save to file first for archival
            conv = self.agent_conversations[agent_name]
            archive_path = self.base_dir / f".conversation_archive_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(archive_path, 'w') as f:
                json.dump(conv.to_dict(), f, indent=2)
            
            del self.agent_conversations[agent_name]
    
    def list_pending_operations(self, agent_name: Optional[str] = None) -> List[dict]:
        """List all pending operations, optionally filtered by agent."""
        pending = []
        for req_id, req in self.operations.items():
            if agent_name and req.agent_name != agent_name:
                continue
            pending.append({
                'request_id': req_id,
                'agent_name': req.agent_name,
                'operation_type': req.operation_type,
                'description': req.description,
                'justification': req.justification,
                'timestamp': req.timestamp,
                'conversation_count': len(req.conversation_history),
            })
        return pending
    
    def get_operation_status(self, request_id: str) -> Optional[dict]:
        """Get the status of a specific operation request."""
        # Check pending operations
        if request_id in self.operations:
            req = self.operations[request_id]
            return {
                'status': 'pending',
                'request': req.to_dict(),
            }
        
        # Check history
        for req in self.operation_history:
            if req.request_id == request_id:
                return {
                    'status': req.status,
                    'request': req.to_dict(),
                }
        
        return None
    
    def set_auto_approve_rule(self, rule_name: str, enabled: bool):
        """Manager can configure auto-approve rules."""
        self.auto_approve_rules[rule_name] = enabled
        return f"Auto-approve rule '{rule_name}' set to {enabled}"
    
    def get_file_owner(self, path: str) -> Optional[str]:
        """Get the owner of a file."""
        return self.file_ownership.get(path)
    
    def get_statistics(self) -> dict:
        """Get operation statistics."""
        return {
            'pending_operations': len(self.operations),
            'total_operations': len(self.operation_history),
            'approved': sum(1 for h in self.operation_history if h.status == OperationStatus.APPROVED.value),
            'rejected': sum(1 for h in self.operation_history if h.status == OperationStatus.REJECTED.value),
            'modified': sum(1 for h in self.operation_history if h.status == OperationStatus.MODIFIED.value),
            'active_agents': len(self.agent_conversations),
        }
