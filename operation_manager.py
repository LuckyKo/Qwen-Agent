"""
Operation Manager - Blocking user-facing approval system for agent operations.

All mutating operations (file write, edit, delete, move, copy, code execution)
require explicit user approval via the WebUI. The tool call blocks (via
threading.Event) until the user clicks Approve or Reject.

Read operations (read_file, list_dir, grep, view_image) are free access.
"""

import json
import re
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class OperationType(Enum):
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_DELETE = "file_delete"
    FILE_COPY = "file_copy"
    FILE_MOVE = "file_move"
    FILE_REPLACE = "file_replace"
    CODE_EXECUTE = "code_execute"
    EXTERNAL_TOOL = "external_tool"
    CONTEXT_COMPRESSION = "context_compression"
    CUSTOM = "custom"


@dataclass
class PendingApproval:
    """Represents a tool call waiting for user approval."""
    request_id: str
    agent_name: str
    tool_name: str
    tool_args: Dict[str, Any]
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Threading primitives for blocking
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    reject_reason: str = ""


# Timeout for user approval (seconds). Auto-rejects after this.
APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes


class OperationManager:
    """
    Manages blocking user-approval for tool operations.

    When a tool needs approval, it calls request_user_approval() which blocks
    the calling thread until the user responds via the WebUI. The WebUI calls
    user_approve() or user_reject() to unblock the thread.
    """

    def __init__(self, base_dir: str = 'workspace', agent_pool=None):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.agent_pool = agent_pool

        # Currently pending approvals (request_id -> PendingApproval)
        self.pending: Dict[str, PendingApproval] = {}

        # Lock for thread-safe access to pending dict
        self._lock = threading.Lock()

        # File ownership tracking (still useful for context in approval UI)
        self.file_ownership: Dict[str, str] = {}
        
        # User toggleable timeout
        self.enable_timeout: bool = True

    # ─── Auto-Approval for Agent-Owned Files ──────────────────────────────

    def _is_auto_approved(self, path: str, agent_name: str, creating_new: bool = False) -> bool:
        """
        Check if this operation can skip user approval.
        Auto-approved when:
          - The file was created by this agent during the current session.
          - The agent is creating a brand new file (doesn't exist yet).
        """
        if creating_new:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return True  # New file — no existing work affected

        resolved = self._resolve_path(path)
        owner = self.file_ownership.get(str(resolved))
        return owner == agent_name

    # ─── Blocking Approval API ────────────────────────────────────────────

    def request_user_approval(
        self,
        agent_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        description: str = "",
    ) -> Tuple[bool, str]:
        """
        Block the calling thread until the user approves or rejects.

        Returns:
            (True, "") if approved
            (False, reason) if rejected or timed out
        """
        request_id = f"op_{uuid.uuid4().hex[:8]}"

        approval = PendingApproval(
            request_id=request_id,
            agent_name=agent_name,
            tool_name=tool_name,
            tool_args=tool_args,
            description=description,
        )

        with self._lock:
            self.pending[request_id] = approval

        # Block until user responds or timeout
        timeout_val = APPROVAL_TIMEOUT_SECONDS if self.enable_timeout else None
        got_response = approval.event.wait(timeout=timeout_val)

        # Clean up
        with self._lock:
            self.pending.pop(request_id, None)

        if not got_response:
            # Timed out
            return False, "User is AFK, try another method if possible"

        if approval.approved:
            return True, ""
        else:
            return False, approval.reject_reason or "Rejected by user."

    def user_approve(self, request_id: str) -> str:
        """Called by WebUI when user clicks Approve."""
        with self._lock:
            approval = self.pending.get(request_id)

        if not approval:
            return f"ERROR: Request '{request_id}' not found or already resolved."

        approval.approved = True
        approval.event.set()
        return f"Approved: {request_id}"

    def user_reject(self, request_id: str, reason: str = "") -> str:
        """Called by WebUI when user clicks Reject."""
        with self._lock:
            approval = self.pending.get(request_id)

        if not approval:
            return f"ERROR: Request '{request_id}' not found or already resolved."

        approval.approved = False
        approval.reject_reason = reason or "Rejected by user."
        approval.event.set()
        return f"Rejected: {request_id}"

    def list_pending_approvals(self) -> List[dict]:
        """List all currently pending approvals (for the WebUI to poll)."""
        with self._lock:
            return [
                {
                    'request_id': a.request_id,
                    'agent_name': a.agent_name,
                    'tool_name': a.tool_name,
                    'tool_args': a.tool_args,
                    'description': a.description,
                    'timestamp': a.timestamp,
                }
                for a in self.pending.values()
            ]

    # ─── Path Resolution ──────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path to be within the base directory (security)."""
        try:
            resolved = (self.base_dir / path).resolve()
            if not str(resolved).startswith(str(self.base_dir)):
                # Handle potential case-sensitivity issues on Windows by lowercase comparison
                if os.name == 'nt' and not str(resolved).lower().startswith(str(self.base_dir).lower()):
                    raise ValueError(f"Path '{path}' is outside the allowed directory")
                elif os.name != 'nt':
                     raise ValueError(f"Path '{path}' is outside the allowed directory")
            return resolved
        except Exception:
            return (self.base_dir / path).resolve()

    # ─── Read Operations (Free Access) ────────────────────────────────────

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
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if pattern_re.search(line):
                                try:
                                    rel_path = file_path.relative_to(self.base_dir)
                                except ValueError:
                                    rel_path = file_path.name # Fallback
                                results.append(f"{rel_path}:{line_num}: {line.strip()}")
                        if len(results) > 100: # Practical limit
                            break
                    except:
                        continue

            if not results:
                return f"No matches found for pattern '{pattern}' in {path}/**/{include}"

            summary = f"Found {len(results)} matches for '{pattern}'"
            if len(results) > 50:
                summary += " (showing first 50)"
            return f"{summary}:\n\n" + '\n'.join(results[:50])
        except Exception as e:
            return f"Error searching: {str(e)}"

    # ─── Write Operations (Require User Approval) ─────────────────────────

    def write_file(self, path: str, content: str, agent_name: str) -> str:
        """Write a file — auto-approved for new files and owned files."""
        resolved = self._resolve_path(path)
        is_new = not resolved.exists()

        if not self._is_auto_approved(path, agent_name, creating_new=True):
            description = f"Overwrite existing file: {path} ({len(content)} chars)"
            approved, reason = self.request_user_approval(
                agent_name=agent_name,
                tool_name='write_file',
                tool_args={'path': path, 'content': content},
                description=description,
            )
            if not approved:
                return f"REJECTED BY USER: {reason}"

        try:
            resolved = self._resolve_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding='utf-8')
            self.file_ownership[str(resolved)] = agent_name
            return f"APPROVED: Created {path} ({len(content)} characters)"
        except Exception as e:
            return f"ERROR: Approved but execution failed: {str(e)}"

    def edit_file(self, path: str, agent_name: str,
                  old_content: Optional[str] = None,
                  new_content: Optional[str] = None,
                  full_content: Optional[str] = None) -> str:
        """Edit a file — auto-approved for agent-owned files."""
        resolved = self._resolve_path(path)

        if full_content is not None:
            description = f"Overwrite file: {path}"
            tool_args = {'path': path, 'content': full_content}
        elif old_content is not None and new_content is not None:
            # Validate the surgical edit before asking for approval
            if not resolved.exists():
                return f"File not found for surgical edit: {path}"
            file_content = resolved.read_text(encoding='utf-8')
            count = file_content.count(old_content)
            if count == 0:
                return f"ERROR: Pattern not found in {path}"
            if count > 1:
                return f"ERROR: Pattern found {count} times in {path}. Edit requires a unique block."

            description = f"Surgical edit to: {path}"
            tool_args = {'path': path, 'old_content': old_content, 'new_content': new_content}
        else:
            return "ERROR: Must provide either (old_content + new_content) or full_content"

        if not self._is_auto_approved(path, agent_name):
            approved, reason = self.request_user_approval(
                agent_name=agent_name,
                tool_name='edit_file',
                tool_args=tool_args,
                description=description,
            )
            if not approved:
                return f"REJECTED BY USER: {reason}"

        try:
            if full_content is not None:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(full_content, encoding='utf-8')
            else:
                file_content = resolved.read_text(encoding='utf-8')
                # Re-verify at execution time
                count = file_content.count(old_content)
                if count == 0:
                    return f"ERROR: Pattern no longer found in {path}"
                if count > 1:
                    return f"ERROR: Pattern no longer unique in {path}"
                new_full_content = file_content.replace(old_content, new_content)
                resolved.write_text(new_full_content, encoding='utf-8')

            self.file_ownership[str(resolved)] = agent_name
            return f"APPROVED: Edited {path}"
        except Exception as e:
            return f"ERROR: Approved but execution failed: {str(e)}"

    def delete_file(self, path: str, agent_name: str) -> str:
        """Delete a file — auto-approved for agent-owned files."""
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"File not found: {path}"

        if not self._is_auto_approved(path, agent_name):
            description = f"Delete: {path}"
            approved, reason = self.request_user_approval(
                agent_name=agent_name,
                tool_name='delete_file',
                tool_args={'path': path},
                description=description,
            )
            if not approved:
                return f"REJECTED BY USER: {reason}"

        try:
            if resolved.is_dir():
                import shutil
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            if str(resolved) in self.file_ownership:
                del self.file_ownership[str(resolved)]
            return f"APPROVED: Deleted {path}"
        except Exception as e:
            return f"ERROR: Approved but execution failed: {str(e)}"

    def copy_file(self, source: str, destination: str, agent_name: str) -> str:
        """Copy a file — auto-approved if destination is new or agent-owned."""
        src_path = self._resolve_path(source)
        if not src_path.exists():
            return f"Source not found: {source}"

        if not self._is_auto_approved(destination, agent_name, creating_new=True):
            description = f"Copy: {source} → {destination}"
            approved, reason = self.request_user_approval(
                agent_name=agent_name,
                tool_name='copy_file',
                tool_args={'source': source, 'destination': destination},
                description=description,
            )
            if not approved:
                return f"REJECTED BY USER: {reason}"

        try:
            dest_path = self._resolve_path(destination)
            import shutil
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
            self.file_ownership[str(dest_path)] = agent_name
            return f"APPROVED: Copied {source} to {destination}"
        except Exception as e:
            return f"ERROR: Approved but execution failed: {str(e)}"

    def move_file(self, source: str, destination: str, agent_name: str) -> str:
        """Move a file — auto-approved if source is agent-owned."""
        src_path = self._resolve_path(source)
        if not src_path.exists():
            return f"Source not found: {source}"

        if not self._is_auto_approved(source, agent_name):
            description = f"Move: {source} → {destination}"
            approved, reason = self.request_user_approval(
                agent_name=agent_name,
                tool_name='move_file',
                tool_args={'source': source, 'destination': destination},
                description=description,
            )
            if not approved:
                return f"REJECTED BY USER: {reason}"

        try:
            dest_path = self._resolve_path(destination)
            import shutil
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_path, dest_path)
            if str(src_path) in self.file_ownership:
                del self.file_ownership[str(src_path)]
            self.file_ownership[str(dest_path)] = agent_name
            return f"APPROVED: Moved {source} to {destination}"
        except Exception as e:
            return f"ERROR: Approved but execution failed: {str(e)}"

    def execute_shell_command(self, command: str, justification: str, agent_name: str) -> str:
        """Execute a shell command — NEVER auto-approved, always requires user approval."""
        description = f"Execute Shell Command:\n```bash\n{command}\n```\nJustification: {justification}"
        
        approved, reason = self.request_user_approval(
            agent_name=agent_name,
            tool_name='shell_cmd',
            tool_args={'command': command, 'justification': justification},
            description=description,
        )
        
        if not approved:
            return f"REJECTED BY USER: {reason}"
            
        try:
            import subprocess
            
            # Execute the command in the workspace directory
            result = subprocess.run(
                command,
                cwd=str(self.base_dir),
                shell=True,
                capture_output=True,
                text=True,
                timeout=120  # Prevent hanging indefinitely
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
                
            if result.returncode == 0:
                status = "Command completed successfully."
            else:
                status = f"Command exited with return code {result.returncode}."
                
            if not output.strip():
                output = "No output produced."
                
            return f"APPROVED: {status}\n\n{output}"
            
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out after 120 seconds."
        except Exception as e:
            return f"ERROR: Approved but execution failed: {str(e)}"

    # ─── Context Compression (still internal, auto-approved) ──────────────

    def apply_context_compression(self, agent_name: str, summary: str, fraction: float, agent_obj: Optional[Any] = None):
        """Apply context compression — this is internal so no user approval needed."""
        if not self.agent_pool:
            raise ValueError("agent_pool not connected to OperationManager")
        self.agent_pool._apply_context_compression(agent_name, summary, fraction, agent_obj=agent_obj)

    # ─── Utilities ────────────────────────────────────────────────────────

    def get_file_owner(self, path: str) -> Optional[str]:
        """Get the owner of a file."""
        return self.file_ownership.get(path)
