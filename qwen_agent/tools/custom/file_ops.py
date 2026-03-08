import json
from pathlib import Path
from qwen_agent.tools.base import BaseTool


class ReadFile(BaseTool):
    """Read a file from the workspace (free access - no approval needed)."""

    name = 'read_file'
    description = 'Read content from a file in the workspace. Supports pagination via line numbers.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file relative to workspace directory'
            },
            'start_line': {
                'type': 'integer',
                'description': 'The line number to start reading from (1-indexed)',
                'default': 1
            },
            'limit': {
                'type': 'integer',
                'description': 'Maximum number of lines to read',
                'default': 1000
            }
        },
        'required': ['path'],
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
        path = params['path']
        start_line = params.get('start_line', 1)
        limit = params.get('limit', 1000)

        if hasattr(self, 'agent_pool') and self.agent_pool:
            base_dir = self.agent_pool.operation_manager.base_dir
        else:
            base_dir = Path('workspace')

        try:
            resolved = (base_dir / path).resolve()
            if not resolved.exists():
                return f"File not found: {path}"

            with open(resolved, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, start_idx + limit)

            subset = lines[start_idx:end_idx]
            content = "".join([f"{i+start_idx+1}: {line}" for i, line in enumerate(subset)])

            header = f"File content ({path}), lines {start_idx+1} to {end_idx} of {total_lines}:"
            if end_idx < total_lines:
                header += f" [TRUNCATED - see note at bottom for more]"

            msg = f"{header}\n```\n{content}\n```"
            if end_idx < total_lines:
                msg += f"\n\n[PAGINATION NOTE: This file is large. Use read_file with start_line={end_idx+1} to read the next {min(limit, total_lines - end_idx)} lines.]"

            return msg
        except Exception as e:
            return f"Error reading file: {str(e)}"


class ViewImage(BaseTool):
    """View an image file from the workspace."""

    name = 'view_image'
    description = 'View an image file in the workspace. Returns the image for the model to see.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the image file relative to workspace directory'
            }
        },
        'required': ['path'],
    }

    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')

    def call(self, params: str, **kwargs):
        from qwen_agent.llm.schema import ContentItem
        params = self._verify_json_format_args(params)
        path = params['path']

        if hasattr(self, 'agent_pool') and self.agent_pool:
            base_dir = self.agent_pool.operation_manager.base_dir
        else:
            base_dir = Path('workspace')

        try:
            resolved = (base_dir / path).resolve()
            if not resolved.exists():
                return f"Image not found: {path}"

            file_url = resolved.as_uri()

            return [
                ContentItem(image=file_url),
                ContentItem(text=f"Viewing image: {path}")
            ]
        except Exception as e:
            return f"Error viewing image: {str(e)}"


class WriteFile(BaseTool):
    """Create a new file in the workspace (requires user approval)."""

    name = 'write_file'
    description = 'Create a NEW file in the workspace. Requires user approval. To edit existing files, use edit_file instead.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file relative to workspace directory'
            },
            'content': {
                'type': 'string',
                'description': 'Content to write to the file'
            },
            'justification': {
                'type': 'string',
                'description': 'Why you need to create this file'
            }
        },
        'required': ['path', 'content'],
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
        path = params['path']
        content = params['content']

        return self.agent_pool.operation_manager.write_file(
            path=path,
            content=content,
            agent_name=self.agent_name,
        )


class EditFile(BaseTool):
    """Edit an existing file (requires user approval)."""

    name = 'edit_file'
    description = 'Edit an EXISTING file. Requires user approval before changes are applied.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file relative to workspace directory'
            },
            'old_content': {
                'type': 'string',
                'description': 'The EXACT unique block of text to be replaced.'
            },
            'new_content': {
                'type': 'string',
                'description': 'The new text to insert.'
            },
            'full_content': {
                'type': 'string',
                'description': 'Optional: use ONLY if you must overwrite the entire file (discouraged for large files).'
            },
            'justification': {
                'type': 'string',
                'description': 'Why you need to edit this file'
            }
        },
        'required': ['path', 'justification'],
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
        path = params['path']
        old_content = params.get('old_content')
        new_content = params.get('new_content')
        full_content = params.get('full_content')

        # Backward compatibility for models that still send 'content' instead of old/new
        if 'content' in params and not old_content:
            full_content = params['content']

        return self.agent_pool.operation_manager.edit_file(
            path=path,
            agent_name=self.agent_name,
            old_content=old_content,
            new_content=new_content,
            full_content=full_content,
        )


class ListDir(BaseTool):
    """List contents of a directory in the workspace."""

    name = 'list_dir'
    description = 'List all files and directories in a given path. Like `ls` or `dir` command.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Directory path relative to workspace (default: ".")'
            }
        },
        'required': [],
    }

    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        path = params.get('path', '.')
        return self.agent_pool.operation_manager.list_directory(path)


class Grep(BaseTool):
    """Search for text patterns in files."""

    name = 'grep'
    description = 'Search for a text pattern in files (supports regex). Like the grep command.'
    parameters = {
        'type': 'object',
        'properties': {
            'pattern': {
                'type': 'string',
                'description': 'Text or regex pattern to search for'
            },
            'path': {
                'type': 'string',
                'description': 'Directory to search in (default: ".")'
            },
            'include': {
                'type': 'string',
                'description': 'File pattern to include (e.g., "*.py", "*.md")'
            }
        },
        'required': ['pattern'],
    }

    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pattern = params['pattern']
        path = params.get('path', '.')
        include = params.get('include', '*')
        return self.agent_pool.operation_manager.grep(pattern, path, include)


class DeleteFile(BaseTool):
    """Delete a file (requires user approval)."""

    name = 'delete_file'
    description = 'Delete a file. Requires user approval before deletion.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file relative to workspace directory'
            }
        },
        'required': ['path'],
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
        path = params['path']
        return self.agent_pool.operation_manager.delete_file(path, self.agent_name)


class CopyFile(BaseTool):
    """Copy a file or directory (requires user approval)."""

    name = 'copy_file'
    description = 'Copy a file or directory to a new location. Requires user approval.'
    parameters = {
        'type': 'object',
        'properties': {
            'source': {
                'type': 'string',
                'description': 'Path to the source file/directory relative to workspace'
            },
            'destination': {
                'type': 'string',
                'description': 'Path to the destination relative to workspace'
            }
        },
        'required': ['source', 'destination'],
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
        source = params['source']
        destination = params['destination']
        return self.agent_pool.operation_manager.copy_file(source, destination, self.agent_name)


class MoveFile(BaseTool):
    """Move a file or directory (requires user approval)."""

    name = 'move_file'
    description = 'Move a file or directory to a new location. Requires user approval.'
    parameters = {
        'type': 'object',
        'properties': {
            'source': {
                'type': 'string',
                'description': 'Path to the source file/directory relative to workspace'
            },
            'destination': {
                'type': 'string',
                'description': 'Path to the destination relative to workspace'
            }
        },
        'required': ['source', 'destination'],
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
        source = params['source']
        destination = params['destination']
        return self.agent_pool.operation_manager.move_file(source, destination, self.agent_name)
