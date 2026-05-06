import json
from pathlib import Path
from qwen_agent.tools.base import BaseTool


class ReadFile(BaseTool):
    """Reads and returns the content of a specified file. Handles text, images, and PDF files."""

    name = 'read_file'
    description = ('Reads and returns the content of a specified file. If the file is large, '
                   'the content will be truncated. The tool\'s response will clearly indicate '
                   'if truncation has occurred and will provide details on how to read more '
                   'of the file using the \'offset\' and \'limit\' parameters. Handles text, '
                   'images (PNG, JPG, GIF, WEBP, SVG, BMP), and PDF files. For text files, '
                   'it can read specific line ranges.\n'
                   'NOTE: All paths are relative to the workspace root (e.g., "src/main.py", "data/input.csv").')
    parameters = {
        'type': 'object',
        'properties': {
            'absolute_path': {
                'type': 'string',
                'description': "Path to the file, relative to the workspace root (e.g., 'src/main.py', 'data/input.csv')."
            },
            'offset': {
                'type': 'integer',
                'description': "Optional: For text files, the 0-based line number to start reading from. Use for paginating through large files.",
                'default': 0
            },
            'limit': {
                'type': 'integer',
                'description': "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files."
            },
            'full_read': {
                'type': 'boolean',
                'description': 'Set to true to read the entire file (bypasses truncation limit). Default is false.',
                'default': False
            }
        },
        'required': ['absolute_path'],
    }

    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name')

    def call(self, params: str, **kwargs) -> str:
        from qwen_agent.utils.utils import json_loads
        import json
        # Mapping for backward compatibility
        try:
            if isinstance(params, str):
                p = json_loads(params)
                if 'path' in p and 'absolute_path' not in p:
                    p['absolute_path'] = p['path']
                if 'start_line' in p and 'offset' not in p:
                    p['offset'] = p['start_line'] - 1
                params = json.dumps(p)
            elif isinstance(params, dict):
                if 'path' in params and 'absolute_path' not in params:
                    params['absolute_path'] = params['path']
                if 'start_line' in params and 'offset' not in params:
                    params['offset'] = params['start_line'] - 1
        except:
            pass

        params = self._verify_json_format_args(params)
        path = params.get('absolute_path')
        offset = params.get('offset', 0)
        start_line = params.get('start_line', offset + 1)
        limit = params.get('limit')
        full_read = params.get('full_read', False)

        # Get the truncation limit from agent/tool options
        cfg_limit = 1000
        if hasattr(self, 'agent_pool') and self.agent_pool:
            cfg_limit = getattr(self.agent_pool, 'llm_cfg', {}).get('read_file_limit', cfg_limit)
        elif self.cfg.get('read_file_limit'):
            cfg_limit = self.cfg.get('read_file_limit')

        if not full_read:
            if limit is None or limit > cfg_limit:
                limit = cfg_limit
        else:
            if limit is None:
                limit = 1000000  # Effectively "full" read

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
            
            # --- Simple per-tool chunk limit ---
            # Cap a single read to ~25% of the context window (in chars).
            # The orchestrator's _truncate_tool_result() handles the 95% context guard.
            max_input_tokens = 58000
            if hasattr(self, 'agent_pool') and self.agent_pool:
                llm_cfg = getattr(self.agent_pool, 'llm_cfg', {})
                pool_max = llm_cfg.get('max_input_tokens') or llm_cfg.get('generate_cfg', {}).get('max_input_tokens')
                if pool_max:
                    max_input_tokens = int(pool_max)
            agent_obj = kwargs.get('agent_obj')
            if agent_obj and hasattr(agent_obj, 'llm') and hasattr(agent_obj.llm, 'generate_cfg'):
                agent_max = agent_obj.llm.generate_cfg.get('max_input_tokens')
                if agent_max and agent_max != 58000:
                    max_input_tokens = int(agent_max)
            
            # 25% of context * ~2.5 chars/token
            char_limit = int(max_input_tokens * 0.25 * 2.5)
            char_limit = max(500, char_limit)  # floor at 500 chars

            end_idx = min(total_lines, start_idx + limit)
            
            # Build content iteratively, respecting the chunk char limit
            content_lines = []
            current_chars = 0
            actual_end_idx = start_idx
            
            for i in range(start_idx, end_idx):
                line_text = f"{i+1}: {lines[i]}"
                if current_chars + len(line_text) > char_limit:
                    if current_chars == 0:
                        # First line is itself huge — include a truncated portion
                        cut = min(len(line_text), max(char_limit, 200))
                        content_lines.append(line_text[:cut] + " ... [LINE TRUNCATED]\n")
                        actual_end_idx = i + 1
                    break
                
                content_lines.append(line_text)
                current_chars += len(line_text)
                actual_end_idx = i + 1

            content = "".join(content_lines)
            header = f"File content ({path}), lines {start_idx+1} to {actual_end_idx} of {total_lines}:"
            
            if actual_end_idx < total_lines:
                header += " [TRUNCATED]"

            msg = f"{header}\n```\n{content}\n```"
            
            if actual_end_idx < total_lines:
                msg += (
                    f"\n\n[PAGINATION NOTE: This file is large. Use read_file with "
                    f"start_line={actual_end_idx+1} to read the next "
                    f"{min(limit, total_lines - actual_end_idx)} lines.]"
                )

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
    """Writes content to a specified file in the local filesystem."""

    name = 'write_file'
    description = ('Writes content to a specified file in the local filesystem. '
                   'This is auto-approved for new files. To edit existing files, '
                   'use edit_file instead. Overwriting an existing file not owned '
                   'by you requires user approval. '
                   'Place the file content in <content></content> XML tags after the JSON arguments '
                   'instead of inside the JSON string.\n'
                   'NOTE: All paths are relative to the workspace root (e.g., "src/main.py", "output/result.txt").')
    parameters = {
        'type': 'object',
        'properties': {
            'file_path': {
                'type': 'string',
                'description': "Path to the file, relative to the workspace root (e.g., 'src/main.py', 'output/result.txt')."
            },
            'content': {
                'type': 'string',
                'description': 'The content to write to the file. Place in <content></content> XML tags.'
            },
            'justification': {
                'type': 'string',
                'description': 'Why you need to create this file'
            }
        },
        'required': ['file_path', 'content'],
    }

    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name')

    def call(self, params: str, **kwargs) -> str:
        import re
        import json
        from qwen_agent.utils.utils import extract_code, json_loads

        # --- Robust Fallback for Non-JSON Input ---
        # Handles the case where the model emits "path\n```code```" instead of JSON
        if isinstance(params, str) and not params.strip().startswith('{'):
            match = re.search(r'^(?:path:?\s*)?([^\n`]+)\s*?\n*?```[^\n]*\n(.*?)\n?```', params.strip(), re.DOTALL | re.IGNORECASE)
            if match:
                path = match.group(1).strip()
                content = match.group(2)
                return self.agent_pool.operation_manager.write_file(
                    path=path,
                    content=content,
                    agent_name=self.agent_name,
                )

        # Mapping for backward compatibility
        try:
            if isinstance(params, str):
                p = json_loads(params)
                if 'path' in p and 'file_path' not in p:
                    p['file_path'] = p['path']
                params = json.dumps(p)
            elif isinstance(params, dict):
                if 'path' in params and 'file_path' not in params:
                    params['file_path'] = params['path']
        except:
            pass

        # --- Standard JSON Path ---
        params_json = self._verify_json_format_args(params)
        path = params_json.get('file_path')
        content = params_json.get('content', '')

        # Only strip markdown wrappers if content looks like it was JSON-embedded
        # (i.e., starts with ``` — this is a legacy fallback for when XML extraction
        # didn't happen and the model put a code block inside the JSON string)
        if isinstance(content, str) and content.strip().startswith('```'):
            content = extract_code(content)

        return self.agent_pool.operation_manager.write_file(
            path=path,
            content=content,
            agent_name=self.agent_name,
        )


class EditFile(BaseTool):
    """Replaces text within a file."""

    name = 'edit_file'
    description = ('Replaces text within a file. By default, replaces a single occurrence. '
                   'Requires user approval before changes are applied for any files not '
                   'owned by the current agent. Always use the read_file tool to examine '
                   'the file\'s current content before attempting a text replacement.\n\n'
                   'Place `old_string` and `new_string` in their respective XML tags '
                   '(<old_string></old_string> and <new_string></new_string>) after the JSON arguments. '
                   'Do NOT put them inside the JSON string. Include at least 3 lines of context '
                   'BEFORE and AFTER the target text, matching whitespace and indentation precisely.\n'
                   'NOTE: All paths are relative to the workspace root.')
    parameters = {
        'type': 'object',
        'properties': {
            'file_path': {
                'type': 'string',
                'description': "Path to the file, relative to the workspace root (e.g., 'src/main.py')."
            },
            'old_string': {
                'type': 'string',
                'description': 'The EXACT literal text to replace. Place in <old_string></old_string> XML tags. Include at least 3 lines of context.'
            },
            'new_string': {
                'type': 'string',
                'description': 'The exact literal text to replace old_string with. Place in <new_string></new_string> XML tags.'
            },
            'full_content': {
                'type': 'string',
                'description': 'Optional: use ONLY if you must overwrite the entire file. Place in <full_content></full_content> XML tags.'
            },
            'justification': {
                'type': 'string',
                'description': 'Why you need to edit this file'
            }
        },
        'required': ['file_path'],
    }

    def __init__(self, cfg=None, **kwargs):
        try:
            super().__init__(cfg)
        except (ValueError, TypeError):
            super().__init__()
        self.agent_pool = kwargs.get('agent_pool')
        self.agent_name = kwargs.get('agent_name')

    def call(self, params: str, **kwargs) -> str:
        import json
        from qwen_agent.utils.utils import extract_code, json_loads
        
        # Mapping for backward compatibility
        try:
            if isinstance(params, str):
                p = json_loads(params)
                if 'path' in p and 'file_path' not in p:
                    p['file_path'] = p['path']
                if 'old_content' in p and 'old_string' not in p:
                    p['old_string'] = p['old_content']
                if 'new_content' in p and 'new_string' not in p:
                    p['new_string'] = p['new_content']
                params = json.dumps(p)
            elif isinstance(params, dict):
                if 'path' in params and 'file_path' not in params:
                    params['file_path'] = params['path']
                if 'old_content' in params and 'old_string' not in params:
                    params['old_string'] = params['old_content']
                if 'new_content' in params and 'new_string' not in params:
                    params['new_string'] = params['new_content']
        except:
            pass

        params_json = self._verify_json_format_args(params)
        path = params_json.get('file_path')
        old_string = params_json.get('old_string')
        new_string = params_json.get('new_string')
        full_content = params_json.get('full_content')

        # Backward compatibility for models that still send 'content' instead of old/new
        if 'content' in params_json and not old_string:
            full_content = params_json['content']

        # Only strip markdown wrappers as a legacy fallback (when content was
        # JSON-embedded instead of XML-extracted)
        if new_string and isinstance(new_string, str) and new_string.strip().startswith('```'):
            new_string = extract_code(new_string)
        if full_content and isinstance(full_content, str) and full_content.strip().startswith('```'):
            full_content = extract_code(full_content)

        return self.agent_pool.operation_manager.edit_file(
            path=path,
            agent_name=self.agent_name,
            old_content=old_string,
            new_content=new_string,
            full_content=full_content,
        )


class ListDir(BaseTool):
    """Lists the names of files and subdirectories directly within a specified directory path."""

    name = 'list_dir'
    description = ('Lists the names of files and subdirectories directly within a specified directory path.\n'
                   'NOTE: All paths are relative to the workspace root. Use "." for the workspace root itself.')
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': "Path to the directory, relative to the workspace root (e.g., '.', 'src', 'data/images')"
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

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        path = params.get('path', '.')
        return self.agent_pool.operation_manager.list_directory(path)


class Grep(BaseTool):
    """Search for text patterns in files."""

    name = 'grep'
    description = ('Search for a text pattern in files (supports regex). Like the grep command.\n'
                   'NOTE: All paths are relative to the workspace root.')
    parameters = {
        'type': 'object',
        'properties': {
            'pattern': {
                'type': 'string',
                'description': 'Text or regex pattern to search for'
            },
            'path': {
                'type': 'string',
                'description': 'Directory to search in, relative to workspace root (default: ".")'
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
    description = ('Delete a file. Requires user approval before deletion for any files not '
                   'owned by the current agent. Deleting files you created in this session is auto-approved.\n'
                   'NOTE: All paths are relative to the workspace root.')
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': "Path to the file, relative to the workspace root (e.g., 'temp/scratch.py')"
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
    """Copy a file or directory."""

    name = 'copy_file'
    description = ('Copy a file or directory to a new location. This is auto-approved if the destination is new. '
                   'You become the owner of the copied file, allowing you to edit it freely without user approval.\n'
                   'NOTE: All paths are relative to the workspace root.')
    parameters = {
        'type': 'object',
        'properties': {
            'source': {
                'type': 'string',
                'description': "Path to the source file/directory, relative to workspace root (e.g., 'src/old.py')"
            },
            'destination': {
                'type': 'string',
                'description': "Path to the destination, relative to workspace root (e.g., 'src/new.py')"
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
    description = ('Move a file or directory to a new location. Requires user approval for any files not owned '
                   'by the current agent. Moving files you created in this session is auto-approved.\n'
                   'NOTE: All paths are relative to the workspace root.')
    parameters = {
        'type': 'object',
        'properties': {
            'source': {
                'type': 'string',
                'description': "Path to the source file/directory, relative to workspace root"
            },
            'destination': {
                'type': 'string',
                'description': "Path to the destination, relative to workspace root"
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
