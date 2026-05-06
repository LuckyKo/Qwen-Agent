import subprocess
from qwen_agent.tools.base import BaseTool

class ShellCmd(BaseTool):
    """Execute a shell command (always requires user approval)."""

    name = 'shell_cmd'
    description = ('Execute a shell command on the host system. This ALWAYS requires explicit user approval. '
                   'Commands run with the workspace directory as the working directory.')
    parameters = {
        'type': 'object',
        'properties': {
            'command': {
                'type': 'string',
                'description': 'The exact shell command to execute.'
            },
            'justification': {
                'type': 'string',
                'description': 'Why you need to execute this command.'
            },
            'cwd': {
                'type': 'string',
                'description': 'Optional working directory, relative to workspace root.'
            }
        },
        'required': ['command', 'justification'],
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

        try:
            if isinstance(params, str):
                p = json_loads(params)
                params = json.dumps(p)
        except Exception:
            pass

        params = self._verify_json_format_args(params)
        command = params['command']
        justification = params.get('justification', 'No justification provided.')
        cwd = params.get('cwd', '.')

        # Get the truncation limit from agent/tool options
        char_limit = 2000
        if hasattr(self, 'agent_pool') and self.agent_pool:
            llm_cfg = getattr(self.agent_pool, 'llm_cfg', {})
            char_limit = llm_cfg.get('shell_char_limit', char_limit)
        elif self.cfg.get('shell_char_limit'):
            char_limit = self.cfg.get('shell_char_limit')

        agent_name = kwargs.get('agent_instance_name') or self.agent_name

        return self.agent_pool.operation_manager.execute_shell_command(
            command=command,
            justification=justification,
            agent_name=agent_name,
            cwd=cwd,
            char_limit=int(char_limit),
        )
