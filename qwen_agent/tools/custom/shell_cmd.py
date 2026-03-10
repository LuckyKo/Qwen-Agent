import subprocess
from qwen_agent.tools.base import BaseTool

class ShellCmd(BaseTool):
    """Execute a shell command (always requires user approval)."""

    name = 'shell_cmd'
    description = 'Execute a shell command on the host system. This ALWAYS requires explicit user approval.'
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
        params = self._verify_json_format_args(params)
        command = params['command']
        justification = params.get('justification', 'No justification provided.')

        return self.agent_pool.operation_manager.execute_shell_command(
            command=command,
            justification=justification,
            agent_name=self.agent_name,
        )
