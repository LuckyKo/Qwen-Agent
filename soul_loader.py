"""
Soul Loader - Load agent personality from soul.md file
"""

import yaml
from pathlib import Path


def load_soul(soul_path: str = 'soul.md') -> dict:
    """
    Load agent configuration from a soul.md file.
    
    Args:
        soul_path: Path to the soul.md configuration file
        
    Returns:
        Dictionary with agent configuration
    """
    path = Path(soul_path)
    if not path.exists():
        raise FileNotFoundError(f"Soul file not found: {soul_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML (ignoring comments)
    config = yaml.safe_load(content)
    
    return config


def build_system_prompt(config: dict) -> str:
    """
    Build a system prompt from the soul configuration.
    
    Args:
        config: Configuration dictionary from load_soul()
        
    Returns:
        Formatted system prompt string
    """
    identity = config.get('identity', {})
    communication = config.get('communication', {})
    rules = config.get('rules', [])
    capabilities = config.get('capabilities', {})
    notes = config.get('notes', '')
    
    # Build identity section
    system_prompt = f"""You are {config.get('name', 'Assistant')}.
{config.get('tagline', '')}

## Who You Are
{identity.get('background', '')}

Personality traits:
"""
    
    for trait in identity.get('personality_traits', []):
        system_prompt += f"- {trait}\n"
    
    # Communication style
    system_prompt += f"""
## How You Communicate
Tone: {communication.get('tone', 'Friendly and helpful')}

Style guidelines:
"""
    
    for note in communication.get('style_notes', []):
        system_prompt += f"- {note}\n"
    
    # Tools available
    tools = capabilities.get('tools', [])
    if tools:
        system_prompt += f"""
## Your Tools
You have access to these tools:
"""
        for tool in tools:
            system_prompt += f"- **{tool}**: Use when you need to {get_tool_description(tool)}\n"
    
    # Rules
    system_prompt += f"""
## Your Rules
"""
    for i, rule in enumerate(rules, 1):
        system_prompt += f"{i}. {rule}\n"
    
    # Special notes
    if notes:
        system_prompt += f"""
## Remember
{notes.strip()}
"""
    
    return system_prompt


def get_tool_description(tool_name: str) -> str:
    """Get a brief description of what each tool is used for."""
    descriptions = {
        'get_weather': 'check current weather conditions',
        'web_search': 'search for current information online',
        'visit_website': 'read content from a specific URL',
        'code_interpreter': 'execute code for calculations or analysis',
    }
    return descriptions.get(tool_name, 'access external information')


def create_agent_from_soul(llm_cfg: dict, soul_path: str = 'soul.md', agent_class=None, **agent_kwargs):
    """
    Create an Agent from a soul.md file.
    
    Args:
        llm_cfg: LLM configuration dictionary
        soul_path: Path to the soul.md file
        agent_class: Optional class to instantiate (defaults to Assistant)
        **agent_kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        Configured agent instance
    """
    from qwen_agent.agents import Assistant
    
    if agent_class is None:
        agent_class = Assistant
    
    # Load soul configuration
    config = load_soul(soul_path)
    
    # Build system prompt
    system_prompt = build_system_prompt(config)
    
    # Note: Tools are added separately by the framework
    # Don't use function_list here as tools are added manually later
    
    # Create agent
    agent = agent_class(
        llm=llm_cfg,
        name=config.get('name', 'Assistant'),
        description=config.get('tagline', 'A helpful AI assistant'),
        system_message=system_prompt,
        function_list=[],  # Empty - tools added manually by agent_orchestrator
        **agent_kwargs
    )
    
    # Store config for later access
    agent.agent_configs = {config.get('name', 'assistant'): config}

    return agent, config


# Example usage
if __name__ == '__main__':
    # Test loading the soul
    config = load_soul()
    print(f"Loaded agent: {config['name']}")
    print(f"Tagline: {config['tagline']}")
    print(f"\nSystem prompt preview:")
    print(build_system_prompt(config)[:500] + "...")
