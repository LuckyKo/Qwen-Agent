## TODO:

1. [ ] Add a way for an agent to resend a tool request with the previous arguments (something like "USE_LAST_ARGS" specified for a certain element, and we use the cached version of the argument from a previous tool call instead of needeing to regenerate the full entry). Would be useful for minor schema errors, like failing the path argument when the agent coded a whole page.
2. [ ] 