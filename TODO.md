## TODO:

1. [x] Add shell_cmd tool (default active for coder agent) always asking for user permission.
2. [x] Add a permission request timeout (default 5 minutes) option. If the user doesn't respond within the timeout, the tool call will be canceled with the reason "User is AFK, try another method if possible"