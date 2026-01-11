#!/usr/bin/env python3
"""Example SessionStart hook for OpenClaudia.

This hook runs when a new session starts.
Output JSON to stdout to inject context into the conversation.
"""

import json
import sys
import os

def main():
    # Read hook input from stdin
    input_data = json.load(sys.stdin)

    # Get project information
    cwd = input_data.get("cwd", os.getcwd())

    # Output context to inject
    output = {
        "systemMessage": f"Working directory: {cwd}"
    }

    print(json.dumps(output))

if __name__ == "__main__":
    main()
