//! Tool definitions and execution for OpenClaudia
//!
//! Implements the core tools that make OpenClaudia an agent:
//! - Bash: Execute shell commands
//! - Read: Read file contents
//! - Write: Write/create files
//! - Edit: Make targeted edits to files
//!
//! Stateful mode adds memory tools:
//! - memory_save: Store information in archival memory
//! - memory_search: Search archival memory
//! - memory_update: Update existing memory
//! - core_memory_update: Update core memory sections

use crate::memory::{MemoryDb, SECTION_PERSONA, SECTION_PROJECT_INFO, SECTION_USER_PREFS};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Tool call from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Function call details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Result of executing a tool
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Get all tool definitions for the API request
pub fn get_tool_definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash/shell command and return the output. Use this for running commands, installing packages, git operations, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Returns the file content as text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a targeted edit to a file by replacing old_string with new_string. The old_string must match exactly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact string to find and replace"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The string to replace it with"
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories at a given path. Returns a list of entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list (defaults to current directory)"
                        }
                    },
                    "required": []
                }
            }
        }
    ])
}

/// Execute a tool call and return the result (non-stateful mode)
///
/// This is a convenience wrapper around `execute_tool_with_memory` for
/// when memory tools are not needed. Memory-related tool calls will
/// return an error indicating stateful mode is required.
pub fn execute_tool(tool_call: &ToolCall) -> ToolResult {
    execute_tool_with_memory(tool_call, None)
}

/// Execute a bash command
fn execute_bash(args: &HashMap<String, Value>) -> (String, bool) {
    let command = match args.get("command").and_then(|v| v.as_str()) {
        Some(cmd) => cmd,
        None => return ("Missing 'command' argument".to_string(), true),
    };

    // Use appropriate shell based on platform
    // On Windows, use PowerShell for better Unix command compatibility (ls, cat, curl, etc.)
    #[cfg(windows)]
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", command])
        .output();

    #[cfg(not(windows))]
    let output = Command::new("sh")
        .args(["-c", command])
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("stderr: ");
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result = "(command completed with no output)".to_string();
            }

            // Truncate if too long
            if result.len() > 50000 {
                result = format!("{}...\n(output truncated, {} total chars)",
                    &result[..50000], result.len());
            }

            (result, !output.status.success())
        }
        Err(e) => (format!("Failed to execute command: {}", e), true),
    }
}

/// Read a file's contents
fn execute_read_file(args: &HashMap<String, Value>) -> (String, bool) {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ("Missing 'path' argument".to_string(), true),
    };

    match fs::read_to_string(path) {
        Ok(content) => {
            // Add line numbers
            let numbered: Vec<String> = content
                .lines()
                .enumerate()
                .map(|(i, line)| format!("{:4}| {}", i + 1, line))
                .collect();

            let result = numbered.join("\n");

            // Truncate if too long
            if result.len() > 100000 {
                (format!("{}...\n(file truncated, {} total chars)",
                    &result[..100000], result.len()), false)
            } else {
                (result, false)
            }
        }
        Err(e) => (format!("Failed to read file '{}': {}", path, e), true),
    }
}

/// Write content to a file
fn execute_write_file(args: &HashMap<String, Value>) -> (String, bool) {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ("Missing 'path' argument".to_string(), true),
    };

    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ("Missing 'content' argument".to_string(), true),
    };

    // Create parent directories if needed
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                return (format!("Failed to create directories: {}", e), true);
            }
        }
    }

    match fs::write(path, content) {
        Ok(()) => (format!("Successfully wrote {} bytes to '{}'", content.len(), path), false),
        Err(e) => (format!("Failed to write file '{}': {}", path, e), true),
    }
}

/// Edit a file by replacing text
fn execute_edit_file(args: &HashMap<String, Value>) -> (String, bool) {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ("Missing 'path' argument".to_string(), true),
    };

    let old_string = match args.get("old_string").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ("Missing 'old_string' argument".to_string(), true),
    };

    let new_string = match args.get("new_string").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ("Missing 'new_string' argument".to_string(), true),
    };

    // Read the file
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return (format!("Failed to read file '{}': {}", path, e), true),
    };

    // Check if old_string exists
    if !content.contains(old_string) {
        return (format!("Could not find the specified text in '{}'. Make sure old_string matches exactly.", path), true);
    }

    // Count occurrences
    let count = content.matches(old_string).count();
    if count > 1 {
        return (format!("Found {} occurrences of the text. Please provide a more specific old_string that matches uniquely.", count), true);
    }

    // Make the replacement
    let new_content = content.replacen(old_string, new_string, 1);

    // Write back
    match fs::write(path, &new_content) {
        Ok(()) => (format!("Successfully edited '{}'. Replaced {} chars with {} chars.",
            path, old_string.len(), new_string.len()), false),
        Err(e) => (format!("Failed to write file '{}': {}", path, e), true),
    }
}

/// List files in a directory
fn execute_list_files(args: &HashMap<String, Value>) -> (String, bool) {
    let path = args.get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    match fs::read_dir(path) {
        Ok(entries) => {
            let mut items: Vec<String> = Vec::new();
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let file_type = entry.file_type().map(|ft| {
                    if ft.is_dir() { "/" } else { "" }
                }).unwrap_or("");
                items.push(format!("{}{}", name, file_type));
            }
            items.sort();
            (items.join("\n"), false)
        }
        Err(e) => (format!("Failed to list directory '{}': {}", path, e), true),
    }
}

/// Parse tool calls from a streaming response delta
/// Returns accumulated tool calls when complete
#[derive(Default, Debug)]
pub struct ToolCallAccumulator {
    pub tool_calls: Vec<PartialToolCall>,
}

#[derive(Default, Debug, Clone)]
pub struct PartialToolCall {
    pub index: usize,
    pub id: String,
    pub call_type: String,
    pub function_name: String,
    pub function_arguments: String,
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self { tool_calls: Vec::new() }
    }

    /// Process a delta from streaming response
    pub fn process_delta(&mut self, delta: &Value) {
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                // Ensure we have enough slots
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(PartialToolCall::default());
                }

                let partial = &mut self.tool_calls[index];
                partial.index = index;

                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    partial.id = id.to_string();
                }
                if let Some(t) = tc.get("type").and_then(|v| v.as_str()) {
                    partial.call_type = t.to_string();
                }
                if let Some(func) = tc.get("function") {
                    if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                        partial.function_name = name.to_string();
                    }
                    if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                        partial.function_arguments.push_str(args);
                    }
                }
            }
        }
    }

    /// Convert accumulated partials to complete tool calls
    pub fn finalize(&self) -> Vec<ToolCall> {
        self.tool_calls
            .iter()
            .filter(|tc| !tc.id.is_empty() && !tc.function_name.is_empty())
            .map(|tc| ToolCall {
                id: tc.id.clone(),
                call_type: if tc.call_type.is_empty() { "function".to_string() } else { tc.call_type.clone() },
                function: FunctionCall {
                    name: tc.function_name.clone(),
                    arguments: tc.function_arguments.clone(),
                },
            })
            .collect()
    }

    /// Check if we have any tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.iter().any(|tc| !tc.id.is_empty())
    }

    /// Clear the accumulator
    pub fn clear(&mut self) {
        self.tool_calls.clear();
    }
}

// === Memory Tools (Stateful Mode) ===

/// Get memory tool definitions for stateful mode
pub fn get_memory_tool_definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "memory_save",
                "description": "Save important information to archival memory for long-term storage. Use this to remember facts, decisions, patterns, and anything worth preserving across sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to save to memory"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional tags for categorizing the memory (e.g., ['architecture', 'decision'])"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search archival memory for relevant information. Use this to recall previously saved facts, decisions, or context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant memories"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_update",
                "description": "Update an existing memory entry by ID. Use this to correct or expand previously saved information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "The ID of the memory to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "The new content for the memory"
                        }
                    },
                    "required": ["id", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "core_memory_update",
                "description": "Update a core memory section. Core memory is always present in context. Sections: 'persona' (your identity/role), 'project_info' (project knowledge), 'user_preferences' (user's preferences).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["persona", "project_info", "user_preferences"],
                            "description": "The core memory section to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "The new content for this section"
                        }
                    },
                    "required": ["section", "content"]
                }
            }
        }
    ])
}

/// Get all tool definitions, optionally including memory tools
pub fn get_all_tool_definitions(stateful: bool) -> Value {
    let mut tools = get_tool_definitions();

    if stateful {
        if let (Some(base_arr), Some(memory_arr)) = (
            tools.as_array_mut(),
            get_memory_tool_definitions().as_array().cloned()
        ) {
            base_arr.extend(memory_arr);
        }
    }

    tools
}

/// Execute a tool call, with optional memory database for stateful mode
pub fn execute_tool_with_memory(tool_call: &ToolCall, memory_db: Option<&MemoryDb>) -> ToolResult {
    let args: HashMap<String, Value> = serde_json::from_str(&tool_call.function.arguments)
        .unwrap_or_default();

    let (content, is_error) = match tool_call.function.name.as_str() {
        // Standard tools
        "bash" => execute_bash(&args),
        "read_file" => execute_read_file(&args),
        "write_file" => execute_write_file(&args),
        "edit_file" => execute_edit_file(&args),
        "list_files" => execute_list_files(&args),

        // Memory tools (require stateful mode)
        "memory_save" => {
            if let Some(db) = memory_db {
                execute_memory_save(&args, db)
            } else {
                ("Memory tools require stateful mode (--stateful flag)".to_string(), true)
            }
        }
        "memory_search" => {
            if let Some(db) = memory_db {
                execute_memory_search(&args, db)
            } else {
                ("Memory tools require stateful mode (--stateful flag)".to_string(), true)
            }
        }
        "memory_update" => {
            if let Some(db) = memory_db {
                execute_memory_update(&args, db)
            } else {
                ("Memory tools require stateful mode (--stateful flag)".to_string(), true)
            }
        }
        "core_memory_update" => {
            if let Some(db) = memory_db {
                execute_core_memory_update(&args, db)
            } else {
                ("Memory tools require stateful mode (--stateful flag)".to_string(), true)
            }
        }

        _ => (format!("Unknown tool: {}", tool_call.function.name), true),
    };

    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content,
        is_error,
    }
}

/// Save content to archival memory
fn execute_memory_save(args: &HashMap<String, Value>, db: &MemoryDb) -> (String, bool) {
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ("Missing 'content' argument".to_string(), true),
    };

    let tags: Vec<String> = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    match db.memory_save(content, &tags) {
        Ok(id) => (format!("Memory saved with ID {}. Tags: {:?}", id, tags), false),
        Err(e) => (format!("Failed to save memory: {}", e), true),
    }
}

/// Search archival memory
fn execute_memory_search(args: &HashMap<String, Value>, db: &MemoryDb) -> (String, bool) {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ("Missing 'query' argument".to_string(), true),
    };

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    match db.memory_search(query, limit) {
        Ok(memories) => {
            if memories.is_empty() {
                return ("No memories found matching query.".to_string(), false);
            }

            let mut result = format!("Found {} memories:\n\n", memories.len());
            for mem in memories {
                result.push_str(&format!(
                    "[ID {}] ({})\n{}\nTags: {:?}\n\n",
                    mem.id, mem.updated_at, mem.content, mem.tags
                ));
            }
            (result, false)
        }
        Err(e) => (format!("Failed to search memory: {}", e), true),
    }
}

/// Update an existing memory
fn execute_memory_update(args: &HashMap<String, Value>, db: &MemoryDb) -> (String, bool) {
    let id = match args.get("id").and_then(|v| v.as_i64()) {
        Some(id) => id,
        None => return ("Missing 'id' argument".to_string(), true),
    };

    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ("Missing 'content' argument".to_string(), true),
    };

    match db.memory_update(id, content) {
        Ok(true) => (format!("Memory {} updated successfully.", id), false),
        Ok(false) => (format!("Memory {} not found.", id), true),
        Err(e) => (format!("Failed to update memory: {}", e), true),
    }
}

/// Update a core memory section
fn execute_core_memory_update(args: &HashMap<String, Value>, db: &MemoryDb) -> (String, bool) {
    let section = match args.get("section").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ("Missing 'section' argument".to_string(), true),
    };

    // Validate section name
    if ![SECTION_PERSONA, SECTION_PROJECT_INFO, SECTION_USER_PREFS].contains(&section) {
        return (format!("Invalid section '{}'. Must be: {}, {}, or {}",
            section, SECTION_PERSONA, SECTION_PROJECT_INFO, SECTION_USER_PREFS), true);
    }

    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ("Missing 'content' argument".to_string(), true),
    };

    match db.update_core_memory(section, content) {
        Ok(()) => (format!("Core memory section '{}' updated.", section), false),
        Err(e) => (format!("Failed to update core memory: {}", e), true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definitions() {
        let tools = get_tool_definitions();
        assert!(tools.is_array());
        let arr = tools.as_array().unwrap();
        assert!(arr.len() >= 4);
    }

    #[test]
    fn test_bash_execution() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("echo hello"));
        let (output, is_error) = execute_bash(&args);
        assert!(!is_error);
        assert!(output.contains("hello"));
    }

    #[test]
    fn test_list_files() {
        let args = HashMap::new();
        let (output, is_error) = execute_list_files(&args);
        assert!(!is_error);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_tool_call_accumulator() {
        let mut acc = ToolCallAccumulator::new();

        // Simulate streaming deltas
        acc.process_delta(&json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "{\"com"
                }
            }]
        }));

        acc.process_delta(&json!({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "arguments": "mand\": \"ls\"}"
                }
            }]
        }));

        let calls = acc.finalize();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, "{\"command\": \"ls\"}");
    }
}
