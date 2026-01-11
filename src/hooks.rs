//! Hook Engine - Executes hooks at key moments in the agent lifecycle.
//!
//! Supports 12 event types and two hook mechanisms:
//! - Command hooks: Execute shell commands with JSON stdin/stdout
//! - Prompt hooks: Inject prompts into the conversation
//!
//! Exit codes:
//! - 0: Success (allow)
//! - 2: Block the action

use crate::config::{Hook, HookEntry, HooksConfig};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// All hook event types supported by OpenClaudia
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    /// Fired when a new session starts
    SessionStart,
    /// Fired when a session ends
    SessionEnd,
    /// Fired before a tool is executed
    PreToolUse,
    /// Fired after a tool executes successfully
    PostToolUse,
    /// Fired after a tool execution fails
    PostToolUseFailure,
    /// Fired when user submits a prompt
    UserPromptSubmit,
    /// Fired when the agent stops
    Stop,
    /// Fired when a subagent starts
    SubagentStart,
    /// Fired when a subagent stops
    SubagentStop,
    /// Fired before context compaction
    PreCompact,
    /// Fired when a permission is requested
    PermissionRequest,
    /// Fired for notifications
    Notification,
}

impl HookEvent {
    /// Get the config field name for this event
    pub fn config_key(&self) -> &'static str {
        match self {
            HookEvent::SessionStart => "session_start",
            HookEvent::SessionEnd => "session_end",
            HookEvent::PreToolUse => "pre_tool_use",
            HookEvent::PostToolUse => "post_tool_use",
            HookEvent::PostToolUseFailure => "post_tool_use_failure",
            HookEvent::UserPromptSubmit => "user_prompt_submit",
            HookEvent::Stop => "stop",
            HookEvent::SubagentStart => "subagent_start",
            HookEvent::SubagentStop => "subagent_stop",
            HookEvent::PreCompact => "pre_compact",
            HookEvent::PermissionRequest => "permission_request",
            HookEvent::Notification => "notification",
        }
    }
}

/// Input provided to hooks via stdin
#[derive(Debug, Clone, Serialize)]
pub struct HookInput {
    /// The event type that triggered this hook
    pub event: HookEvent,
    /// Current working directory
    pub cwd: String,
    /// Session ID if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Tool name for tool-related events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Tool input for tool-related events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<Value>,
    /// User prompt for UserPromptSubmit event
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Additional context data
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl HookInput {
    pub fn new(event: HookEvent) -> Self {
        Self {
            event,
            cwd: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            session_id: None,
            tool_name: None,
            tool_input: None,
            prompt: None,
            extra: HashMap::new(),
        }
    }

    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    pub fn with_tool(mut self, name: impl Into<String>, input: Value) -> Self {
        self.tool_name = Some(name.into());
        self.tool_input = Some(input);
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// Output from a hook execution
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct HookOutput {
    /// Decision: "allow", "deny", or "ask"
    pub decision: Option<String>,
    /// Reason for the decision
    pub reason: Option<String>,
    /// System message to inject
    #[serde(rename = "systemMessage")]
    pub system_message: Option<String>,
    /// Modified prompt (for UserPromptSubmit)
    pub prompt: Option<String>,
    /// Additional data from the hook
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Result of running hooks
#[derive(Debug, Clone)]
pub struct HookResult {
    /// Whether the action should be allowed
    pub allowed: bool,
    /// Combined outputs from all hooks
    pub outputs: Vec<HookOutput>,
    /// Any errors that occurred
    pub errors: Vec<HookError>,
}

impl HookResult {
    pub fn allowed() -> Self {
        Self {
            allowed: true,
            outputs: vec![],
            errors: vec![],
        }
    }

    pub fn denied(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            outputs: vec![HookOutput {
                decision: Some("deny".to_string()),
                reason: Some(reason.into()),
                ..Default::default()
            }],
            errors: vec![],
        }
    }

    /// Get all system messages from hook outputs
    pub fn system_messages(&self) -> Vec<&str> {
        self.outputs
            .iter()
            .filter_map(|o| o.system_message.as_deref())
            .collect()
    }

    /// Get modified prompt if any hook provided one
    pub fn modified_prompt(&self) -> Option<&str> {
        self.outputs.iter().find_map(|o| o.prompt.as_deref())
    }
}

/// Errors that can occur during hook execution
#[derive(Error, Debug, Clone)]
pub enum HookError {
    #[error("Hook timed out after {0} seconds")]
    Timeout(u64),

    #[error("Hook command failed: {0}")]
    CommandFailed(String),

    #[error("Hook output parse error: {0}")]
    ParseError(String),

    #[error("Hook blocked action: {0}")]
    Blocked(String),

    #[error("Invalid matcher regex: {0}")]
    InvalidMatcher(String),
}

/// The hook engine that executes hooks
#[derive(Clone)]
pub struct HookEngine {
    config: HooksConfig,
}

impl HookEngine {
    pub fn new(config: HooksConfig) -> Self {
        Self { config }
    }

    /// Run all matching hooks for an event
    pub async fn run(&self, event: HookEvent, input: &HookInput) -> HookResult {
        let entries = self.get_entries_for_event(event);

        if entries.is_empty() {
            return HookResult::allowed();
        }

        let matcher_context = self.get_matcher_context(input);

        // Filter entries by matcher
        let matching_entries: Vec<&HookEntry> = entries
            .iter()
            .filter(|entry| self.matches_entry(entry, &matcher_context))
            .collect();

        if matching_entries.is_empty() {
            return HookResult::allowed();
        }

        info!(
            event = ?event,
            count = matching_entries.len(),
            "Running hooks"
        );

        // Collect all hooks to run
        let mut hooks_to_run: Vec<(&Hook, u64)> = Vec::new();
        for entry in &matching_entries {
            for hook in &entry.hooks {
                let timeout_secs = match hook {
                    Hook::Command { timeout, .. } => *timeout,
                    Hook::Prompt { timeout, .. } => *timeout,
                };
                hooks_to_run.push((hook, timeout_secs));
            }
        }

        // Run hooks in parallel
        let input_json = serde_json::to_string(input).unwrap_or_default();
        let futures: Vec<_> = hooks_to_run
            .iter()
            .map(|(hook, timeout_secs)| self.run_hook(hook, &input_json, *timeout_secs))
            .collect();

        let results = futures::future::join_all(futures).await;

        // Combine results
        let mut hook_result = HookResult::allowed();
        for result in results {
            match result {
                Ok((output, exit_code)) => {
                    // Exit code 2 means block
                    if exit_code == 2 {
                        hook_result.allowed = false;
                        let reason = output
                            .reason
                            .clone()
                            .unwrap_or_else(|| "Hook blocked action".to_string());
                        warn!(reason = %reason, "Hook blocked action");
                    }
                    // Check decision field
                    if let Some(decision) = &output.decision {
                        if decision == "deny" || decision == "block" {
                            hook_result.allowed = false;
                        }
                    }
                    hook_result.outputs.push(output);
                }
                Err(e) => {
                    error!(error = %e, "Hook execution failed");
                    hook_result.errors.push(e);
                }
            }
        }

        hook_result
    }

    /// Get hook entries for a specific event
    fn get_entries_for_event(&self, event: HookEvent) -> &[HookEntry] {
        match event {
            HookEvent::SessionStart => &self.config.session_start,
            HookEvent::SessionEnd => &self.config.session_end,
            HookEvent::PreToolUse => &self.config.pre_tool_use,
            HookEvent::PostToolUse => &self.config.post_tool_use,
            HookEvent::UserPromptSubmit => &self.config.user_prompt_submit,
            HookEvent::Stop => &self.config.stop,
            // Events not yet in config (return empty)
            HookEvent::PostToolUseFailure
            | HookEvent::SubagentStart
            | HookEvent::SubagentStop
            | HookEvent::PreCompact
            | HookEvent::PermissionRequest
            | HookEvent::Notification => &[],
        }
    }

    /// Get the string to match against for this input
    fn get_matcher_context(&self, input: &HookInput) -> String {
        // For tool events, match against tool name
        if let Some(tool_name) = &input.tool_name {
            return tool_name.clone();
        }
        // For other events, match against prompt or event name
        if let Some(prompt) = &input.prompt {
            return prompt.clone();
        }
        input.event.config_key().to_string()
    }

    /// Check if a hook entry matches the current context
    fn matches_entry(&self, entry: &HookEntry, context: &str) -> bool {
        match &entry.matcher {
            None => true, // No matcher means always match
            Some(pattern) => match self.validate_and_match(pattern, context) {
                Ok(matched) => matched,
                Err(e) => {
                    warn!(pattern = %pattern, error = %e, "Matcher validation failed");
                    false
                }
            },
        }
    }

    /// Validate regex pattern and check for match
    fn validate_and_match(&self, pattern: &str, context: &str) -> Result<bool, HookError> {
        // Check for invalid patterns
        if pattern.is_empty() {
            return Err(HookError::InvalidMatcher("Empty pattern".to_string()));
        }

        match Regex::new(pattern) {
            Ok(re) => Ok(re.is_match(context)),
            Err(e) => Err(HookError::InvalidMatcher(e.to_string())),
        }
    }

    /// Parse hook output and handle errors
    fn parse_hook_output(stdout: &str) -> Result<HookOutput, HookError> {
        if stdout.trim().is_empty() {
            return Ok(HookOutput::default());
        }

        serde_json::from_str(stdout)
            .map_err(|e| HookError::ParseError(format!("Failed to parse hook output: {}", e)))
    }

    /// Check if an action should be blocked based on hook result
    pub fn check_blocked(result: &HookResult) -> Result<(), HookError> {
        if !result.allowed {
            let reason = result
                .outputs
                .first()
                .and_then(|o| o.reason.clone())
                .unwrap_or_else(|| "Action blocked by hook".to_string());
            Err(HookError::Blocked(reason))
        } else {
            Ok(())
        }
    }

    /// Run a single hook
    async fn run_hook(
        &self,
        hook: &Hook,
        input_json: &str,
        timeout_secs: u64,
    ) -> Result<(HookOutput, i32), HookError> {
        match hook {
            Hook::Command { command, .. } => {
                self.run_command_hook(command, input_json, timeout_secs)
                    .await
            }
            Hook::Prompt { prompt, .. } => {
                // Prompt hooks just return the prompt as system message
                Ok((
                    HookOutput {
                        system_message: Some(prompt.clone()),
                        ..Default::default()
                    },
                    0,
                ))
            }
        }
    }

    /// Execute a command hook
    async fn run_command_hook(
        &self,
        command: &str,
        input_json: &str,
        timeout_secs: u64,
    ) -> Result<(HookOutput, i32), HookError> {
        debug!(command = %command, "Running command hook");

        // Determine shell based on platform
        let (shell, shell_arg) = if cfg!(windows) {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };

        let mut child = Command::new(shell)
            .arg(shell_arg)
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env(
                "CLAUDE_PROJECT_DIR",
                std::env::current_dir().unwrap_or_default(),
            )
            .spawn()
            .map_err(|e| HookError::CommandFailed(e.to_string()))?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(input_json.as_bytes()).await;
        }

        // Wait for completion with timeout
        let result = timeout(Duration::from_secs(timeout_secs), child.wait_with_output()).await;

        match result {
            Ok(Ok(output)) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if !stderr.is_empty() {
                    debug!(stderr = %stderr, "Hook stderr");
                }

                // Parse JSON output if present
                let hook_output = match Self::parse_hook_output(&stdout) {
                    Ok(output) => output,
                    Err(e) => {
                        warn!(error = %e, stdout = %stdout, "Failed to parse hook output");
                        HookOutput::default()
                    }
                };

                Ok((hook_output, exit_code))
            }
            Ok(Err(e)) => Err(HookError::CommandFailed(e.to_string())),
            Err(_) => Err(HookError::Timeout(timeout_secs)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_event_config_keys() {
        assert_eq!(HookEvent::SessionStart.config_key(), "session_start");
        assert_eq!(HookEvent::PreToolUse.config_key(), "pre_tool_use");
        assert_eq!(
            HookEvent::UserPromptSubmit.config_key(),
            "user_prompt_submit"
        );
    }

    #[test]
    fn test_hook_input_builder() {
        let input = HookInput::new(HookEvent::PreToolUse)
            .with_session_id("test-session")
            .with_tool("Write", serde_json::json!({"path": "/tmp/test"}));

        assert_eq!(input.event, HookEvent::PreToolUse);
        assert_eq!(input.session_id, Some("test-session".to_string()));
        assert_eq!(input.tool_name, Some("Write".to_string()));
    }

    #[test]
    fn test_hook_result_system_messages() {
        let result = HookResult {
            allowed: true,
            outputs: vec![
                HookOutput {
                    system_message: Some("Message 1".to_string()),
                    ..Default::default()
                },
                HookOutput {
                    system_message: Some("Message 2".to_string()),
                    ..Default::default()
                },
            ],
            errors: vec![],
        };

        let messages = result.system_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], "Message 1");
        assert_eq!(messages[1], "Message 2");
    }

    #[tokio::test]
    async fn test_empty_hooks_config() {
        let engine = HookEngine::new(HooksConfig::default());
        let input = HookInput::new(HookEvent::SessionStart);
        let result = engine.run(HookEvent::SessionStart, &input).await;

        assert!(result.allowed);
        assert!(result.outputs.is_empty());
    }
}
