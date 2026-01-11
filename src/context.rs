//! Context Injector - Modifies API messages before sending to provider.
//!
//! Injects hook output as system messages using <system-reminder> tags.
//! Supports message array manipulation for context injection.

use crate::hooks::HookResult;
use crate::proxy::{ChatCompletionRequest, ChatMessage, MessageContent};

/// Wraps content in a system-reminder tag
fn wrap_system_reminder(content: &str) -> String {
    format!("<system-reminder>\n{}\n</system-reminder>", content)
}

/// Context injector that modifies requests based on hook results
pub struct ContextInjector;

impl ContextInjector {
    /// Inject context from hook results into the request
    ///
    /// This modifies the request in-place, adding system messages from hooks
    /// and applying any prompt modifications.
    pub fn inject(request: &mut ChatCompletionRequest, hook_result: &HookResult) {
        // Collect all system messages from hook outputs
        let system_messages: Vec<&str> = hook_result.system_messages();

        if system_messages.is_empty() {
            return;
        }

        // Combine all system messages into one wrapped reminder
        let combined = system_messages.join("\n\n");
        let reminder = wrap_system_reminder(&combined);

        // Find the last user message and inject the reminder after it
        // This ensures the reminder is seen just before the model responds
        if let Some(last_user_idx) = request.messages.iter().rposition(|m| m.role == "user") {
            // Append reminder to the last user message content
            Self::append_to_message(&mut request.messages[last_user_idx], &reminder);
        } else {
            // No user message found, add as a separate system message
            request.messages.push(ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(reminder),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    /// Apply prompt modification from hooks
    ///
    /// If a hook returned a modified prompt, this replaces the last user message.
    pub fn apply_prompt_modification(
        request: &mut ChatCompletionRequest,
        hook_result: &HookResult,
    ) {
        if let Some(modified_prompt) = hook_result.modified_prompt() {
            // Find and update the last user message
            if let Some(last_user) = request.messages.iter_mut().rev().find(|m| m.role == "user") {
                last_user.content = MessageContent::Text(modified_prompt.to_string());
            }
        }
    }

    /// Inject a system message at the beginning of the conversation
    pub fn inject_system_prefix(request: &mut ChatCompletionRequest, content: &str) {
        let reminder = wrap_system_reminder(content);

        // Check if first message is already a system message
        if let Some(first) = request.messages.first_mut() {
            if first.role == "system" {
                // Append to existing system message
                Self::append_to_message(first, &reminder);
                return;
            }
        }

        // Insert new system message at the beginning
        request.messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(reminder),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }

    /// Inject a system message at the end of the conversation (before response)
    pub fn inject_system_suffix(request: &mut ChatCompletionRequest, content: &str) {
        let reminder = wrap_system_reminder(content);

        // Find last user message and append
        if let Some(last_user_idx) = request.messages.iter().rposition(|m| m.role == "user") {
            Self::append_to_message(&mut request.messages[last_user_idx], &reminder);
        } else {
            // Add as separate system message at the end
            request.messages.push(ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(reminder),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    /// Append content to a message
    fn append_to_message(message: &mut ChatMessage, content: &str) {
        match &mut message.content {
            MessageContent::Text(text) => {
                text.push_str("\n\n");
                text.push_str(content);
            }
            MessageContent::Parts(parts) => {
                // Add as a new text part
                parts.push(crate::proxy::ContentPart {
                    content_type: "text".to_string(),
                    text: Some(content.to_string()),
                    image_url: None,
                });
            }
        }
    }

    /// Inject multiple context items from a rules engine or plugin
    pub fn inject_all(request: &mut ChatCompletionRequest, contexts: &[String]) {
        if contexts.is_empty() {
            return;
        }

        let combined = contexts.join("\n\n");
        Self::inject_system_suffix(request, &combined);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookOutput;

    fn create_test_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text("You are a helpful assistant.".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text("Hello!".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            extra: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_inject_system_messages() {
        let mut request = create_test_request();
        let hook_result = HookResult {
            allowed: true,
            outputs: vec![
                HookOutput {
                    system_message: Some("Remember to be concise.".to_string()),
                    ..Default::default()
                },
                HookOutput {
                    system_message: Some("Use markdown formatting.".to_string()),
                    ..Default::default()
                },
            ],
            errors: vec![],
        };

        ContextInjector::inject(&mut request, &hook_result);

        // Check that the user message was modified
        let user_msg = &request.messages[1];
        if let MessageContent::Text(text) = &user_msg.content {
            assert!(text.contains("<system-reminder>"));
            assert!(text.contains("Remember to be concise."));
            assert!(text.contains("Use markdown formatting."));
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_inject_system_prefix() {
        let mut request = create_test_request();
        ContextInjector::inject_system_prefix(&mut request, "Security context here");

        // Should append to existing system message
        let system_msg = &request.messages[0];
        if let MessageContent::Text(text) = &system_msg.content {
            assert!(text.contains("You are a helpful assistant."));
            assert!(text.contains("<system-reminder>"));
            assert!(text.contains("Security context here"));
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_apply_prompt_modification() {
        let mut request = create_test_request();
        let hook_result = HookResult {
            allowed: true,
            outputs: vec![HookOutput {
                prompt: Some("Modified prompt here".to_string()),
                ..Default::default()
            }],
            errors: vec![],
        };

        ContextInjector::apply_prompt_modification(&mut request, &hook_result);

        let user_msg = &request.messages[1];
        if let MessageContent::Text(text) = &user_msg.content {
            assert_eq!(text, "Modified prompt here");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_empty_hook_result() {
        let mut request = create_test_request();
        let original_len = request.messages.len();
        let hook_result = HookResult::allowed();

        ContextInjector::inject(&mut request, &hook_result);

        // Should not modify anything
        assert_eq!(request.messages.len(), original_len);
    }
}
