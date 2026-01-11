//! Session Manager - Tracks agent sessions with initializer/coding agent patterns.
//!
//! Implements two-part session architecture:
//! - Initializer Agent: First session, creates progress files and feature lists
//! - Coding Agent: Subsequent sessions, reads git logs and progress files
//!
//! Treats agents like shift workers with documented handoffs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Session state indicating the agent mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    /// First session - creates initial context
    Initializer,
    /// Subsequent sessions - continues from handoff
    Coding,
}

/// Progress tracking for a session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionProgress {
    /// Tasks completed in this session
    pub completed_tasks: Vec<String>,
    /// Tasks in progress
    pub in_progress_tasks: Vec<String>,
    /// Tasks pending
    pub pending_tasks: Vec<String>,
    /// Key decisions made
    pub decisions: Vec<String>,
    /// Files modified
    pub files_modified: Vec<String>,
    /// Notes for next session
    pub handoff_notes: String,
}

/// A single agent session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier
    pub id: String,
    /// Session mode (initializer or coding)
    pub mode: SessionMode,
    /// When the session started
    pub created_at: DateTime<Utc>,
    /// When the session was last updated
    pub updated_at: DateTime<Utc>,
    /// Session progress tracking
    pub progress: SessionProgress,
    /// Parent session ID if this is a continuation
    pub parent_session_id: Option<String>,
    /// Number of API requests in this session
    pub request_count: u64,
    /// Total tokens used (approximate)
    pub total_tokens: u64,
}

impl Session {
    /// Create a new initializer session
    pub fn new_initializer() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            mode: SessionMode::Initializer,
            created_at: now,
            updated_at: now,
            progress: SessionProgress::default(),
            parent_session_id: None,
            request_count: 0,
            total_tokens: 0,
        }
    }

    /// Create a new coding session continuing from a parent
    pub fn new_coding(parent_id: &str) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            mode: SessionMode::Coding,
            created_at: now,
            updated_at: now,
            progress: SessionProgress::default(),
            parent_session_id: Some(parent_id.to_string()),
            request_count: 0,
            total_tokens: 0,
        }
    }

    /// Update the session timestamp
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Increment request count
    pub fn increment_requests(&mut self) {
        self.request_count += 1;
        self.touch();
    }

    /// Add tokens to the total
    pub fn add_tokens(&mut self, tokens: u64) {
        self.total_tokens += tokens;
        self.touch();
    }

    /// Add a completed task
    pub fn complete_task(&mut self, task: impl Into<String>) {
        self.progress.completed_tasks.push(task.into());
        self.touch();
    }

    /// Add a file to the modified list
    pub fn add_modified_file(&mut self, path: impl Into<String>) {
        let path = path.into();
        if !self.progress.files_modified.contains(&path) {
            self.progress.files_modified.push(path);
            self.touch();
        }
    }

    /// Set handoff notes for the next session
    pub fn set_handoff_notes(&mut self, notes: impl Into<String>) {
        self.progress.handoff_notes = notes.into();
        self.touch();
    }

    /// Generate a handoff summary for the next agent
    pub fn generate_handoff(&self) -> String {
        let mut handoff = String::new();

        handoff.push_str("## Session Handoff\n\n");
        handoff.push_str(&format!("Previous Session: {}\n", self.id));
        handoff.push_str(&format!("Mode: {:?}\n", self.mode));
        handoff.push_str(&format!(
            "Duration: {} to {}\n\n",
            self.created_at.format("%Y-%m-%d %H:%M UTC"),
            self.updated_at.format("%Y-%m-%d %H:%M UTC")
        ));

        if !self.progress.completed_tasks.is_empty() {
            handoff.push_str("### Completed Tasks\n");
            for task in &self.progress.completed_tasks {
                handoff.push_str(&format!("- [x] {}\n", task));
            }
            handoff.push('\n');
        }

        if !self.progress.in_progress_tasks.is_empty() {
            handoff.push_str("### In Progress\n");
            for task in &self.progress.in_progress_tasks {
                handoff.push_str(&format!("- [ ] {}\n", task));
            }
            handoff.push('\n');
        }

        if !self.progress.pending_tasks.is_empty() {
            handoff.push_str("### Pending Tasks\n");
            for task in &self.progress.pending_tasks {
                handoff.push_str(&format!("- [ ] {}\n", task));
            }
            handoff.push('\n');
        }

        if !self.progress.decisions.is_empty() {
            handoff.push_str("### Key Decisions\n");
            for decision in &self.progress.decisions {
                handoff.push_str(&format!("- {}\n", decision));
            }
            handoff.push('\n');
        }

        if !self.progress.files_modified.is_empty() {
            handoff.push_str("### Files Modified\n");
            for file in &self.progress.files_modified {
                handoff.push_str(&format!("- {}\n", file));
            }
            handoff.push('\n');
        }

        if !self.progress.handoff_notes.is_empty() {
            handoff.push_str("### Notes for Next Session\n");
            handoff.push_str(&self.progress.handoff_notes);
            handoff.push('\n');
        }

        handoff
    }
}

/// Manages session lifecycle and persistence
#[derive(Debug, Clone)]
pub struct SessionManager {
    /// Directory for session persistence
    persist_dir: PathBuf,
    /// Current active session
    current_session: Option<Session>,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(persist_dir: impl Into<PathBuf>) -> Self {
        let persist_dir = persist_dir.into();

        // Ensure the directory exists
        if let Err(e) = fs::create_dir_all(&persist_dir) {
            warn!(error = %e, path = ?persist_dir, "Failed to create session directory");
        }

        Self {
            persist_dir,
            current_session: None,
        }
    }

    /// Get the current session, creating one if none exists
    pub fn get_or_create_session(&mut self) -> &Session {
        if self.current_session.is_none() {
            self.current_session = Some(self.create_session());
        }
        self.current_session.as_ref().unwrap()
    }

    /// Get the current session mutably
    pub fn get_session_mut(&mut self) -> Option<&mut Session> {
        self.current_session.as_mut()
    }

    /// Get the current session immutably
    pub fn get_session(&self) -> Option<&Session> {
        self.current_session.as_ref()
    }

    /// Create a new session (initializer or coding based on history)
    fn create_session(&self) -> Session {
        // Check if there's a previous session to continue from
        if let Some(last_session) = self.load_latest_session() {
            info!(
                parent_id = %last_session.id,
                "Creating coding session continuing from previous"
            );
            Session::new_coding(&last_session.id)
        } else {
            info!("Creating new initializer session");
            Session::new_initializer()
        }
    }

    /// Start a fresh initializer session
    pub fn start_initializer(&mut self) -> &Session {
        let session = Session::new_initializer();
        info!(session_id = %session.id, "Started initializer session");
        self.current_session = Some(session);
        self.current_session.as_ref().unwrap()
    }

    /// Start a coding session from a parent
    pub fn start_coding(&mut self, parent_id: &str) -> &Session {
        let session = Session::new_coding(parent_id);
        info!(
            session_id = %session.id,
            parent_id = %parent_id,
            "Started coding session"
        );
        self.current_session = Some(session);
        self.current_session.as_ref().unwrap()
    }

    /// End the current session and persist it
    pub fn end_session(&mut self, handoff_notes: Option<&str>) -> Option<Session> {
        if let Some(mut session) = self.current_session.take() {
            if let Some(notes) = handoff_notes {
                session.set_handoff_notes(notes);
            }

            // Persist the session
            if let Err(e) = self.persist_session(&session) {
                warn!(error = %e, "Failed to persist session");
            }

            info!(
                session_id = %session.id,
                requests = session.request_count,
                "Ended session"
            );

            Some(session)
        } else {
            None
        }
    }

    /// Persist a session to disk
    fn persist_session(&self, session: &Session) -> anyhow::Result<()> {
        let filename = format!("{}.json", session.id);
        let path = self.persist_dir.join(&filename);

        let json = serde_json::to_string_pretty(session)?;
        fs::write(&path, json)?;

        debug!(path = ?path, "Persisted session");

        // Also update the "latest" symlink/file
        let latest_path = self.persist_dir.join("latest.json");
        fs::write(&latest_path, serde_json::to_string_pretty(session)?)?;

        // Generate and save handoff document
        let handoff_path = self.persist_dir.join("handoff.md");
        fs::write(&handoff_path, session.generate_handoff())?;

        Ok(())
    }

    /// Load a session by ID
    pub fn load_session(&self, session_id: &str) -> Option<Session> {
        let path = self.persist_dir.join(format!("{}.json", session_id));
        self.load_session_from_path(&path)
    }

    /// Load the most recent session
    pub fn load_latest_session(&self) -> Option<Session> {
        let path = self.persist_dir.join("latest.json");
        self.load_session_from_path(&path)
    }

    /// Load a session from a file path
    fn load_session_from_path(&self, path: &Path) -> Option<Session> {
        if !path.exists() {
            return None;
        }

        match fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(session) => Some(session),
                Err(e) => {
                    warn!(error = %e, path = ?path, "Failed to parse session file");
                    None
                }
            },
            Err(e) => {
                warn!(error = %e, path = ?path, "Failed to read session file");
                None
            }
        }
    }

    /// Get the handoff context from the last session
    pub fn get_handoff_context(&self) -> Option<String> {
        let handoff_path = self.persist_dir.join("handoff.md");
        fs::read_to_string(&handoff_path).ok()
    }

    /// List all persisted sessions
    pub fn list_sessions(&self) -> Vec<Session> {
        let mut sessions = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.persist_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    // Skip latest.json as it's a copy
                    if path.file_stem().map(|s| s == "latest").unwrap_or(false) {
                        continue;
                    }
                    if let Some(session) = self.load_session_from_path(&path) {
                        sessions.push(session);
                    }
                }
            }
        }

        // Sort by created_at descending
        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        sessions
    }

    /// Clean up old sessions, keeping only the most recent N
    pub fn cleanup_old_sessions(&self, keep_count: usize) {
        let sessions = self.list_sessions();

        if sessions.len() <= keep_count {
            return;
        }

        for session in sessions.iter().skip(keep_count) {
            let path = self.persist_dir.join(format!("{}.json", session.id));
            if let Err(e) = fs::remove_file(&path) {
                warn!(error = %e, path = ?path, "Failed to remove old session");
            } else {
                debug!(session_id = %session.id, "Removed old session");
            }
        }
    }
}

/// Context to inject at session start based on mode
pub fn get_session_context(session: &Session) -> String {
    match session.mode {
        SessionMode::Initializer => r#"## Session Context: Initializer Agent

You are the first agent working on this task. Your responsibilities:
1. Understand the full scope of the work
2. Create a clear plan with actionable steps
3. Document key decisions and rationale
4. Set up any necessary project structure
5. Prepare detailed handoff notes for subsequent sessions

Focus on establishing a solid foundation that future agents can build upon."#
            .to_string(),
        SessionMode::Coding => {
            let mut context = r#"## Session Context: Coding Agent

You are continuing work from a previous session. Your responsibilities:
1. Review the handoff notes from the previous session
2. Continue from where the last agent left off
3. Track your progress and decisions
4. Prepare handoff notes if you won't complete the task

"#
            .to_string();

            // Add parent session info if available
            if let Some(parent_id) = &session.parent_session_id {
                context.push_str(&format!("Previous session ID: {}\n", parent_id));
            }

            context
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new_initializer_session() {
        let session = Session::new_initializer();
        assert_eq!(session.mode, SessionMode::Initializer);
        assert!(session.parent_session_id.is_none());
        assert_eq!(session.request_count, 0);
    }

    #[test]
    fn test_new_coding_session() {
        let session = Session::new_coding("parent-123");
        assert_eq!(session.mode, SessionMode::Coding);
        assert_eq!(session.parent_session_id, Some("parent-123".to_string()));
    }

    #[test]
    fn test_session_progress() {
        let mut session = Session::new_initializer();
        session.complete_task("Task 1");
        session.add_modified_file("src/main.rs");
        session.set_handoff_notes("Continue with task 2");

        assert_eq!(session.progress.completed_tasks.len(), 1);
        assert_eq!(session.progress.files_modified.len(), 1);
        assert!(!session.progress.handoff_notes.is_empty());
    }

    #[test]
    fn test_generate_handoff() {
        let mut session = Session::new_initializer();
        session.complete_task("Implemented feature X");
        session
            .progress
            .pending_tasks
            .push("Test feature X".to_string());
        session.set_handoff_notes("Feature X works but needs tests");

        let handoff = session.generate_handoff();
        assert!(handoff.contains("Implemented feature X"));
        assert!(handoff.contains("Test feature X"));
        assert!(handoff.contains("needs tests"));
    }

    #[test]
    fn test_session_manager_persistence() {
        let dir = TempDir::new().unwrap();
        let mut manager = SessionManager::new(dir.path().join("sessions"));

        // Create and end a session
        let session = manager.get_or_create_session().clone();
        assert_eq!(session.mode, SessionMode::Initializer);

        manager.end_session(Some("Test handoff notes"));

        // Load it back
        let loaded = manager.load_session(&session.id);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().id, session.id);
    }

    #[test]
    fn test_session_manager_coding_continuation() {
        let dir = TempDir::new().unwrap();
        let mut manager = SessionManager::new(dir.path().join("sessions"));

        // First session
        let first = manager.get_or_create_session().clone();
        manager.end_session(None);

        // Second session should be coding mode
        let second = manager.get_or_create_session().clone();
        assert_eq!(second.mode, SessionMode::Coding);
        assert_eq!(second.parent_session_id, Some(first.id));
    }
}
