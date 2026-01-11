//! Stateful memory module for OpenClaudia
//!
//! Implements Letta/MemGPT-style archival memory using SQLite.
//! Each project gets its own memory database that persists across sessions.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

/// Memory database file name
const MEMORY_DB_NAME: &str = "memory.db";

/// Core memory section names
pub const SECTION_PERSONA: &str = "persona";
pub const SECTION_PROJECT_INFO: &str = "project_info";
pub const SECTION_USER_PREFS: &str = "user_preferences";

/// A single archival memory entry
#[derive(Debug, Clone)]
pub struct ArchivalMemory {
    pub id: i64,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Core memory block (always in context)
#[derive(Debug, Clone)]
pub struct CoreMemory {
    pub section: String,
    pub content: String,
    pub updated_at: String,
}

/// Memory database handle
pub struct MemoryDb {
    conn: Connection,
    path: PathBuf,
}

impl MemoryDb {
    /// Open or create memory database at the specified path
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open memory database at {:?}", path))?;

        let mut db = Self {
            conn,
            path: path.to_path_buf(),
        };

        db.ensure_schema()?;
        Ok(db)
    }

    /// Open or create memory database in .openclaudia directory
    pub fn open_for_project(project_dir: &Path) -> Result<Self> {
        let openclaudia_dir = project_dir.join(".openclaudia");
        std::fs::create_dir_all(&openclaudia_dir)
            .with_context(|| format!("Failed to create .openclaudia directory at {:?}", openclaudia_dir))?;

        let db_path = openclaudia_dir.join(MEMORY_DB_NAME);
        Self::open(&db_path)
    }

    /// Get the database path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ensure database schema exists
    fn ensure_schema(&mut self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            -- Archival memory table for long-term storage
            CREATE TABLE IF NOT EXISTS archival_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS archival_memory_fts USING fts5(
                content,
                tags,
                content=archival_memory,
                content_rowid=id
            );

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS archival_memory_ai AFTER INSERT ON archival_memory BEGIN
                INSERT INTO archival_memory_fts(rowid, content, tags)
                VALUES (new.id, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS archival_memory_ad AFTER DELETE ON archival_memory BEGIN
                INSERT INTO archival_memory_fts(archival_memory_fts, rowid, content, tags)
                VALUES('delete', old.id, old.content, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS archival_memory_au AFTER UPDATE ON archival_memory BEGIN
                INSERT INTO archival_memory_fts(archival_memory_fts, rowid, content, tags)
                VALUES('delete', old.id, old.content, old.tags);
                INSERT INTO archival_memory_fts(rowid, content, tags)
                VALUES (new.id, new.content, new.tags);
            END;

            -- Core memory table (always in context)
            CREATE TABLE IF NOT EXISTS core_memory (
                section TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            -- Initialize default core memory sections if not exist
            INSERT OR IGNORE INTO core_memory (section, content) VALUES
                ('persona', 'I am an AI assistant helping with this project. I will learn about the codebase and remember important details across sessions.'),
                ('project_info', 'No project information recorded yet.'),
                ('user_preferences', 'No user preferences recorded yet.');
            "#,
        ).context("Failed to create memory database schema")?;

        Ok(())
    }

    // === Archival Memory Operations ===

    /// Save a new memory entry
    pub fn memory_save(&self, content: &str, tags: &[String]) -> Result<i64> {
        let tags_str = tags.join(",");
        self.conn.execute(
            "INSERT INTO archival_memory (content, tags) VALUES (?1, ?2)",
            params![content, tags_str],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Search archival memory using full-text search
    pub fn memory_search(&self, query: &str, limit: usize) -> Result<Vec<ArchivalMemory>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT am.id, am.content, am.tags, am.created_at, am.updated_at,
                   bm25(archival_memory_fts) as rank
            FROM archival_memory_fts
            JOIN archival_memory am ON archival_memory_fts.rowid = am.id
            WHERE archival_memory_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
            "#,
        )?;

        let memories = stmt
            .query_map(params![query, limit as i64], |row| {
                Ok(ArchivalMemory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    tags: row.get::<_, String>(2)?
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(memories)
    }

    /// Get a memory by ID
    pub fn memory_get(&self, id: i64) -> Result<Option<ArchivalMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, tags, created_at, updated_at FROM archival_memory WHERE id = ?1",
        )?;

        let memory = stmt
            .query_row(params![id], |row| {
                Ok(ArchivalMemory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    tags: row.get::<_, String>(2)?
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            })
            .optional()?;

        Ok(memory)
    }

    /// Update an existing memory
    pub fn memory_update(&self, id: i64, content: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "UPDATE archival_memory SET content = ?1, updated_at = datetime('now') WHERE id = ?2",
            params![content, id],
        )?;
        Ok(rows > 0)
    }

    /// Delete a memory entry
    pub fn memory_delete(&self, id: i64) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM archival_memory WHERE id = ?1",
            params![id],
        )?;
        Ok(rows > 0)
    }

    /// List recent memories
    pub fn memory_list(&self, limit: usize) -> Result<Vec<ArchivalMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, tags, created_at, updated_at FROM archival_memory ORDER BY updated_at DESC LIMIT ?1",
        )?;

        let memories = stmt
            .query_map(params![limit as i64], |row| {
                Ok(ArchivalMemory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    tags: row.get::<_, String>(2)?
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(memories)
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Result<MemoryStats> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM archival_memory",
            [],
            |row| row.get(0),
        )?;

        let total_size: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM archival_memory",
            [],
            |row| row.get(0),
        )?;

        let last_updated: Option<String> = self.conn.query_row(
            "SELECT MAX(updated_at) FROM archival_memory",
            [],
            |row| row.get(0),
        )?;

        Ok(MemoryStats {
            count: count as usize,
            total_size: total_size as usize,
            last_updated,
        })
    }

    // === Core Memory Operations ===

    /// Get all core memory sections
    pub fn get_core_memory(&self) -> Result<Vec<CoreMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT section, content, updated_at FROM core_memory ORDER BY section",
        )?;

        let memories = stmt
            .query_map([], |row| {
                Ok(CoreMemory {
                    section: row.get(0)?,
                    content: row.get(1)?,
                    updated_at: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(memories)
    }

    /// Get a specific core memory section
    pub fn get_core_memory_section(&self, section: &str) -> Result<Option<CoreMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT section, content, updated_at FROM core_memory WHERE section = ?1",
        )?;

        let memory = stmt
            .query_row(params![section], |row| {
                Ok(CoreMemory {
                    section: row.get(0)?,
                    content: row.get(1)?,
                    updated_at: row.get(2)?,
                })
            })
            .optional()?;

        Ok(memory)
    }

    /// Update a core memory section
    pub fn update_core_memory(&self, section: &str, content: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO core_memory (section, content, updated_at) VALUES (?1, ?2, datetime('now'))",
            params![section, content],
        )?;
        Ok(())
    }

    /// Format core memory for injection into system prompt
    pub fn format_core_memory_for_prompt(&self) -> Result<String> {
        let core = self.get_core_memory()?;

        let mut output = String::from("<core_memory>\n");

        for mem in core {
            output.push_str(&format!("<{}>\n{}\n</{}>\n", mem.section, mem.content, mem.section));
        }

        output.push_str("</core_memory>");
        Ok(output)
    }

    /// Clear all archival memory (keeps core memory)
    pub fn clear_archival_memory(&self) -> Result<usize> {
        let rows = self.conn.execute("DELETE FROM archival_memory", [])?;
        Ok(rows)
    }

    /// Reset everything including core memory
    pub fn reset_all(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            DELETE FROM archival_memory;
            DELETE FROM core_memory;
            INSERT INTO core_memory (section, content) VALUES
                ('persona', 'I am an AI assistant helping with this project. I will learn about the codebase and remember important details across sessions.'),
                ('project_info', 'No project information recorded yet.'),
                ('user_preferences', 'No user preferences recorded yet.');
            "#,
        )?;
        Ok(())
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub count: usize,
    pub total_size: usize,
    pub last_updated: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_db_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let _db = MemoryDb::open(&db_path).unwrap();
        assert!(db_path.exists());
    }

    #[test]
    fn test_memory_save_and_search() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDb::open(&db_path).unwrap();

        // Save a memory
        let id = db.memory_save("The project uses Rust and tokio for async", &["rust".into(), "async".into()]).unwrap();
        assert!(id > 0);

        // Search for it
        let results = db.memory_search("Rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn test_memory_update() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDb::open(&db_path).unwrap();

        let id = db.memory_save("Original content", &[]).unwrap();
        db.memory_update(id, "Updated content").unwrap();

        let mem = db.memory_get(id).unwrap().unwrap();
        assert_eq!(mem.content, "Updated content");
    }

    #[test]
    fn test_core_memory() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDb::open(&db_path).unwrap();

        // Check default core memory exists
        let core = db.get_core_memory().unwrap();
        assert_eq!(core.len(), 3);

        // Update persona
        db.update_core_memory("persona", "I am the OpenClaudia assistant").unwrap();

        let persona = db.get_core_memory_section("persona").unwrap().unwrap();
        assert_eq!(persona.content, "I am the OpenClaudia assistant");
    }

    #[test]
    fn test_format_core_memory() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDb::open(&db_path).unwrap();

        let formatted = db.format_core_memory_for_prompt().unwrap();
        assert!(formatted.contains("<core_memory>"));
        assert!(formatted.contains("<persona>"));
        assert!(formatted.contains("</core_memory>"));
    }
}
