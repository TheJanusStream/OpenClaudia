//! Plugin API - Load and manage plugins for OpenClaudia.
//!
//! Plugins are loaded from:
//! - ~/.openclaudia/plugins/ (user plugins)
//! - .openclaudia/plugins/ (project plugins)
//!
//! Each plugin must have a manifest.json file defining its hooks and commands.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Plugin manifest (manifest.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    #[serde(default)]
    pub description: Option<String>,
    /// Plugin author
    #[serde(default)]
    pub author: Option<String>,
    /// Hooks provided by this plugin
    #[serde(default)]
    pub hooks: Vec<PluginHook>,
    /// Commands provided by this plugin
    #[serde(default)]
    pub commands: Vec<PluginCommand>,
    /// MCP servers to connect
    #[serde(default)]
    pub mcp_servers: Vec<PluginMcpServer>,
    /// Minimum OpenClaudia version required
    #[serde(default)]
    pub min_version: Option<String>,
    /// Plugin capabilities required
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Hook definition in a plugin manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHook {
    /// Hook event type (session_start, pre_tool_use, etc.)
    pub event: String,
    /// Matcher pattern for the hook
    #[serde(default)]
    pub matcher: Option<String>,
    /// Hook type (command or prompt)
    #[serde(rename = "type")]
    pub hook_type: String,
    /// Command to run (for command hooks)
    #[serde(default)]
    pub command: Option<String>,
    /// Prompt to inject (for prompt hooks)
    #[serde(default)]
    pub prompt: Option<String>,
    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    30
}

/// Command definition in a plugin manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCommand {
    /// Command name (used as /command)
    pub name: String,
    /// Command description
    #[serde(default)]
    pub description: Option<String>,
    /// Script to run
    pub script: String,
    /// Arguments schema
    #[serde(default)]
    pub args: Option<serde_json::Value>,
}

/// MCP server definition in a plugin manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMcpServer {
    /// Server name
    pub name: String,
    /// Transport type (stdio or http)
    pub transport: String,
    /// Command to run (for stdio)
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments for the command
    #[serde(default)]
    pub args: Vec<String>,
    /// URL (for http)
    #[serde(default)]
    pub url: Option<String>,
}

/// A loaded plugin
#[derive(Debug, Clone)]
pub struct Plugin {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Path to the plugin directory
    pub path: PathBuf,
    /// Whether the plugin is enabled
    pub enabled: bool,
}

impl Plugin {
    /// Load a plugin from a directory
    pub fn load(path: &Path) -> Result<Self, PluginError> {
        let manifest_path = path.join("manifest.json");

        if !manifest_path.exists() {
            return Err(PluginError::ManifestNotFound(path.to_path_buf()));
        }

        let manifest_content =
            fs::read_to_string(&manifest_path).map_err(|e| PluginError::IoError(e.to_string()))?;

        let manifest: PluginManifest = serde_json::from_str(&manifest_content)
            .map_err(|e| PluginError::InvalidManifest(e.to_string()))?;

        // Validate manifest
        Self::validate_manifest(&manifest)?;

        Ok(Self {
            manifest,
            path: path.to_path_buf(),
            enabled: true,
        })
    }

    /// Validate the plugin manifest
    fn validate_manifest(manifest: &PluginManifest) -> Result<(), PluginError> {
        if manifest.name.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin name is required".to_string(),
            ));
        }

        if manifest.version.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin version is required".to_string(),
            ));
        }

        // Validate hooks
        for hook in &manifest.hooks {
            if hook.event.is_empty() {
                return Err(PluginError::InvalidManifest(
                    "Hook event is required".to_string(),
                ));
            }

            match hook.hook_type.as_str() {
                "command" => {
                    if hook.command.is_none() {
                        return Err(PluginError::InvalidManifest(
                            "Command hook requires 'command' field".to_string(),
                        ));
                    }
                }
                "prompt" => {
                    if hook.prompt.is_none() {
                        return Err(PluginError::InvalidManifest(
                            "Prompt hook requires 'prompt' field".to_string(),
                        ));
                    }
                }
                other => {
                    return Err(PluginError::InvalidManifest(format!(
                        "Unknown hook type: {}",
                        other
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get the plugin name
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Get the plugin root path (for PLUGIN_ROOT env var)
    pub fn root(&self) -> &Path {
        &self.path
    }

    /// Get environment variables to set when running plugin scripts
    pub fn env_vars(&self) -> HashMap<String, String> {
        let mut vars = HashMap::new();
        vars.insert(
            "PLUGIN_ROOT".to_string(),
            self.path.to_string_lossy().to_string(),
        );
        vars.insert("PLUGIN_NAME".to_string(), self.manifest.name.clone());
        vars.insert("PLUGIN_VERSION".to_string(), self.manifest.version.clone());
        vars
    }

    /// Resolve a path relative to the plugin root
    pub fn resolve_path(&self, relative: &str) -> PathBuf {
        self.path.join(relative)
    }
}

/// Errors that can occur during plugin operations
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("Manifest not found: {0}")]
    ManifestNotFound(PathBuf),

    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Plugin not found: {0}")]
    NotFound(String),
}

/// Manages plugin discovery and loading
pub struct PluginManager {
    /// Loaded plugins by name
    plugins: HashMap<String, Plugin>,
    /// Search paths for plugins
    search_paths: Vec<PathBuf>,
}

impl PluginManager {
    /// Create a new plugin manager with default search paths
    pub fn new() -> Self {
        let mut search_paths = Vec::new();

        // User plugins directory
        if let Some(home) = dirs::home_dir() {
            search_paths.push(home.join(".openclaudia").join("plugins"));
        }

        // Project plugins directory
        search_paths.push(PathBuf::from(".openclaudia/plugins"));

        Self {
            plugins: HashMap::new(),
            search_paths,
        }
    }

    /// Create a plugin manager with custom search paths
    pub fn with_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            plugins: HashMap::new(),
            search_paths: paths,
        }
    }

    /// Discover and load all plugins from search paths
    pub fn discover(&mut self) -> Vec<PluginError> {
        let mut errors = Vec::new();

        for search_path in &self.search_paths.clone() {
            if !search_path.exists() {
                debug!(path = ?search_path, "Plugin search path does not exist");
                continue;
            }

            let entries = match fs::read_dir(search_path) {
                Ok(entries) => entries,
                Err(e) => {
                    warn!(path = ?search_path, error = %e, "Failed to read plugin directory");
                    continue;
                }
            };

            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    match Plugin::load(&path) {
                        Ok(plugin) => {
                            info!(
                                name = %plugin.name(),
                                version = %plugin.manifest.version,
                                path = ?path,
                                "Loaded plugin"
                            );
                            self.plugins.insert(plugin.name().to_string(), plugin);
                        }
                        Err(e) => {
                            warn!(path = ?path, error = %e, "Failed to load plugin");
                            errors.push(e);
                        }
                    }
                }
            }
        }

        errors
    }

    /// Get a plugin by name
    pub fn get(&self, name: &str) -> Option<&Plugin> {
        self.plugins.get(name)
    }

    /// Get all loaded plugins
    pub fn all(&self) -> impl Iterator<Item = &Plugin> {
        self.plugins.values()
    }

    /// Get the number of loaded plugins
    pub fn count(&self) -> usize {
        self.plugins.len()
    }

    /// Get all hooks from all plugins
    pub fn all_hooks(&self) -> Vec<(&Plugin, &PluginHook)> {
        self.plugins
            .values()
            .filter(|p| p.enabled)
            .flat_map(|plugin| plugin.manifest.hooks.iter().map(move |hook| (plugin, hook)))
            .collect()
    }

    /// Get hooks for a specific event
    pub fn hooks_for_event(&self, event: &str) -> Vec<(&Plugin, &PluginHook)> {
        self.all_hooks()
            .into_iter()
            .filter(|(_, hook)| hook.event == event)
            .collect()
    }

    /// Get all commands from all plugins
    pub fn all_commands(&self) -> Vec<(&Plugin, &PluginCommand)> {
        self.plugins
            .values()
            .filter(|p| p.enabled)
            .flat_map(|plugin| {
                plugin
                    .manifest
                    .commands
                    .iter()
                    .map(move |cmd| (plugin, cmd))
            })
            .collect()
    }

    /// Get all MCP servers from all plugins
    pub fn all_mcp_servers(&self) -> Vec<(&Plugin, &PluginMcpServer)> {
        self.plugins
            .values()
            .filter(|p| p.enabled)
            .flat_map(|plugin| {
                plugin
                    .manifest
                    .mcp_servers
                    .iter()
                    .map(move |server| (plugin, server))
            })
            .collect()
    }

    /// Enable a plugin
    pub fn enable(&mut self, name: &str) -> Result<(), PluginError> {
        if let Some(plugin) = self.plugins.get_mut(name) {
            plugin.enabled = true;
            Ok(())
        } else {
            Err(PluginError::NotFound(name.to_string()))
        }
    }

    /// Disable a plugin
    pub fn disable(&mut self, name: &str) -> Result<(), PluginError> {
        if let Some(plugin) = self.plugins.get_mut(name) {
            plugin.enabled = false;
            Ok(())
        } else {
            Err(PluginError::NotFound(name.to_string()))
        }
    }

    /// Reload all plugins
    pub fn reload(&mut self) -> Vec<PluginError> {
        self.plugins.clear();
        self.discover()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_plugin(dir: &Path, name: &str) {
        let plugin_dir = dir.join(name);
        fs::create_dir_all(&plugin_dir).unwrap();

        let manifest = serde_json::json!({
            "name": name,
            "version": "1.0.0",
            "description": "Test plugin",
            "hooks": [
                {
                    "event": "session_start",
                    "type": "command",
                    "command": "echo hello"
                }
            ],
            "commands": [
                {
                    "name": "test",
                    "description": "Test command",
                    "script": "echo test"
                }
            ]
        });

        fs::write(
            plugin_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn test_plugin_manifest_parsing() {
        let manifest_json = r#"{
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "hooks": [
                {
                    "event": "pre_tool_use",
                    "matcher": "Write|Edit",
                    "type": "command",
                    "command": "python validate.py"
                }
            ]
        }"#;

        let manifest: PluginManifest = serde_json::from_str(manifest_json).unwrap();
        assert_eq!(manifest.name, "test-plugin");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.hooks.len(), 1);
    }

    #[test]
    fn test_plugin_load() {
        let dir = TempDir::new().unwrap();
        create_test_plugin(dir.path(), "my-plugin");

        let plugin = Plugin::load(&dir.path().join("my-plugin")).unwrap();
        assert_eq!(plugin.name(), "my-plugin");
        assert_eq!(plugin.manifest.version, "1.0.0");
        assert!(plugin.enabled);
    }

    #[test]
    fn test_plugin_env_vars() {
        let dir = TempDir::new().unwrap();
        create_test_plugin(dir.path(), "env-test");

        let plugin = Plugin::load(&dir.path().join("env-test")).unwrap();
        let vars = plugin.env_vars();

        assert!(vars.contains_key("PLUGIN_ROOT"));
        assert_eq!(vars.get("PLUGIN_NAME"), Some(&"env-test".to_string()));
        assert_eq!(vars.get("PLUGIN_VERSION"), Some(&"1.0.0".to_string()));
    }

    #[test]
    fn test_plugin_manager_discover() {
        let dir = TempDir::new().unwrap();
        let plugins_dir = dir.path().join("plugins");
        fs::create_dir_all(&plugins_dir).unwrap();

        create_test_plugin(&plugins_dir, "plugin-a");
        create_test_plugin(&plugins_dir, "plugin-b");

        let mut manager = PluginManager::with_paths(vec![plugins_dir]);
        let errors = manager.discover();

        assert!(errors.is_empty());
        assert_eq!(manager.count(), 2);
        assert!(manager.get("plugin-a").is_some());
        assert!(manager.get("plugin-b").is_some());
    }

    #[test]
    fn test_plugin_manager_hooks() {
        let dir = TempDir::new().unwrap();
        let plugins_dir = dir.path().join("plugins");
        fs::create_dir_all(&plugins_dir).unwrap();

        create_test_plugin(&plugins_dir, "hook-plugin");

        let mut manager = PluginManager::with_paths(vec![plugins_dir]);
        manager.discover();

        let hooks = manager.hooks_for_event("session_start");
        assert_eq!(hooks.len(), 1);
    }

    #[test]
    fn test_invalid_manifest() {
        let dir = TempDir::new().unwrap();
        let plugin_dir = dir.path().join("bad-plugin");
        fs::create_dir_all(&plugin_dir).unwrap();

        // Missing required fields
        fs::write(plugin_dir.join("manifest.json"), r#"{"name": ""}"#).unwrap();

        let result = Plugin::load(&plugin_dir);
        assert!(result.is_err());
    }
}
