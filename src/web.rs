//! Web tools for OpenClaudia
//!
//! Provides web access capabilities for agents:
//! - `web_fetch`: Fetch URL content via Jina Reader (free, handles JS/Cloudflare)
//! - `web_search`: Search the web via Tavily or Brave API (requires API key)
//! - `web_browser`: Full browser automation via headless Chrome (optional feature)

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Jina Reader base URL - converts any URL to clean markdown
const JINA_READER_URL: &str = "https://r.jina.ai/";

/// Tavily API endpoint
const TAVILY_API_URL: &str = "https://api.tavily.com/search";

/// Brave Search API endpoint
const BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";

/// Web configuration for API keys
#[derive(Debug, Clone, Default)]
pub struct WebConfig {
    pub tavily_api_key: Option<String>,
    pub brave_api_key: Option<String>,
}

impl WebConfig {
    /// Load web config from environment variables
    pub fn from_env() -> Self {
        Self {
            tavily_api_key: std::env::var("TAVILY_API_KEY").ok(),
            brave_api_key: std::env::var("BRAVE_API_KEY").ok(),
        }
    }

    /// Check if any search provider is configured
    pub fn has_search_provider(&self) -> bool {
        self.tavily_api_key.is_some() || self.brave_api_key.is_some()
    }
}

/// Result from web_fetch
#[derive(Debug, Clone)]
pub struct FetchResult {
    pub content: String,
    pub title: Option<String>,
    pub url: String,
}

/// Search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Fetch a URL using Jina Reader
///
/// Jina Reader handles:
/// - JavaScript rendering
/// - Cloudflare bypass
/// - Clean markdown output
pub async fn fetch_url(url: &str) -> Result<FetchResult, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    // Use Jina Reader to fetch and convert to markdown
    let jina_url = format!("{}{}", JINA_READER_URL, url);

    let response = client
        .get(&jina_url)
        .header("Accept", "text/markdown")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {} - {}", response.status(), url));
    }

    let content = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Extract title from markdown if present (first # heading)
    let title = content
        .lines()
        .find(|line| line.starts_with("# "))
        .map(|line| line.trim_start_matches("# ").to_string());

    Ok(FetchResult {
        content,
        title,
        url: url.to_string(),
    })
}

/// Tavily API response structure
#[derive(Debug, Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}

/// Brave Search API response structure
#[derive(Debug, Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Vec<BraveResult>,
}

#[derive(Debug, Deserialize)]
struct BraveResult {
    title: String,
    url: String,
    description: String,
}

/// Search the web using configured provider (Tavily or Brave)
pub async fn search_web(query: &str, config: &WebConfig, limit: usize) -> Result<Vec<SearchResult>, String> {
    // Try Tavily first, then Brave
    if let Some(api_key) = &config.tavily_api_key {
        return search_tavily(query, api_key, limit).await;
    }

    if let Some(api_key) = &config.brave_api_key {
        return search_brave(query, api_key, limit).await;
    }

    Err("No search API configured. Set TAVILY_API_KEY or BRAVE_API_KEY environment variable.".to_string())
}

/// Search using Tavily API
async fn search_tavily(query: &str, api_key: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    #[derive(Serialize)]
    struct TavilyRequest<'a> {
        api_key: &'a str,
        query: &'a str,
        max_results: usize,
        include_answer: bool,
    }

    let request = TavilyRequest {
        api_key,
        query,
        max_results: limit,
        include_answer: false,
    };

    let response = client
        .post(TAVILY_API_URL)
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Tavily API request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Tavily API error {}: {}", status, body));
    }

    let tavily_response: TavilyResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Tavily response: {}", e))?;

    Ok(tavily_response
        .results
        .into_iter()
        .map(|r| SearchResult {
            title: r.title,
            url: r.url,
            snippet: r.content,
        })
        .collect())
}

/// Search using Brave Search API
async fn search_brave(query: &str, api_key: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(BRAVE_SEARCH_URL)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .query(&[("q", query), ("count", &limit.to_string())])
        .send()
        .await
        .map_err(|e| format!("Brave Search API request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Brave Search API error {}: {}", status, body));
    }

    let brave_response: BraveResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Brave response: {}", e))?;

    Ok(brave_response
        .web
        .map(|w| {
            w.results
                .into_iter()
                .map(|r| SearchResult {
                    title: r.title,
                    url: r.url,
                    snippet: r.description,
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Fetch URL using headless Chrome browser
///
/// Use this when Jina Reader fails (e.g., complex authentication, specific Cloudflare challenges)
#[cfg(feature = "browser")]
pub fn fetch_with_browser(url: &str) -> Result<FetchResult, String> {
    use headless_chrome::{Browser, LaunchOptions};

    let browser = Browser::new(
        LaunchOptions::default_builder()
            .headless(true)
            .build()
            .map_err(|e| format!("Failed to configure browser: {}", e))?,
    )
    .map_err(|e| format!("Failed to launch browser: {}", e))?;

    let tab = browser
        .new_tab()
        .map_err(|e| format!("Failed to create browser tab: {}", e))?;

    tab.navigate_to(url)
        .map_err(|e| format!("Failed to navigate to URL: {}", e))?;

    tab.wait_until_navigated()
        .map_err(|e| format!("Navigation timeout: {}", e))?;

    // Wait a bit for JavaScript to render
    std::thread::sleep(Duration::from_secs(2));

    // Get page content
    let content = tab
        .get_content()
        .map_err(|e| format!("Failed to get page content: {}", e))?;

    // Get title
    let title = tab
        .get_title()
        .ok();

    Ok(FetchResult {
        content,
        title,
        url: url.to_string(),
    })
}

#[cfg(not(feature = "browser"))]
pub fn fetch_with_browser(_url: &str) -> Result<FetchResult, String> {
    Err("Browser feature not enabled. Rebuild with `cargo build --features browser`".to_string())
}

/// Format search results for display to the agent
pub fn format_search_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let mut output = format!("Found {} results:\n\n", results.len());

    for (i, result) in results.iter().enumerate() {
        output.push_str(&format!(
            "{}. **{}**\n   {}\n   URL: {}\n\n",
            i + 1,
            result.title,
            result.snippet,
            result.url
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_config_from_env() {
        // Just test that it doesn't panic
        let _config = WebConfig::from_env();
    }

    #[test]
    fn test_format_search_results() {
        let results = vec![
            SearchResult {
                title: "Test Result".to_string(),
                url: "https://example.com".to_string(),
                snippet: "This is a test result".to_string(),
            },
        ];

        let formatted = format_search_results(&results);
        assert!(formatted.contains("Test Result"));
        assert!(formatted.contains("https://example.com"));
    }

    #[test]
    fn test_format_empty_results() {
        let results: Vec<SearchResult> = vec![];
        let formatted = format_search_results(&results);
        assert!(formatted.contains("No results found"));
    }
}
