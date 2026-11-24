//! Update checks against GitHub releases, plus `fennec update --experimental`
//! for devs pulling the latest commit from main.
//!
//! The engine ships from source: `Liso7150/fennec` publishes no releases, so
//! what moves this binary is the CLI's `fennec update`, which fast-forwards
//! the checkout and rebuilds. The release path stays correct for a tag that
//! does attach binaries, and verifies such a download against the release's
//! SHA-256 digest list.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::Result;

/// GitHub repository for releases
const GITHUB_REPO: &str = "Liso7150/fennec";
const EXPERIMENTAL_BRANCH: &str = "main";

/// Platform triple for binary selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Platform {
    pub os: String,
    pub arch: String,
    pub ext: String,
}

impl Platform {
    /// Detect the current platform
    pub fn current() -> Self {
        let os = if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        }
        .to_string();

        let arch = if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "unknown"
        }
        .to_string();

        let ext = if cfg!(target_os = "windows") {
            "exe"
        } else {
            "bin"
        }
        .to_string();

        Platform { os, arch, ext }
    }

    /// Asset name for this platform (e.g., "fennec-windows-x86_64.exe")
    pub fn asset_name(&self, base: &str) -> String {
        if self.ext == "exe" {
            format!("{}-{}-{}.{}", base, self.os, self.arch, self.ext)
        } else {
            format!("{}-{}-{}", base, self.os, self.arch)
        }
    }
}

/// Release information from GitHub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub version: String,
    pub tag: String,
    pub published_at: String,
    pub assets: Vec<Asset>,
    pub checksums_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub download_url: String,
    pub size: u64,
}

/// `Some(true)` when `candidate` is a strictly newer dotted version than
/// `current`, `Some(false)` when it is the same or older, and `None` when
/// either side is not three plain numbers (a `dev-<sha>` tag, say) so the
/// caller refuses to guess.
///
/// This exists because the published releases run *behind* a working tree: a
/// 1.0.0 build must not pull the 0.9.0 assets just because the tag differs.
pub fn is_newer(candidate: &str, current: &str) -> Option<bool> {
    fn parts(version: &str) -> Option<[u64; 3]> {
        let version = version.trim().trim_start_matches('v');
        let mut numbers = [0u64; 3];
        for (field, slot) in version.split('.').zip(numbers.iter_mut()) {
            *slot = field.parse().ok()?;
        }
        Some(numbers)
    }
    let (candidate, current) = (parts(candidate)?, parts(current)?);
    Some(candidate > current)
}

/// Release asset that carries the SHA-256 digests of the other assets.
pub const CHECKSUMS_ASSET: &str = "SHA256SUMS";

/// The digest this release publishes for `asset_name`, `None` when it publishes
/// no digest list or not one for that asset. `download_asset` takes the digest,
/// not the URL of the list — handing it the URL compares the binary's hash
/// against a string and fails every download that is actually intact.
pub async fn checksum_for(release: &Release, asset_name: &str) -> Option<String> {
    let url = release.checksums_url.as_deref()?;
    let text = download_asset(url, None).await.ok()?;
    let text = String::from_utf8(text).ok()?;
    // Format: "<64 hex>  <filename>", the filename prefixed with `*` for a
    // binary-mode read.
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 && parts[1].trim_start_matches('*') == asset_name {
            return Some(parts[0].to_string());
        }
    }
    None
}

/// Check for updates from GitHub releases
pub async fn check_stable(current_version: &str) -> Result<Option<Release>> {
    let url = format!("https://api.github.com/repos/{}/releases/latest", GITHUB_REPO);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("fennec-updater")
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("GitHub API: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("GitHub API returned {}", resp.status()));
    }

    let release: GitHubRelease = resp
        .json()
        .await
        .map_err(|e| format!("Parse release: {e}"))?;

    // Only a release newer than the running binary is an update. The published
    // ladder runs behind a working tree, so an unequal tag is not enough.
    let version = release.tag_name.trim_start_matches('v').to_string();
    if is_newer(&version, current_version) != Some(true) {
        return Ok(None);
    }

    let assets: Vec<Asset> = release
        .assets
        .into_iter()
        .map(|a| Asset {
            name: a.name,
            download_url: a.browser_download_url,
            size: a.size,
        })
        .collect();
    let checksums_url = assets
        .iter()
        .find(|a| a.name == CHECKSUMS_ASSET)
        .map(|a| a.download_url.clone());

    Ok(Some(Release {
        version,
        tag: release.tag_name,
        published_at: release.published_at,
        assets,
        checksums_url,
    }))
}

/// Check for experimental updates from main branch
pub async fn check_experimental() -> Result<Option<Release>> {
    let url = format!(
        "https://api.github.com/repos/{}/commits/{EXPERIMENTAL_BRANCH}",
        GITHUB_REPO
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("fennec-updater")
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("GitHub API: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("GitHub API returned {}", resp.status()));
    }

    let commit: GitHubCommit = resp
        .json()
        .await
        .map_err(|e| format!("Parse commit: {e}"))?;

    // For experimental, we always "update" to the latest commit
    let short_sha = &commit.sha[..7];

    Ok(Some(Release {
        version: format!("dev-{}", short_sha),
        tag: commit.sha.clone(),
        published_at: commit.commit.author.date,
        assets: vec![],
        checksums_url: None,
    }))
}

/// Download an asset with checksum verification
pub async fn download_asset(url: &str, expected_checksum: Option<&str>) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .user_agent("fennec-updater")
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Download: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Download returned {}", resp.status()));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Read bytes: {e}"))?
        .to_vec();

    // Verify checksum if provided
    if let Some(expected) = expected_checksum {
        let actual = sha256_hex(&bytes);
        if actual != expected {
            return Err(format!(
                "Checksum mismatch: expected {}, got {}",
                expected, actual
            ));
        }
    }

    Ok(bytes)
}

/// Compute SHA-256 checksum of bytes
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    hex::encode(result)
}

// GitHub API response types
#[derive(Deserialize)]
struct GitHubRelease {
    tag_name: String,
    published_at: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Deserialize)]
struct GitHubCommit {
    sha: String,
    commit: GitHubCommitDetail,
}

#[derive(Deserialize)]
struct GitHubCommitDetail {
    author: GitHubAuthor,
}

#[derive(Deserialize)]
struct GitHubAuthor {
    date: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_higher_release_is_an_update_and_an_older_one_is_not() {
        assert_eq!(is_newer("1.1.0", "1.0.0"), Some(true));
        assert_eq!(is_newer("1.10.0", "1.9.0"), Some(true));
        assert_eq!(is_newer("v1.1.0", "1.0.0"), Some(true));
        assert_eq!(is_newer("1.0.0", "1.0.0"), Some(false));
        assert_eq!(is_newer("0.9.0", "1.0.0"), Some(false));
    }

    #[test]
    fn a_tag_the_comparator_does_not_understand_offers_nothing() {
        assert_eq!(is_newer("dev-abc1234", "1.0.0"), None);
        assert_eq!(is_newer("1.0.0", "1.0.0-rc1"), None);
    }
}
