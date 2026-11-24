//! Update checks against GitHub releases, plus `fennec update --experimental`
//! for devs pulling the latest commit from master.
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
const EXPERIMENTAL_BRANCH: &str = "master";

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

/// The digest this release publishes for `asset_name`. A missing checksums
/// asset or an unreadable manifest is an error; `None` means the manifest has no
/// entry for that exact basename. Callers installing a binary must reject both.
pub async fn checksum_for(release: &Release, asset_name: &str) -> Result<Option<String>> {
    let url = release
        .checksums_url
        .as_deref()
        .ok_or_else(|| format!("Release is missing {CHECKSUMS_ASSET}"))?;
    let bytes = download_asset(url, None)
        .await
        .map_err(|e| format!("Download {CHECKSUMS_ASSET}: {e}"))?;
    let text = String::from_utf8(bytes).map_err(|e| format!("Parse {CHECKSUMS_ASSET}: {e}"))?;
    parse_checksum(&text, asset_name)
}

fn parse_checksum(text: &str, asset_name: &str) -> Result<Option<String>> {
    let mut found = None;
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let digest = fields
            .next()
            .ok_or_else(|| format!("Parse {CHECKSUMS_ASSET} line {}", index + 1))?;
        let file = fields
            .next()
            .ok_or_else(|| format!("Parse {CHECKSUMS_ASSET} line {}", index + 1))?;
        if fields.next().is_some() {
            return Err(format!(
                "Parse {CHECKSUMS_ASSET} line {}: expected two fields",
                index + 1
            ));
        }
        let file = file.strip_prefix('*').unwrap_or(file);
        let digest = normalize_checksum(digest)
            .map_err(|e| format!("Parse {CHECKSUMS_ASSET} line {}: {e}", index + 1))?;
        if file == asset_name {
            if found.is_some() {
                return Err(format!(
                    "Duplicate {CHECKSUMS_ASSET} entry for {asset_name}"
                ));
            }
            found = Some(digest);
        }
    }
    Ok(found)
}

fn normalize_checksum(checksum: &str) -> Result<String> {
    if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("Checksum must be exactly 64 hexadecimal characters".to_string());
    }
    Ok(checksum.to_ascii_lowercase())
}

/// Download a release binary only after its exact manifest entry and digest
/// have been verified. No caller needs to stage an unverified release asset.
pub async fn download_release_asset(release: &Release, asset_name: &str) -> Result<Vec<u8>> {
    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .ok_or_else(|| format!("No asset for platform: {asset_name}"))?;
    let checksum = checksum_for(release, asset_name)
        .await?
        .ok_or_else(|| format!("{CHECKSUMS_ASSET} has no entry for {asset_name}"))?;
    download_asset(&asset.download_url, Some(&checksum)).await
}

/// Check for updates from GitHub releases
pub async fn check_stable(current_version: &str) -> Result<Option<Release>> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
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

/// Check for experimental updates from master branch
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

/// Download an asset, requiring a valid digest whenever one is supplied.
/// `None` is only for metadata such as the checksums manifest itself.
pub async fn download_asset(url: &str, expected_checksum: Option<&str>) -> Result<Vec<u8>> {
    let expected_checksum = expected_checksum.map(normalize_checksum).transpose()?;
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

    #[test]
    fn experimental_updates_follow_master() {
        assert_eq!(EXPERIMENTAL_BRANCH, "master");
    }

    fn test_release(server_url: Option<String>) -> Release {
        let server_url = server_url.unwrap_or_else(|| "http://127.0.0.1".to_string());
        Release {
            version: "1.0.1".to_string(),
            tag: "v1.0.1".to_string(),
            published_at: "2026-11-24T00:00:00Z".to_string(),
            assets: vec![Asset {
                name: "fennec-windows-x86_64.exe".to_string(),
                download_url: format!("{server_url}/binary"),
                size: 5,
            }],
            checksums_url: Some(format!("{server_url}/SHA256SUMS")),
        }
    }

    fn serve(
        responses: Vec<(&'static str, &'static str, Vec<u8>)>,
    ) -> (
        String,
        std::thread::JoinHandle<()>,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let paths = Arc::new(Mutex::new(Vec::new()));
        let requested = paths.clone();
        let handle = std::thread::spawn(move || {
            for (expected_path, status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    request.push_str(&line);
                }
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or_default();
                requested.lock().unwrap().push(path.to_string());
                assert_eq!(path, expected_path);
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(&body).unwrap();
                stream.flush().unwrap();
            }
        });
        (format!("http://{address}"), handle, paths)
    }

    #[test]
    fn checksum_manifest_requires_an_exact_basename_and_normalizes_case() {
        let digest = "2CF24DBA5FB0A30E26E83B2AC5B9E29E1B161E5C1FA7425E73043362938B9824";
        let text = format!(
            "{}  other.exe\n{} *fennec-windows-x86_64.exe\n",
            "a".repeat(64),
            digest
        );
        assert_eq!(
            parse_checksum(&text, "fennec-windows-x86_64.exe").unwrap(),
            Some(digest.to_ascii_lowercase())
        );
        for file in [
            "./fennec-windows-x86_64.exe",
            "dir/fennec-windows-x86_64.exe",
            "fennec-windows-x86_64.exe.bak",
            "prefix-fennec-windows-x86_64.exe",
        ] {
            let text = format!("{digest}  {file}\n");
            assert_eq!(
                parse_checksum(&text, "fennec-windows-x86_64.exe").unwrap(),
                None
            );
        }
    }

    #[test]
    fn checksum_manifest_rejects_malformed_or_ambiguous_entries() {
        for digest in ["a".repeat(63), "a".repeat(65), "g".repeat(64)] {
            let text = format!("{digest}  fennec-windows-x86_64.exe\n");
            assert!(parse_checksum(&text, "fennec-windows-x86_64.exe").is_err());
        }
        for text in [
            "not-a-checksum\n",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  fennec-windows-x86_64.exe extra\n",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  other.exe extra\n",
        ] {
            assert!(parse_checksum(text, "fennec-windows-x86_64.exe").is_err());
        }
        let duplicate = format!(
            "{}  fennec-windows-x86_64.exe\n{} *fennec-windows-x86_64.exe\n",
            "a".repeat(64),
            "b".repeat(64)
        );
        assert!(
            parse_checksum(&duplicate, "fennec-windows-x86_64.exe")
                .unwrap_err()
                .contains("Duplicate")
        );
    }

    #[tokio::test]
    async fn binary_download_requires_the_checksums_asset() {
        let mut release = test_release(None);
        release.checksums_url = None;
        let error = download_release_asset(&release, "fennec-windows-x86_64.exe")
            .await
            .unwrap_err();
        assert!(error.contains("missing SHA256SUMS"), "{error}");
    }

    #[tokio::test]
    async fn binary_download_rejects_a_missing_exact_entry_before_fetching_binary() {
        let body = format!("{}  other.exe\n", "a".repeat(64)).into_bytes();
        let (url, server, paths) = serve(vec![("/SHA256SUMS", "200 OK", body)]);
        let error = download_release_asset(&test_release(Some(url)), "fennec-windows-x86_64.exe")
            .await
            .unwrap_err();
        server.join().unwrap();
        assert!(
            error.contains("no entry for fennec-windows-x86_64.exe"),
            "{error}"
        );
        assert_eq!(*paths.lock().unwrap(), ["/SHA256SUMS"]);
    }

    #[tokio::test]
    async fn binary_download_rejects_a_malformed_manifest_before_fetching_binary() {
        let body = b"short  fennec-windows-x86_64.exe\n".to_vec();
        let (url, server, paths) = serve(vec![("/SHA256SUMS", "200 OK", body)]);
        let error = download_release_asset(&test_release(Some(url)), "fennec-windows-x86_64.exe")
            .await
            .unwrap_err();
        server.join().unwrap();
        assert!(error.contains("64 hexadecimal"), "{error}");
        assert_eq!(*paths.lock().unwrap(), ["/SHA256SUMS"]);
    }

    #[tokio::test]
    async fn binary_download_rejects_manifest_fetch_and_parse_failures() {
        for (status, body) in [
            ("404 Not Found", b"missing".to_vec()),
            ("200 OK", vec![0xff, 0xfe]),
        ] {
            let (url, server, paths) = serve(vec![("/SHA256SUMS", status, body)]);
            let error =
                download_release_asset(&test_release(Some(url)), "fennec-windows-x86_64.exe")
                    .await
                    .unwrap_err();
            server.join().unwrap();
            assert!(error.contains("SHA256SUMS"), "{error}");
            assert_eq!(*paths.lock().unwrap(), ["/SHA256SUMS"]);
        }
    }

    #[tokio::test]
    async fn binary_download_rejects_a_mismatched_digest() {
        let binary = b"tampered";
        let manifest = format!("{}  fennec-windows-x86_64.exe\n", sha256_hex(b"expected"));
        let (url, server, paths) = serve(vec![
            ("/SHA256SUMS", "200 OK", manifest.into_bytes()),
            ("/binary", "200 OK", binary.to_vec()),
        ]);
        let error = download_release_asset(&test_release(Some(url)), "fennec-windows-x86_64.exe")
            .await
            .unwrap_err();
        server.join().unwrap();
        assert!(error.contains("Checksum mismatch"), "{error}");
        assert_eq!(*paths.lock().unwrap(), ["/SHA256SUMS", "/binary"]);
    }

    #[tokio::test]
    async fn binary_download_accepts_a_matching_uppercase_digest() {
        let binary = b"hello";
        let manifest = format!(
            "{} *fennec-windows-x86_64.exe\n",
            sha256_hex(binary).to_ascii_uppercase()
        );
        let (url, server, paths) = serve(vec![
            ("/SHA256SUMS", "200 OK", manifest.into_bytes()),
            ("/binary", "200 OK", binary.to_vec()),
        ]);
        let bytes = download_release_asset(&test_release(Some(url)), "fennec-windows-x86_64.exe")
            .await
            .unwrap();
        server.join().unwrap();
        assert_eq!(bytes, binary);
        assert_eq!(*paths.lock().unwrap(), ["/SHA256SUMS", "/binary"]);
    }

    #[tokio::test]
    async fn supplied_digest_must_be_well_formed() {
        let error = download_asset("http://127.0.0.1:0", Some("not-a-digest"))
            .await
            .unwrap_err();
        assert!(error.contains("64 hexadecimal"), "{error}");
    }
}
