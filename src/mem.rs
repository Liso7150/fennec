//! What the process has actually committed, and what the OS is letting it keep.
//!
//! A tiered engine lives or dies by this pair. The host expert pool is anonymous
//! memory: if private commit exceeds physical RAM, Windows trims the working set
//! and moves the rest to the compressed store, and every "RAM-resident" expert
//! pick then costs a page fault instead of a load. The gap between the two
//! numbers is the whole difference between a memory tier and a disk tier, so it
//! is measured rather than assumed.

use std::sync::OnceLock;

/// `(private_commit_bytes, working_set_bytes)`, or `(0, 0)` where the platform
/// does not expose it.
pub fn usage() -> (u64, u64) {
    #[cfg(windows)]
    {
        windows::usage()
    }
    #[cfg(not(windows))]
    {
        (0, 0)
    }
}

pub fn private_gib() -> f64 {
    usage().0 as f64 / 2f64.powi(30)
}

/// System-wide available physical memory in bytes, or `0` where the platform
/// does not expose it. This is the RAM a new host pool can actually keep
/// resident: free pages plus the reclaimable cache, not the total installed.
/// The planner caps the host tier to this so a busy box streams cold experts
/// from disk on purpose instead of faulting a too-large "RAM" pool every token.
pub fn available_bytes() -> u64 {
    #[cfg(windows)]
    {
        windows::available()
    }
    #[cfg(target_os = "linux")]
    {
        linux::available()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        0
    }
}

pub fn resident_gib() -> f64 {
    usage().1 as f64 / 2f64.powi(30)
}

/// One line for the load report: committed against resident, so a trimmed
/// working set is visible at the phase that caused it.
pub fn report(tag: &str) {
    let (p, w) = usage();
    if p == 0 && w == 0 {
        return;
    }
    eprintln!(
        "[fennec] memory after {tag}: {:.2} GiB committed, {:.2} GiB resident",
        p as f64 / 2f64.powi(30),
        w as f64 / 2f64.powi(30)
    );
}

#[cfg(windows)]
mod windows {
    use super::OnceLock;
    use std::ffi::c_void;

    // PROCESS_MEMORY_COUNTERS, x64: two DWORDs then eight SIZE_T.
    const CB: u32 = 72;
    const OFF_WS: usize = 16;
    const OFF_PRIVATE: usize = 56;

    fn psapi() -> Option<&'static libloading::Library> {
        static LIB: OnceLock<Option<libloading::Library>> = OnceLock::new();
        LIB.get_or_init(|| unsafe { libloading::Library::new("psapi.dll") }.ok())
            .as_ref()
    }

    pub fn usage() -> (u64, u64) {
        let Some(lib) = psapi() else {
            return (0, 0);
        };
        // The current-process pseudo-handle, so no kernel32 lookup is needed.
        let handle = -1isize as *mut c_void;
        let mut buf = [0u8; CB as usize];
        let ok = unsafe {
            let f: libloading::Symbol<
                unsafe extern "system" fn(*mut c_void, *mut u8, u32) -> i32,
            > = match lib.get(b"GetProcessMemoryInfo") {
                Ok(f) => f,
                Err(_) => return (0, 0),
            };
            f(handle, buf.as_mut_ptr(), CB)
        };
        if ok == 0 {
            return (0, 0);
        }
        let rd = |off: usize| u64::from_ne_bytes(buf[off..off + 8].try_into().unwrap());
        (rd(OFF_PRIVATE), rd(OFF_WS))
    }

    // MEMORYSTATUSEX, x64: dwLength, dwMemoryLoad, then seven u64. ullAvailPhys
    // is the second u64, at byte offset 16. GlobalMemoryStatusEx lives in
    // kernel32, which is always loaded, so no library handle is kept.
    const MSX_LEN: u32 = 64;
    const OFF_AVAIL_PHYS: usize = 16;

    pub fn available() -> u64 {
        let Ok(lib) = (unsafe { libloading::Library::new("kernel32.dll") }) else {
            return 0;
        };
        let mut buf = [0u8; MSX_LEN as usize];
        buf[..4].copy_from_slice(&MSX_LEN.to_ne_bytes());
        let ok = unsafe {
            let f: libloading::Symbol<unsafe extern "system" fn(*mut u8) -> i32> =
                match lib.get(b"GlobalMemoryStatusEx") {
                    Ok(f) => f,
                    Err(_) => return 0,
                };
            f(buf.as_mut_ptr())
        };
        if ok == 0 {
            return 0;
        }
        u64::from_ne_bytes(buf[OFF_AVAIL_PHYS..OFF_AVAIL_PHYS + 8].try_into().unwrap())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    /// `MemAvailable` from /proc/meminfo (kB): the kernel's own estimate of what
    /// a new allocation can reclaim without swapping. Falls back to 0 (no cap)
    /// when the file or the field is absent.
    pub fn available() -> u64 {
        let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
            return 0;
        };
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest
                    .trim()
                    .trim_end_matches(" kB")
                    .trim()
                    .parse()
                    .unwrap_or(0);
                return kb * 1024;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_memory_is_reported_where_the_platform_exposes_it() {
        let a = available_bytes();
        if cfg!(any(windows, target_os = "linux")) {
            assert!(a > 0, "Windows and Linux expose available physical memory");
            // Available can never exceed the installed total; a sane upper bound
            // is 1 TiB, far above any box this runs on, catching a bad offset.
            assert!(a < (1u64 << 40), "available bytes {a} is implausibly large");
        }
    }

    #[test]
    fn process_usage_is_a_pair_of_byte_counts() {
        let (private, resident) = usage();
        if cfg!(windows) {
            assert!(private > 0 && resident > 0, "a live process commits memory");
        } else {
            assert_eq!((private, resident), (0, 0));
        }
    }
}