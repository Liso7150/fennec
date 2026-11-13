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
}