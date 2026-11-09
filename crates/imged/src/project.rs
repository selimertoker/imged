//! Project saving (PSD) and periodic autosave to a temporary folder.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use imged_core::io::psd::{self, Snapshot};
use imged_core::Document;

/// Where this instance's autosaves go: a randomly named folder in the
/// temporary directory (e.g. `/tmp/imged-autosave-3f9c1a7e`), so several
/// running copies of imged never write to the same place.
pub fn autosave_dir() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        use std::hash::{BuildHasher, Hasher};
        let tmp = std::env::temp_dir();
        loop {
            // RandomState is seeded randomly per process.
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u32(std::process::id());
            h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let dir = tmp.join(format!("imged-autosave-{:08x}", h.finish() as u32));
            if !dir.exists() {
                return dir;
            }
        }
    })
    .clone()
}

/// How often a changed document is autosaved.
pub const AUTOSAVE_EVERY: Duration = Duration::from_secs(60);

/// Saves changed documents in the background every [`AUTOSAVE_EVERY`].
pub struct AutoSaver {
    last_token: Option<u64>,
    last_time: Instant,
    busy: bool,
    tx: Sender<Result<PathBuf, String>>,
    rx: Receiver<Result<PathBuf, String>>,
    pub last_path: Option<PathBuf>,
    pub last_error: Option<String>,
}

impl Default for AutoSaver {
    fn default() -> Self {
        let (tx, rx) = channel();
        AutoSaver { last_token: None, last_time: Instant::now(), busy: false, tx, rx, last_path: None, last_error: None }
    }
}

impl AutoSaver {
    /// The autosave file for a document: `<name>.psd` in [`autosave_dir`].
    fn file_for(name: &str) -> PathBuf {
        let stem: String = name
            .chars()
            .map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' })
            .collect();
        let stem = Path::new(&stem).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let stem = if stem.is_empty() { "untitled".to_string() } else { stem };
        autosave_dir().join(format!("{stem}.psd"))
    }

    /// Call every frame; starts a background save when it is time.
    /// `name` is the document's display name.
    pub fn tick(&mut self, doc: &Document, name: &str) {
        while let Ok(r) = self.rx.try_recv() {
            self.busy = false;
            match r {
                Ok(p) => {
                    self.last_path = Some(p);
                    self.last_error = None;
                }
                Err(e) => self.last_error = Some(e),
            }
        }
        if self.busy || self.last_time.elapsed() < AUTOSAVE_EVERY {
            return;
        }
        self.last_time = Instant::now();
        let token = doc.content_token();
        if self.last_token == Some(token) {
            return;
        }
        // Don't autosave a document nobody has touched.
        if self.last_token.is_none() && doc.undo_label().is_none() {
            self.last_token = Some(token);
            return;
        }
        self.last_token = Some(token);
        let snap = Snapshot::of(doc);
        let path = Self::file_for(name);
        let tx = self.tx.clone();
        self.busy = true;
        std::thread::spawn(move || {
            let r = std::fs::create_dir_all(autosave_dir())
                .map_err(|e| e.to_string())
                .and_then(|_| psd::save_snapshot(&snap, &path))
                .map(|_| path);
            let _ = tx.send(r);
        });
    }

    /// Forget the previous document (a new one was opened).
    pub fn reset(&mut self) {
        self.last_token = None;
        self.last_time = Instant::now();
    }
}
