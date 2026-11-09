//! Remembered settings: checkbox states, the palette and the ImageMagick
//! command, in `$XDG_DATA_HOME/imged/settings.json` (usually
//! `~/.local/share/imged/settings.json`).
//!
//! Everything is a plain key → value map so new options need no migration:
//! a missing key keeps the built-in default.

use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub bools: BTreeMap<String, bool>,
    pub strings: BTreeMap<String, String>,
    /// Palette colors as hex codes.
    pub palette: Option<Vec<String>>,
}

impl Prefs {
    /// Loads into `v` (when `load`) or stores `v`.
    pub fn bool(&mut self, load: bool, key: &str, v: &mut bool) {
        if load {
            if let Some(b) = self.bools.get(key) {
                *v = *b;
            }
        } else {
            self.bools.insert(key.to_string(), *v);
        }
    }

    pub fn string(&mut self, load: bool, key: &str, v: &mut String) {
        if load {
            if let Some(s) = self.strings.get(key) {
                v.clone_from(s);
            }
        } else {
            self.strings.insert(key.to_string(), v.clone());
        }
    }
}

/// The settings file.
pub fn path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("imged").join("settings.json"))
}

pub fn load() -> Prefs {
    path().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

pub fn save(p: &Prefs) -> Result<(), String> {
    let path = path().ok_or("No home directory")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(p).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.partial");
    std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}
