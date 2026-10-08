// PY_SOURCE: pce500/oz9600/persistence.py
//! Host file ownership and atomic battery-image persistence. No guest access.
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "state path needs a filename")
    })?;
    let mut temp_name = name.to_os_string();
    temp_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    atomic_replace_with(path, path.with_file_name(temp_name), write)
}

fn atomic_replace_with(
    path: &Path,
    temporary_path: PathBuf,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary_path)?;
    // Cleanup owns this file only after exclusive creation succeeds.
    let temporary = Temporary(temporary_path);
    write(&mut file)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary.0, path)?;
    // The rename and complete payload are durable together on supported Unix
    // filesystems. Windows rename replaces the file; directory fsync is absent.
    #[cfg(unix)]
    File::open(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?
    .sync_all()?;
    Ok(())
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    atomic_write_with(path, |file| file.write_all(bytes))
}

pub struct RetainedStore {
    path: PathBuf,
    // Advisory OS lock survives atomic replacement and releases on process
    // exit/crash. Keep the sidecar inode, rather than unlinking another owner.
    _lock: File,
    last_hash: Option<[u8; 32]>,
}
impl Drop for RetainedStore {
    fn drop(&mut self) {
        // A concurrent fork can briefly retain a duplicate open description.
        // Release our ownership before closing the last local descriptor.
        let _ = self._lock.unlock();
    }
}
impl RetainedStore {
    pub fn open(path: &Path) -> io::Result<(Self, Option<Vec<u8>>)> {
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "state path needs a filename")
        })?;
        let mut lock_name = name.to_os_string();
        lock_name.push(".lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.with_file_name(lock_name))?;
        lock.try_lock().map_err(|e| {
            io::Error::other(format!(
                "cannot acquire state-file lock for {}: {e}",
                path.display()
            ))
        })?;
        let saved = match fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        let last_hash = saved.as_ref().map(|bytes| Sha256::digest(bytes).into());
        Ok((
            Self {
                path: path.to_path_buf(),
                _lock: lock,
                last_hash,
            },
            saved,
        ))
    }
    /// Call only after the core validates a loaded image or exports current
    /// backing. Failed writes leave both the prior image and dedup hash intact.
    pub fn save(&mut self, image: &[u8]) -> io::Result<bool> {
        let hash: [u8; 32] = Sha256::digest(image).into();
        if self.last_hash == Some(hash) {
            return Ok(false);
        }
        atomic_write(&self.path, image)?;
        self.last_hash = Some(hash);
        Ok(true)
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn directory() -> Temporary {
        let p = std::env::temp_dir().join(format!(
            "oz-native-store-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Temporary(p)
    }
    #[test]
    fn occupied_temporary_is_not_removed_or_overwritten() {
        let dir = directory();
        let path = dir.0.join("saved.ozbat");
        let temporary = dir.0.join("occupied.tmp");
        fs::write(&path, b"previous").unwrap();
        fs::write(&temporary, b"another owner").unwrap();
        assert!(atomic_replace_with(&path, temporary.clone(), |f| f.write_all(b"new")).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous");
        assert_eq!(fs::read(&temporary).unwrap(), b"another owner");
        fs::remove_dir_all(&dir.0).unwrap();
    }
    #[test]
    fn interrupted_write_keeps_prior_image_and_removes_partial_temporary() {
        let dir = directory();
        let path = dir.0.join("saved.ozbat");
        fs::write(&path, b"previous valid image").unwrap();
        let result = atomic_write_with(&path, |f| {
            f.write_all(b"partial replacement")?;
            Err(io::Error::other("injected write failure"))
        });
        assert!(result.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous valid image");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        fs::remove_dir_all(&dir.0).unwrap();
    }
    #[test]
    fn replacement_is_complete_and_duplicate_images_are_not_rewritten() {
        let dir = directory();
        let path = dir.0.join("saved.ozbat");
        let (mut store, loaded) = RetainedStore::open(&path).unwrap();
        assert!(loaded.is_none());
        assert!(store.save(b"first").unwrap());
        assert!(store.save(b"second complete image").unwrap());
        assert!(!store.save(b"second complete image").unwrap());
        drop(store);
        let (store, loaded) = RetainedStore::open(&path).unwrap();
        assert_eq!(loaded.unwrap(), b"second complete image");
        drop(store);
        fs::remove_dir_all(&dir.0).unwrap();
    }
    #[test]
    fn simultaneous_writer_is_rejected_and_reopening_after_release_works() {
        let dir = directory();
        let path = dir.0.join("saved.ozbat");
        let (store, _) = RetainedStore::open(&path).unwrap();
        assert!(RetainedStore::open(&path).is_err());
        drop(store);
        assert!(RetainedStore::open(&path).is_ok());
        fs::remove_dir_all(&dir.0).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn releasing_store_unlocks_even_with_an_inherited_descriptor() {
        let dir = directory();
        let path = dir.0.join("saved.ozbat");
        let (store, _) = RetainedStore::open(&path).unwrap();
        // A concurrent fork may inherit this open description until exec.
        // Store ownership, rather than the last descriptor, must release it.
        let inherited = store._lock.try_clone().unwrap();
        assert!(RetainedStore::open(&path).is_err());
        drop(store);
        let (reopened, _) = RetainedStore::open(&path).unwrap();
        drop(reopened);
        drop(inherited);
        fs::remove_dir_all(&dir.0).unwrap();
    }
    #[test]
    fn failed_store_write_can_retry_the_same_new_image() {
        let dir = directory();
        let moved = dir.0.with_extension("moved");
        let path = dir.0.join("saved.ozbat");
        let (mut store, _) = RetainedStore::open(&path).unwrap();
        store.save(b"previous").unwrap();
        fs::rename(&dir.0, &moved).unwrap();
        assert!(store.save(b"new").is_err());
        assert_eq!(fs::read(moved.join("saved.ozbat")).unwrap(), b"previous");
        fs::rename(&moved, &dir.0).unwrap();
        assert!(store.save(b"new").unwrap());
        assert_eq!(fs::read(&path).unwrap(), b"new");
        drop(store);
        fs::remove_dir_all(&dir.0).unwrap();
    }
    #[test]
    fn unreadable_existing_target_is_never_treated_as_empty_memory() {
        let dir = directory();
        let path = dir.0.join("saved.ozbat");
        fs::create_dir(&path).unwrap();
        assert!(RetainedStore::open(&path).is_err());
        assert!(path.is_dir());
        fs::remove_dir_all(&dir.0).unwrap();
    }
}
