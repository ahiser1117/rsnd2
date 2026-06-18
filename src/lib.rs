use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::ffi::{CStr, CString, c_void};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

const MODERN_MAGIC: [u8; 4] = [0xda, 0xce, 0xbe, 0x0a];
const LEGACY_MAGIC: [u8; 4] = [0x6a, 0x50, 0x20, 0x20];
const CHUNK_MAP_SIGNATURE: &[u8] = b"ND2 CHUNK MAP SIGNATURE 0000001!";
const MAX_NAME_BYTES: u64 = 1 << 20;
const MAX_MAP_BYTES: u64 = 512 << 20;

#[derive(Debug)]
pub enum Nd2Error {
    Io(io::Error),
    Invalid(String),
    Unsupported(String),
}

impl fmt::Display for Nd2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nd2Error::Io(err) => write!(f, "{err}"),
            Nd2Error::Invalid(msg) => write!(f, "{msg}"),
            Nd2Error::Unsupported(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Nd2Error {}

impl From<io::Error> for Nd2Error {
    fn from(value: io::Error) -> Self {
        Nd2Error::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, Nd2Error>;

#[repr(C)]
pub struct Nd2Buffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub status: c_int,
    pub error: *mut c_char,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Nd2Variant {
    ModernChunked,
    LegacyJp2Like,
}

#[derive(Debug, Clone)]
pub struct ChunkHeader {
    pub offset: u64,
    pub name_len: u64,
    pub payload_len: u64,
    pub name: String,
    pub payload_offset: u64,
}

impl ChunkHeader {
    pub fn end_offset(&self) -> u64 {
        self.payload_offset.saturating_add(self.payload_len)
    }
}

#[derive(Debug, Clone)]
pub struct MapEntry {
    pub name: String,
    pub offset: u64,
    pub secondary_offset: u64,
}

#[derive(Debug, Clone)]
pub struct PlaneRecord {
    pub sequence: usize,
    pub chunk_offset: u64,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub prefix8: [u8; 8],
}

#[derive(Debug, Clone)]
pub struct ImageAttributes {
    pub bits_per_component_in_memory: u32,
    pub bits_per_component_significant: u32,
    pub component_count: u32,
    pub height_px: u32,
    pub sequence_count: u32,
    pub width_bytes: u32,
    pub width_px: u32,
    pub compression: Option<u32>,
    pub compression_level: Option<f64>,
    pub tile_height_px: Option<u32>,
    pub tile_width_px: Option<u32>,
    pub channel_count: u32,
}

#[derive(Debug, Clone)]
pub struct Nd2Index {
    pub path: PathBuf,
    pub file_size: u64,
    pub variant: Nd2Variant,
    pub signature_version: Option<String>,
    pub filemap_offset: Option<u64>,
    pub chunk_count: usize,
    pub chunk_name_counts: BTreeMap<String, usize>,
    pub planes: Vec<PlaneRecord>,
    pub attributes: Option<ImageAttributes>,
}

#[derive(Debug, Clone)]
pub struct Nd2Summary {
    pub path: PathBuf,
    pub file_size: u64,
    pub variant: Nd2Variant,
    pub signature_version: Option<String>,
    pub filemap_offset: Option<u64>,
    pub chunk_count: usize,
    pub chunk_name_counts: BTreeMap<String, usize>,
    pub plane_count: usize,
    pub first_plane_payload_len: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Nd2VersionProbe {
    pub path: PathBuf,
    pub file_size: u64,
    pub variant: Nd2Variant,
    pub signature_name: String,
    pub signature_version: Option<String>,
}

impl Nd2Index {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        if file_size < 16 {
            return Err(Nd2Error::Invalid("file is too small to be ND2".to_string()));
        }

        let mut magic = [0; 4];
        read_exact_at(&file, &mut magic, 0)?;
        if magic == LEGACY_MAGIC {
            return Err(Nd2Error::Unsupported(
                "legacy JP2-like ND2 files are recognized but not implemented".to_string(),
            ));
        }
        if magic != MODERN_MAGIC {
            return Err(Nd2Error::Invalid(format!(
                "unexpected ND2 magic bytes {:02x?}",
                magic
            )));
        }

        let signature = read_chunk_header(&file, file_size, 0)?;
        let signature_version = read_signature_version(&file, &signature).ok();
        let filemap_offset = find_filemap_offset(&file, file_size)?;
        let map_header = read_chunk_header(&file, file_size, filemap_offset)?;
        if !map_header.name.starts_with("ND2 FILEMAP SIGNATURE NAME") {
            return Err(Nd2Error::Invalid(format!(
                "footer points to non-filemap chunk {:?} at {filemap_offset}",
                map_header.name
            )));
        }
        if map_header.payload_len > MAX_MAP_BYTES {
            return Err(Nd2Error::Invalid(format!(
                "file map payload is unreasonably large: {} bytes",
                map_header.payload_len
            )));
        }

        let mut map_payload = vec![0; map_header.payload_len as usize];
        read_exact_at(&file, &mut map_payload, map_header.payload_offset)?;
        let entries = parse_filemap_payload(&map_payload)?;

        let mut chunk_name_counts = BTreeMap::new();
        let mut plane_entries = Vec::new();
        let mut attributes_offset = None;
        for entry in &entries {
            let base = chunk_base_name(&entry.name).to_string();
            *chunk_name_counts.entry(base).or_insert(0) += 1;
            if let Some(sequence) = image_sequence(&entry.name) {
                plane_entries.push((sequence, entry.offset));
            } else if entry.name == "ImageAttributesLV!" || entry.name == "ImageAttributes!" {
                attributes_offset = Some(entry.offset);
            }
        }

        let attributes = attributes_offset
            .map(|offset| read_image_attributes(&file, file_size, offset))
            .transpose()?;

        plane_entries.sort_unstable_by_key(|(sequence, _)| *sequence);
        let mut planes = Vec::with_capacity(plane_entries.len());
        if let Some(pixel_bytes_per_plane) = attributes
            .as_ref()
            .and_then(uncompressed_pixel_bytes_per_plane)
        {
            for (sequence, chunk_offset) in plane_entries {
                let payload_offset = chunk_offset.checked_add(4088).ok_or_else(|| {
                    Nd2Error::Invalid("image payload offset overflow".to_string())
                })?;
                let payload_len = pixel_bytes_per_plane.checked_add(8).ok_or_else(|| {
                    Nd2Error::Invalid("image payload length overflow".to_string())
                })?;
                let end = payload_offset.checked_add(payload_len).ok_or_else(|| {
                    Nd2Error::Invalid("image payload end offset overflow".to_string())
                })?;
                if end > file_size {
                    return Err(Nd2Error::Invalid(format!(
                        "image payload for sequence {sequence} extends past file end"
                    )));
                }
                planes.push(PlaneRecord {
                    sequence,
                    chunk_offset,
                    payload_offset,
                    payload_len,
                    prefix8: [0; 8],
                });
            }
        } else {
            for (sequence, chunk_offset) in plane_entries {
                let header = read_chunk_header(&file, file_size, chunk_offset)?;
                let mut prefix8 = [0; 8];
                if header.payload_len >= 8 {
                    read_exact_at(&file, &mut prefix8, header.payload_offset)?;
                }
                planes.push(PlaneRecord {
                    sequence,
                    chunk_offset,
                    payload_offset: header.payload_offset,
                    payload_len: header.payload_len,
                    prefix8,
                });
            }
        }

        Ok(Nd2Index {
            path: path.to_path_buf(),
            file_size,
            variant: Nd2Variant::ModernChunked,
            signature_version,
            filemap_offset: Some(filemap_offset),
            chunk_count: entries.len(),
            chunk_name_counts,
            planes,
            attributes,
        })
    }

    pub fn read_plane_payload(&self, sequence: usize) -> Result<Vec<u8>> {
        let plane = self
            .planes
            .iter()
            .find(|plane| plane.sequence == sequence)
            .ok_or_else(|| {
                Nd2Error::Invalid(format!("plane sequence {sequence} is not indexed"))
            })?;
        let file = File::open(&self.path)?;
        let mut payload = vec![0; plane.payload_len as usize];
        read_exact_at(&file, &mut payload, plane.payload_offset)?;
        Ok(payload)
    }

    pub fn pixel_bytes_len_after_prefix(&self, prefix_len: u64) -> Result<u64> {
        self.planes.iter().try_fold(0u64, |total, plane| {
            let pixel_len = plane.payload_len.checked_sub(prefix_len).ok_or_else(|| {
                Nd2Error::Invalid(format!(
                    "plane {} payload is shorter than {prefix_len} bytes",
                    plane.sequence
                ))
            })?;
            total
                .checked_add(pixel_len)
                .ok_or_else(|| Nd2Error::Invalid("pixel byte length overflow".to_string()))
        })
    }

    pub fn read_pixel_bytes_after_prefix(&self, prefix_len: u64) -> Result<Vec<u8>> {
        let total_len = self.pixel_bytes_len_after_prefix(prefix_len)?;
        let mut out = vec![0; total_len as usize];
        let file = File::open(&self.path)?;
        let mut dst = 0usize;
        for plane in &self.planes {
            let pixel_len = plane.payload_len.checked_sub(prefix_len).ok_or_else(|| {
                Nd2Error::Invalid(format!(
                    "plane {} payload is shorter than {prefix_len} bytes",
                    plane.sequence
                ))
            })?;
            let end = dst + pixel_len as usize;
            read_exact_at(&file, &mut out[dst..end], plane.payload_offset + prefix_len)?;
            dst = end;
        }
        Ok(out)
    }

    pub fn write_pixel_bytes_after_prefix<W: Write>(
        &self,
        prefix_len: u64,
        writer: &mut W,
    ) -> Result<u64> {
        let file = File::open(&self.path)?;
        let mut total = 0u64;
        for plane in &self.planes {
            let pixel_len = plane.payload_len.checked_sub(prefix_len).ok_or_else(|| {
                Nd2Error::Invalid(format!(
                    "plane {} payload is shorter than {prefix_len} bytes",
                    plane.sequence
                ))
            })?;
            let mut buf = vec![0; pixel_len as usize];
            read_exact_at(&file, &mut buf, plane.payload_offset + prefix_len)?;
            writer.write_all(&buf)?;
            total += pixel_len;
        }
        Ok(total)
    }
}

/// A persistent reader over a single ND2 file: the parsed plane index plus an
/// open file handle, so repeated batched reads avoid re-parsing the chunk map
/// and re-opening the file on every call. Positional reads (`pread`) are used
/// throughout, so a single handle is safe to share across threads.
pub struct Nd2Reader {
    index: Nd2Index,
    file: File,
}

impl Nd2Reader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let index = Nd2Index::open(path)?;
        let file = File::open(path)?;
        Ok(Self { index, file })
    }

    pub fn index(&self) -> &Nd2Index {
        &self.index
    }

    fn plane_for(&self, sequence: usize) -> Result<&PlaneRecord> {
        let planes = &self.index.planes;
        if sequence < planes.len() && planes[sequence].sequence == sequence {
            return Ok(&planes[sequence]);
        }
        planes
            .iter()
            .find(|plane| plane.sequence == sequence)
            .ok_or_else(|| Nd2Error::Invalid(format!("frame index {sequence} is not indexed")))
    }

    /// Pack the pixel bytes (each plane's payload minus the leading `prefix_len`
    /// stamp) for the requested plane sequences contiguously into `out`, in the
    /// order given by `indices`. Reads are issued positionally so up to
    /// `n_threads` planes can be fetched concurrently — important for breaking
    /// the single-stream throughput ceiling on networked (NFS) storage.
    pub fn read_frames_into(
        &self,
        indices: &[u64],
        prefix_len: u64,
        out: &mut [u8],
        n_threads: usize,
    ) -> Result<()> {
        // Resolve each requested sequence to a (source offset, length) job and
        // validate that the destination buffer is exactly the right size.
        let mut jobs: Vec<(u64, usize)> = Vec::with_capacity(indices.len());
        let mut total = 0usize;
        for &seq in indices {
            let plane = self.plane_for(seq as usize)?;
            let pixel_len = plane.payload_len.checked_sub(prefix_len).ok_or_else(|| {
                Nd2Error::Invalid(format!(
                    "plane {} payload is shorter than {prefix_len} bytes",
                    plane.sequence
                ))
            })? as usize;
            jobs.push((plane.payload_offset + prefix_len, pixel_len));
            total = total
                .checked_add(pixel_len)
                .ok_or_else(|| Nd2Error::Invalid("batch pixel length overflow".to_string()))?;
        }
        if total != out.len() {
            return Err(Nd2Error::Invalid(format!(
                "output buffer is {} bytes but {} frames need {total} bytes",
                out.len(),
                indices.len()
            )));
        }

        let threads = n_threads.clamp(1, jobs.len().max(1));
        if threads <= 1 {
            let mut dst = 0usize;
            for (src, len) in &jobs {
                read_exact_at(&self.file, &mut out[dst..dst + len], *src)?;
                dst += len;
            }
            return Ok(());
        }

        // Partition the jobs into `threads` contiguous groups; because jobs are
        // emitted in destination order, each group owns one contiguous slice of
        // `out`, carved out with split_at_mut so the borrows stay disjoint.
        let per = jobs.len().div_ceil(threads);
        let file = &self.file;
        let mut group_slices: Vec<(&mut [u8], &[(u64, usize)])> = Vec::new();
        let mut remaining: &mut [u8] = out;
        let mut start = 0usize;
        while start < jobs.len() {
            let end = (start + per).min(jobs.len());
            let group = &jobs[start..end];
            let bytes: usize = group.iter().map(|(_, len)| *len).sum();
            let (head, tail) = remaining.split_at_mut(bytes);
            group_slices.push((head, group));
            remaining = tail;
            start = end;
        }

        let mut first_err: Option<Nd2Error> = None;
        std::thread::scope(|scope| {
            let handles: Vec<_> = group_slices
                .into_iter()
                .map(|(slice, group)| {
                    scope.spawn(move || -> Result<()> {
                        let mut dst = 0usize;
                        for (src, len) in group {
                            read_exact_at(file, &mut slice[dst..dst + len], *src)?;
                            dst += len;
                        }
                        Ok(())
                    })
                })
                .collect();
            for handle in handles {
                match handle.join() {
                    Ok(Err(err)) if first_err.is_none() => first_err = Some(err),
                    Err(_) if first_err.is_none() => {
                        first_err = Some(Nd2Error::Invalid("reader thread panicked".to_string()))
                    }
                    _ => {}
                }
            }
        });
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl Nd2Summary {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let (signature_version, filemap_offset, entries) = read_modern_filemap(&file, file_size)?;

        let mut chunk_name_counts = BTreeMap::new();
        let mut first_plane: Option<(usize, u64)> = None;
        let mut plane_count = 0usize;

        for entry in &entries {
            let base = chunk_base_name(&entry.name).to_string();
            *chunk_name_counts.entry(base).or_insert(0) += 1;
            if let Some(sequence) = image_sequence(&entry.name) {
                plane_count += 1;
                if first_plane
                    .as_ref()
                    .map_or(true, |(first_sequence, _)| sequence < *first_sequence)
                {
                    first_plane = Some((sequence, entry.offset));
                }
            }
        }

        let first_plane_payload_len = first_plane
            .map(|(_, offset)| read_chunk_header(&file, file_size, offset).map(|h| h.payload_len))
            .transpose()?;

        Ok(Nd2Summary {
            path: path.to_path_buf(),
            file_size,
            variant: Nd2Variant::ModernChunked,
            signature_version,
            filemap_offset: Some(filemap_offset),
            chunk_count: entries.len(),
            chunk_name_counts,
            plane_count,
            first_plane_payload_len,
        })
    }
}

impl Nd2VersionProbe {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        if file_size < 16 {
            return Err(Nd2Error::Invalid("file is too small to be ND2".to_string()));
        }

        let mut magic = [0; 4];
        read_exact_at(&file, &mut magic, 0)?;
        if magic == LEGACY_MAGIC {
            return Ok(Nd2VersionProbe {
                path: path.to_path_buf(),
                file_size,
                variant: Nd2Variant::LegacyJp2Like,
                signature_name: String::new(),
                signature_version: None,
            });
        }
        if magic != MODERN_MAGIC {
            return Err(Nd2Error::Invalid(format!(
                "unexpected ND2 magic bytes {:02x?}",
                magic
            )));
        }

        let signature = read_chunk_header(&file, file_size, 0)?;
        if signature.name != "ND2 FILE SIGNATURE CHUNK NAME01!" {
            return Err(Nd2Error::Invalid(format!(
                "unexpected signature chunk name {:?}",
                signature.name
            )));
        }
        let signature_version = read_signature_version(&file, &signature).ok();
        if !signature_version
            .as_deref()
            .is_some_and(|version| version.starts_with("Ver"))
        {
            return Err(Nd2Error::Invalid(format!(
                "unexpected signature version {:?}",
                signature_version
            )));
        }
        Ok(Nd2VersionProbe {
            path: path.to_path_buf(),
            file_size,
            variant: Nd2Variant::ModernChunked,
            signature_name: signature.name,
            signature_version,
        })
    }
}

pub fn discover_nd2_files(root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    discover_nd2_files_inner(root.as_ref(), &mut files)?;
    files.sort();
    Ok(files)
}

fn discover_nd2_files_inner(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        if path.extension() == Some(OsStr::new("nd2")) {
            files.push(path.to_path_buf());
        }
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            discover_nd2_files_inner(&entry.path(), files)?;
        } else if file_type.is_file() && entry.path().extension() == Some(OsStr::new("nd2")) {
            files.push(entry.path());
        }
    }
    Ok(())
}

pub fn read_chunk_header(file: &File, file_size: u64, offset: u64) -> Result<ChunkHeader> {
    let mut header = [0; 16];
    read_exact_at(file, &mut header, offset)?;
    if header[0..4] != MODERN_MAGIC {
        return Err(Nd2Error::Invalid(format!(
            "missing modern chunk magic at offset {offset}"
        )));
    }

    let name_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;
    let payload_len = u64::from_le_bytes(header[8..16].try_into().unwrap());
    if name_len == 0 || name_len > MAX_NAME_BYTES {
        return Err(Nd2Error::Invalid(format!(
            "invalid chunk name area length {name_len} at offset {offset}"
        )));
    }
    let payload_offset = offset
        .checked_add(16)
        .and_then(|v| v.checked_add(name_len))
        .ok_or_else(|| Nd2Error::Invalid("chunk offset overflow".to_string()))?;
    let end = payload_offset
        .checked_add(payload_len)
        .ok_or_else(|| Nd2Error::Invalid("chunk end offset overflow".to_string()))?;
    if end > file_size {
        return Err(Nd2Error::Invalid(format!(
            "chunk at offset {offset} extends past file end"
        )));
    }

    let mut name_bytes = vec![0; name_len as usize];
    read_exact_at(file, &mut name_bytes, offset + 16)?;
    let nul = name_bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(name_bytes.len());
    let name = String::from_utf8_lossy(&name_bytes[..nul]).to_string();

    Ok(ChunkHeader {
        offset,
        name_len,
        payload_len,
        name,
        payload_offset,
    })
}

fn read_signature_version(file: &File, signature: &ChunkHeader) -> Result<String> {
    let len = signature.payload_len.min(128) as usize;
    let mut payload = vec![0; len];
    read_exact_at(file, &mut payload, signature.payload_offset)?;
    let nul = payload
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(payload.len());
    Ok(String::from_utf8_lossy(&payload[..nul]).to_string())
}

fn read_modern_filemap(
    file: &File,
    file_size: u64,
) -> Result<(Option<String>, u64, Vec<MapEntry>)> {
    if file_size < 16 {
        return Err(Nd2Error::Invalid("file is too small to be ND2".to_string()));
    }

    let mut magic = [0; 4];
    read_exact_at(file, &mut magic, 0)?;
    if magic == LEGACY_MAGIC {
        return Err(Nd2Error::Unsupported(
            "legacy JP2-like ND2 files are recognized but not implemented".to_string(),
        ));
    }
    if magic != MODERN_MAGIC {
        return Err(Nd2Error::Invalid(format!(
            "unexpected ND2 magic bytes {:02x?}",
            magic
        )));
    }

    let signature = read_chunk_header(file, file_size, 0)?;
    let signature_version = read_signature_version(file, &signature).ok();
    let filemap_offset = find_filemap_offset(file, file_size)?;
    let map_header = read_chunk_header(file, file_size, filemap_offset)?;
    if !map_header.name.starts_with("ND2 FILEMAP SIGNATURE NAME") {
        return Err(Nd2Error::Invalid(format!(
            "footer points to non-filemap chunk {:?} at {filemap_offset}",
            map_header.name
        )));
    }
    if map_header.payload_len > MAX_MAP_BYTES {
        return Err(Nd2Error::Invalid(format!(
            "file map payload is unreasonably large: {} bytes",
            map_header.payload_len
        )));
    }

    let mut map_payload = vec![0; map_header.payload_len as usize];
    read_exact_at(file, &mut map_payload, map_header.payload_offset)?;
    let entries = parse_filemap_payload(&map_payload)?;
    Ok((signature_version, filemap_offset, entries))
}

fn find_filemap_offset(file: &File, file_size: u64) -> Result<u64> {
    if file_size < 40 {
        return Err(Nd2Error::Invalid(
            "file is too small for ND2 footer".to_string(),
        ));
    }
    let mut tail = [0; 40];
    read_exact_at(file, &mut tail, file_size - 40)?;
    if &tail[..CHUNK_MAP_SIGNATURE.len()] != CHUNK_MAP_SIGNATURE {
        return Err(Nd2Error::Invalid(format!(
            "Invalid ChunkMap signature {:?}",
            &tail[..CHUNK_MAP_SIGNATURE.len()]
        )));
    }
    let offset = u64::from_le_bytes(tail[CHUNK_MAP_SIGNATURE.len()..].try_into().unwrap());
    if offset >= file_size {
        return Err(Nd2Error::Invalid(format!(
            "filemap offset {offset} is outside file"
        )));
    }
    Ok(offset)
}

fn parse_filemap_payload(payload: &[u8]) -> Result<Vec<MapEntry>> {
    let mut entries = Vec::new();
    let mut pos = 0;
    while pos < payload.len() {
        while pos < payload.len() && payload[pos] == 0 {
            pos += 1;
        }
        if pos >= payload.len() {
            break;
        }

        let start = pos;
        while pos < payload.len() && payload[pos] != b'!' {
            pos += 1;
        }
        if pos >= payload.len() {
            return Err(Nd2Error::Invalid("unterminated filemap name".to_string()));
        }
        let name = String::from_utf8_lossy(&payload[start..=pos]).to_string();
        pos += 1;

        if name.starts_with("ND2 CHUNK MAP SIGNATURE") {
            break;
        }

        if pos + 16 > payload.len() {
            return Err(Nd2Error::Invalid(format!(
                "filemap entry {name:?} is missing offsets"
            )));
        }
        let offset = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
        let secondary_offset = u64::from_le_bytes(payload[pos + 8..pos + 16].try_into().unwrap());
        pos += 16;
        entries.push(MapEntry {
            name,
            offset,
            secondary_offset,
        });
    }
    Ok(entries)
}

fn chunk_base_name(name: &str) -> &str {
    name.split_once('|').map_or(name, |(base, _)| base)
}

fn image_sequence(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("ImageDataSeq|")?;
    let number = rest.strip_suffix('!').unwrap_or(rest);
    number.parse().ok()
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum LiteValue {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Level(BTreeMap<String, LiteValue>),
    List(Vec<LiteValue>),
    Bytes(Vec<u8>),
    String(String),
    None,
}

struct LiteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> LiteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Nd2Error::Invalid("truncated CLX lite metadata".to_string()));
        }
        let value = self.data[self.pos];
        self.pos += 1;
        Ok(value)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Nd2Error::Invalid("CLX lite offset overflow".to_string()))?;
        if end > self.data.len() {
            return Err(Nd2Error::Invalid("truncated CLX lite metadata".to_string()));
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_utf16_name(&mut self, code_units_with_nul: usize) -> Result<String> {
        let bytes = self.read_exact(code_units_with_nul * 2)?;
        let units = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .take_while(|unit| *unit != 0)
            .collect::<Vec<_>>();
        String::from_utf16(&units)
            .map_err(|err| Nd2Error::Invalid(format!("invalid UTF-16 metadata name: {err}")))
    }

    fn skip(&mut self, len: usize) -> Result<()> {
        self.read_exact(len).map(|_| ())
    }
}

fn parse_lite_records(data: &[u8], count: usize) -> Result<BTreeMap<String, LiteValue>> {
    let mut reader = LiteReader::new(data);
    parse_lite_records_from(&mut reader, count)
}

fn parse_lite_records_from(
    reader: &mut LiteReader<'_>,
    count: usize,
) -> Result<BTreeMap<String, LiteValue>> {
    let mut output = BTreeMap::new();
    for _ in 0..count {
        if reader.pos >= reader.data.len() {
            break;
        }
        let record_start = reader.pos;
        let data_type = reader.read_u8()?;
        let name_len = reader.read_u8()? as usize;
        if data_type == 0 || data_type == 10 {
            return Err(Nd2Error::Invalid(format!(
                "unknown CLX lite metadata type {data_type}"
            )));
        }
        let name = if data_type == 76 {
            String::new()
        } else {
            reader.read_utf16_name(name_len)?
        };
        let value = match data_type {
            1 => LiteValue::Bool(reader.read_u8()? != 0),
            2 => LiteValue::Int(i64::from(reader.read_i32()?)),
            3 => LiteValue::UInt(u64::from(reader.read_u32()?)),
            4 => LiteValue::Int(reader.read_i64()?),
            5 | 7 => LiteValue::UInt(reader.read_u64()?),
            6 => LiteValue::Float(reader.read_f64()?),
            8 => read_lite_string(reader)?,
            9 => {
                let size = reader.read_u64()? as usize;
                LiteValue::Bytes(reader.read_exact(size)?.to_vec())
            }
            11 => {
                let item_count = reader.read_u32()? as usize;
                let length = reader.read_u64()? as usize;
                let consumed = reader.pos.checked_sub(record_start).ok_or_else(|| {
                    Nd2Error::Invalid("CLX lite record position underflow".to_string())
                })?;
                if length < consumed {
                    return Err(Nd2Error::Invalid(
                        "CLX lite level length is shorter than its header".to_string(),
                    ));
                }
                let body = reader.read_exact(length - consumed)?;
                let val = parse_lite_records(body, item_count)?;
                reader.skip(item_count * 8)?;
                LiteValue::Level(val)
            }
            76 => {
                return Err(Nd2Error::Unsupported(
                    "compressed CLX lite metadata is not implemented".to_string(),
                ));
            }
            _ => LiteValue::None,
        };
        insert_lite_value(&mut output, name, value);
    }
    Ok(output)
}

fn read_lite_string(reader: &mut LiteReader<'_>) -> Result<LiteValue> {
    let mut units = Vec::new();
    loop {
        let bytes = reader.read_exact(2)?;
        let unit = u16::from_le_bytes(bytes.try_into().unwrap());
        if unit == 0 {
            break;
        }
        units.push(unit);
    }
    let value = String::from_utf16(&units)
        .map_err(|err| Nd2Error::Invalid(format!("invalid UTF-16 metadata string: {err}")))?;
    Ok(LiteValue::String(value))
}

fn insert_lite_value(output: &mut BTreeMap<String, LiteValue>, name: String, value: LiteValue) {
    if let Some(existing) = output.get_mut(&name) {
        match existing {
            LiteValue::List(items) => items.push(value),
            other => {
                let first = std::mem::replace(other, LiteValue::None);
                *other = LiteValue::List(vec![first, value]);
            }
        }
    } else {
        output.insert(name, value);
    }
}

fn lite_u32(map: &BTreeMap<String, LiteValue>, key: &str) -> Result<u32> {
    match map.get(key) {
        Some(LiteValue::UInt(value)) => u32::try_from(*value)
            .map_err(|_| Nd2Error::Invalid(format!("metadata value {key} overflows u32"))),
        Some(LiteValue::Int(value)) if *value >= 0 => u32::try_from(*value)
            .map_err(|_| Nd2Error::Invalid(format!("metadata value {key} overflows u32"))),
        Some(other) => Err(Nd2Error::Invalid(format!(
            "metadata value {key} has unexpected type {other:?}"
        ))),
        None => Err(Nd2Error::Invalid(format!(
            "metadata value {key} is missing"
        ))),
    }
}

fn lite_optional_u32(map: &BTreeMap<String, LiteValue>, key: &str) -> Option<u32> {
    map.get(key).and_then(|value| match value {
        LiteValue::UInt(value) => u32::try_from(*value).ok(),
        LiteValue::Int(value) if *value >= 0 => u32::try_from(*value).ok(),
        _ => None,
    })
}

fn lite_optional_f64(map: &BTreeMap<String, LiteValue>, key: &str) -> Option<f64> {
    map.get(key).and_then(|value| match value {
        LiteValue::Float(value) => Some(*value),
        LiteValue::Int(value) => Some(*value as f64),
        LiteValue::UInt(value) => Some(*value as f64),
        _ => None,
    })
}

fn read_image_attributes(file: &File, file_size: u64, offset: u64) -> Result<ImageAttributes> {
    let header = read_chunk_header(file, file_size, offset)?;
    let mut payload = vec![0; header.payload_len as usize];
    read_exact_at(file, &mut payload, header.payload_offset)?;
    let parsed = parse_lite_records(&payload, 1)?;
    let attrs = match parsed.get("SLxImageAttributes") {
        Some(LiteValue::Level(attrs)) => attrs,
        _ => &parsed,
    };

    let component_count = lite_u32(attrs, "uiComp")?;
    let width_px = lite_u32(attrs, "uiWidth")?;
    let height_px = lite_u32(attrs, "uiHeight")?;
    let channel_count = if component_count == 3 || component_count == 4 {
        1
    } else {
        component_count.max(1)
    };
    let tile_width =
        lite_optional_u32(attrs, "uiTileWidth").filter(|value| *value > 0 && *value != width_px);
    let tile_height =
        lite_optional_u32(attrs, "uiTileHeight").filter(|value| *value > 0 && *value != height_px);

    Ok(ImageAttributes {
        bits_per_component_in_memory: lite_u32(attrs, "uiBpcInMemory")?,
        bits_per_component_significant: lite_u32(attrs, "uiBpcSignificant")?,
        component_count,
        height_px,
        sequence_count: lite_u32(attrs, "uiSequenceCount")?,
        width_bytes: lite_u32(attrs, "uiWidthBytes")?,
        width_px,
        compression: lite_optional_u32(attrs, "eCompression"),
        compression_level: lite_optional_f64(attrs, "dCompressionParam"),
        tile_height_px: tile_height,
        tile_width_px: tile_width,
        channel_count,
    })
}

fn uncompressed_pixel_bytes_per_plane(attributes: &ImageAttributes) -> Option<u64> {
    if attributes.compression.is_some_and(|value| value < 2) {
        return None;
    }
    u64::from(attributes.height_px).checked_mul(u64::from(attributes.width_bytes))
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_free_buffer(buffer: Nd2Buffer) {
    if !buffer.ptr.is_null() && buffer.len > 0 {
        unsafe {
            drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len));
        }
    }
    rsnd2_free_string(buffer.error);
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_version_probe_json(path: *const c_char) -> *mut c_char {
    ffi_string_result(path, |path| {
        let probe = Nd2VersionProbe::open(path)?;
        Ok(version_probe_json(&probe))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_summary_json(path: *const c_char) -> *mut c_char {
    ffi_string_result(path, |path| {
        let summary = Nd2Summary::open(path)?;
        Ok(summary_json(&summary))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_index_json(path: *const c_char) -> *mut c_char {
    ffi_string_result(path, |path| {
        let index = Nd2Index::open(path)?;
        Ok(index_json(&index))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_read_plane_payload(path: *const c_char, sequence: usize) -> Nd2Buffer {
    let result = ffi_path(path).and_then(|path| {
        let index = Nd2Index::open(&path)?;
        index.read_plane_payload(sequence)
    });
    match result {
        Ok(mut bytes) => {
            let ptr = bytes.as_mut_ptr();
            let len = bytes.len();
            std::mem::forget(bytes);
            Nd2Buffer {
                ptr,
                len,
                status: 0,
                error: std::ptr::null_mut(),
            }
        }
        Err(err) => Nd2Buffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            status: 1,
            error: c_string(format!("{err}")),
        },
    }
}

#[repr(C)]
pub struct Nd2Status {
    pub status: c_int,
    pub error: *mut c_char,
}

impl Nd2Status {
    fn ok() -> Self {
        Nd2Status {
            status: 0,
            error: std::ptr::null_mut(),
        }
    }

    fn err(message: impl Into<String>) -> Self {
        Nd2Status {
            status: 1,
            error: c_string(message.into()),
        }
    }
}

/// Open a persistent reader handle for batched frame reads. On success
/// `handle` is a non-null pointer that must be released with
/// `rsnd2_reader_free`. On failure `status` is non-zero and `error` carries
/// an owned message that must be released with `rsnd2_free_string`.
#[repr(C)]
pub struct Nd2OpenResult {
    pub handle: *mut c_void,
    pub status: c_int,
    pub error: *mut c_char,
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_reader_open(path: *const c_char) -> Nd2OpenResult {
    match ffi_path(path).and_then(|path| Nd2Reader::open(&path)) {
        Ok(reader) => Nd2OpenResult {
            handle: Box::into_raw(Box::new(reader)) as *mut c_void,
            status: 0,
            error: std::ptr::null_mut(),
        },
        Err(err) => Nd2OpenResult {
            handle: std::ptr::null_mut(),
            status: 1,
            error: c_string(format!("{err}")),
        },
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_reader_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle as *mut Nd2Reader));
        }
    }
}

/// Pack the pixel bytes for `n_indices` planes (each minus a `prefix_len`-byte
/// stamp) contiguously into the caller-owned buffer at `out_ptr`/`out_len`,
/// using up to `n_threads` concurrent positional reads (0 selects a default).
#[unsafe(no_mangle)]
pub extern "C" fn rsnd2_reader_read_frames(
    handle: *mut c_void,
    indices: *const u64,
    n_indices: usize,
    prefix_len: u64,
    out_ptr: *mut u8,
    out_len: usize,
    n_threads: usize,
) -> Nd2Status {
    if handle.is_null() {
        return Nd2Status::err("reader handle is null");
    }
    if out_ptr.is_null() && out_len > 0 {
        return Nd2Status::err("output buffer pointer is null");
    }
    if indices.is_null() && n_indices > 0 {
        return Nd2Status::err("indices pointer is null");
    }
    let reader = unsafe { &*(handle as *const Nd2Reader) };
    // `slice::from_raw_parts` requires a non-null, aligned pointer even for a
    // zero length, so synthesize empty slices when nothing was requested.
    let idx = if n_indices == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(indices, n_indices) }
    };
    let out = if out_len == 0 {
        &mut [][..]
    } else {
        unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) }
    };
    let threads = if n_threads == 0 { 1 } else { n_threads };
    match reader.read_frames_into(idx, prefix_len, out, threads) {
        Ok(()) => Nd2Status::ok(),
        Err(err) => Nd2Status::err(format!("{err}")),
    }
}

fn ffi_string_result<F>(path: *const c_char, f: F) -> *mut c_char
where
    F: FnOnce(&Path) -> Result<String>,
{
    let result = ffi_path(path).and_then(|path| f(&path));
    match result {
        Ok(value) => c_string(value),
        Err(err) => c_string(format!(r#"{{"error":{}}}"#, json_string(&err.to_string()))),
    }
}

fn ffi_path(path: *const c_char) -> Result<PathBuf> {
    if path.is_null() {
        return Err(Nd2Error::Invalid("path pointer is null".to_string()));
    }
    let path = unsafe { CStr::from_ptr(path) }
        .to_str()
        .map_err(|err| Nd2Error::Invalid(format!("path is not valid UTF-8: {err}")))?;
    Ok(PathBuf::from(path))
}

fn c_string(value: String) -> *mut c_char {
    let sanitized = value.replace('\0', "\\u0000");
    CString::new(sanitized)
        .expect("sanitized string should not contain interior NUL")
        .into_raw()
}

fn version_probe_json(probe: &Nd2VersionProbe) -> String {
    format!(
        "{{\"path\":{},\"file_size\":{},\"variant\":{},\"signature_name\":{},\"signature_version\":{}}}",
        json_string(&probe.path.display().to_string()),
        probe.file_size,
        json_string(variant_name(probe.variant)),
        json_string(&probe.signature_name),
        json_option_string(probe.signature_version.as_deref()),
    )
}

fn summary_json(summary: &Nd2Summary) -> String {
    format!(
        "{{\"path\":{},\"file_size\":{},\"variant\":{},\"signature_version\":{},\"filemap_offset\":{},\"chunk_count\":{},\"chunk_name_counts\":{},\"plane_count\":{},\"first_plane_payload_len\":{}}}",
        json_string(&summary.path.display().to_string()),
        summary.file_size,
        json_string(variant_name(summary.variant)),
        json_option_string(summary.signature_version.as_deref()),
        json_option_u64(summary.filemap_offset),
        summary.chunk_count,
        json_counts(&summary.chunk_name_counts),
        summary.plane_count,
        json_option_u64(summary.first_plane_payload_len),
    )
}

fn index_json(index: &Nd2Index) -> String {
    let planes = index
        .planes
        .iter()
        .map(|plane| {
            let prefix = plane
                .prefix8
                .iter()
                .map(|byte| byte.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"sequence\":{},\"chunk_offset\":{},\"payload_offset\":{},\"payload_len\":{},\"prefix8\":[{}]}}",
                plane.sequence, plane.chunk_offset, plane.payload_offset, plane.payload_len, prefix
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"path\":{},\"file_size\":{},\"variant\":{},\"signature_version\":{},\"filemap_offset\":{},\"chunk_count\":{},\"chunk_name_counts\":{},\"plane_count\":{},\"attributes\":{},\"planes\":[{}]}}",
        json_string(&index.path.display().to_string()),
        index.file_size,
        json_string(variant_name(index.variant)),
        json_option_string(index.signature_version.as_deref()),
        json_option_u64(index.filemap_offset),
        index.chunk_count,
        json_counts(&index.chunk_name_counts),
        index.planes.len(),
        json_image_attributes(index.attributes.as_ref()),
        planes,
    )
}

fn json_image_attributes(attributes: Option<&ImageAttributes>) -> String {
    match attributes {
        Some(attrs) => format!(
            "{{\"bitsPerComponentInMemory\":{},\"bitsPerComponentSignificant\":{},\"componentCount\":{},\"heightPx\":{},\"pixelDataType\":{},\"sequenceCount\":{},\"widthBytes\":{},\"widthPx\":{},\"compressionLevel\":{},\"compressionType\":{},\"tileHeightPx\":{},\"tileWidthPx\":{},\"channelCount\":{}}}",
            attrs.bits_per_component_in_memory,
            attrs.bits_per_component_significant,
            attrs.component_count,
            attrs.height_px,
            json_string(if attrs.bits_per_component_in_memory == 32 {
                "float"
            } else {
                "unsigned"
            }),
            attrs.sequence_count,
            attrs.width_bytes,
            attrs.width_px,
            json_option_f64(
                attrs
                    .compression_level
                    .filter(|_| attrs.compression.is_some_and(|value| value < 2))
            ),
            json_compression_type(attrs.compression),
            json_option_u64(attrs.tile_height_px.map(u64::from)),
            json_option_u64(attrs.tile_width_px.map(u64::from)),
            attrs.channel_count,
        ),
        None => "null".to_string(),
    }
}

fn json_compression_type(value: Option<u32>) -> String {
    match value {
        Some(0) => json_string("lossless"),
        Some(1) => json_string("lossy"),
        _ => "null".to_string(),
    }
}

fn variant_name(variant: Nd2Variant) -> &'static str {
    match variant {
        Nd2Variant::ModernChunked => "ModernChunked",
        Nd2Variant::LegacyJp2Like => "LegacyJp2Like",
    }
}

fn json_counts(counts: &BTreeMap<String, usize>) -> String {
    let entries = counts
        .iter()
        .map(|(key, value)| format!("{}:{}", json_string(key), value))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{entries}}}")
}

fn json_option_string(value: Option<&str>) -> String {
    value.map_or_else(|| "null".to_string(), json_string)
}

fn json_option_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_option_f64(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ch if ch <= '\u{1f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = file.read_at(buf, offset)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ));
        }
        offset += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(_file: &File, _buf: &mut [u8], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "positional reads are currently implemented for unix targets",
    ))
}
