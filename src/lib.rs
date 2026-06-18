use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
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
pub struct Nd2Index {
    pub path: PathBuf,
    pub file_size: u64,
    pub variant: Nd2Variant,
    pub signature_version: Option<String>,
    pub filemap_offset: Option<u64>,
    pub chunk_count: usize,
    pub chunk_name_counts: BTreeMap<String, usize>,
    pub planes: Vec<PlaneRecord>,
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
        for entry in &entries {
            let base = chunk_base_name(&entry.name).to_string();
            *chunk_name_counts.entry(base).or_insert(0) += 1;
            if let Some(sequence) = image_sequence(&entry.name) {
                plane_entries.push((sequence, entry.offset));
            }
        }

        plane_entries.sort_unstable_by_key(|(sequence, _)| *sequence);
        let mut planes = Vec::with_capacity(plane_entries.len());
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

        Ok(Nd2Index {
            path: path.to_path_buf(),
            file_size,
            variant: Nd2Variant::ModernChunked,
            signature_version,
            filemap_offset: Some(filemap_offset),
            chunk_count: entries.len(),
            chunk_name_counts,
            planes,
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
    let tail_len = file_size.min(8192) as usize;
    let tail_offset = file_size - tail_len as u64;
    let mut tail = vec![0; tail_len];
    read_exact_at(file, &mut tail, tail_offset)?;
    let sig_pos = find_last(&tail, CHUNK_MAP_SIGNATURE)
        .ok_or_else(|| Nd2Error::Invalid("missing ND2 chunk map signature footer".to_string()))?;
    let ptr_pos = sig_pos + CHUNK_MAP_SIGNATURE.len();
    if ptr_pos + 8 > tail.len() {
        return Err(Nd2Error::Invalid(
            "truncated ND2 chunk map signature footer".to_string(),
        ));
    }
    let offset = u64::from_le_bytes(tail[ptr_pos..ptr_pos + 8].try_into().unwrap());
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

fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .enumerate()
        .rev()
        .find_map(|(idx, window)| (window == needle).then_some(idx))
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
