//! Weight-hash audit trail: sha256 over a checkpoint's `model.safetensors`,
//! hex-encoded. Computed once at load time and carried on every model's
//! [`crate::types::ModelInfo`] and every inference response — see the crate
//! root docs on why.

use std::path::Path;

use sha2::{Digest, Sha256};

/// sha256 of `path`'s bytes, as 64 lowercase hex characters.
///
/// Streams bounded chunks, including multi-gigabyte GGUFs. Paid once during
/// startup; no second checkpoint-sized host allocation is needed.
pub fn sha256_hex_file(path: impl AsRef<Path>) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

/// sha256 of `bytes`, as 64 lowercase hex characters. Used for the checkpoint
/// weight-hash audit trail ([`sha256_hex_file`]) AND, separately, for the
/// optional `--log-input-hash` trace attribute on `/v1/spans` requests (see
/// `worker.rs`'s `spans`/`spans_credentials` methods) — the same primitive,
/// two different inputs (a file's bytes vs. a request's input text), so it's
/// factored out once rather than duplicated.
pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    hex_encode(&digest)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha256_hex_file_matches_a_known_vector() {
        // sha256("abc") — the standard NIST test vector.
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(b"abc").expect("write");
        let got = sha256_hex_file(f.path()).expect("hash");
        assert_eq!(got, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sha256_hex_file_is_64_lowercase_hex_chars() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(b"some checkpoint bytes").expect("write");
        let got = sha256_hex_file(f.path()).expect("hash");
        assert_eq!(got.len(), 64);
        assert!(got.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn sha256_hex_file_differs_for_different_content() {
        let mut a = tempfile::NamedTempFile::new().expect("tempfile");
        a.write_all(b"content a").expect("write");
        let mut b = tempfile::NamedTempFile::new().expect("tempfile");
        b.write_all(b"content b").expect("write");
        assert_ne!(sha256_hex_file(a.path()).unwrap(), sha256_hex_file(b.path()).unwrap());
    }

    #[test]
    fn sha256_hex_bytes_matches_the_same_nist_test_vector() {
        assert_eq!(
            sha256_hex_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn streamed_hash_handles_multiple_chunks_and_partial_tail() {
        let bytes: Vec<u8> = (0..(2 * 65536 + 17)).map(|i| (i % 251) as u8).collect();
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&bytes).unwrap();
        assert_eq!(sha256_hex_file(f.path()).unwrap(), sha256_hex_bytes(&bytes));
    }

    #[test]
    fn sha256_hex_file_missing_file_is_a_loud_io_error() {
        let err = sha256_hex_file("/nonexistent/path/to/nowhere.safetensors")
            .expect_err("missing file must error, not silently hash nothing");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
