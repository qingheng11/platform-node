use std::{
    fs::{self, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
    sync::Arc,
};

use tokio_rustls::rustls::{
    self,
    crypto::ring,
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer},
    server::WebPkiClientVerifier,
    ClientConfig, RootCertStore, ServerConfig,
};
pub use tokio_rustls::{rustls::pki_types::ServerName, TlsAcceptor, TlsConnector};

const ALPN: &[u8] = b"compute-node-test/1";

pub fn certificates(path: &Path) -> Result<Vec<CertificateDer<'static>>, String> {
    let file = fs::File::open(path).map_err(|error| format!("读取联调证书失败（{error}）"))?;
    let mut bytes = Vec::new();
    file.take(65_537)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > 65_536 {
        return Err("联调证书文件超出大小限制。".into());
    }
    let chain = CertificateDer::pem_slice_iter(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("联调证书格式无效（{error}）"))?;
    if chain.is_empty() {
        return Err("联调证书文件为空。".into());
    }
    Ok(chain)
}

fn key(path: &Path) -> Result<PrivateKeyDer<'static>, String> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| format!("打开联调私钥失败（{error}）"))?;
    let meta = file.metadata().map_err(|error| error.to_string())?;
    // SAFETY: geteuid takes no arguments.
    if !meta.is_file()
        || meta.uid() != unsafe { libc::geteuid() }
        || meta.mode() & 0o077 != 0
        || meta.nlink() != 1
        || meta.len() > 16_384
    {
        return Err("联调私钥必须是当前用户私有的普通文件。".into());
    }
    let mut bytes = Vec::new();
    file.take(16_385)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > 16_384 {
        return Err("联调私钥超出大小限制。".into());
    }
    PrivateKeyDer::from_pem_slice(&bytes).map_err(|_| "联调私钥格式无效。".into())
}

fn roots(path: &Path) -> Result<RootCertStore, String> {
    let mut store = RootCertStore::empty();
    for cert in certificates(path)? {
        store
            .add(cert)
            .map_err(|error| format!("联调 CA 无效（{error}）"))?;
    }
    Ok(store)
}

pub fn server(ca: &Path, cert: &Path, private_key: &Path) -> Result<TlsAcceptor, String> {
    server_with_alpn(ca, cert, private_key, ALPN)
}

pub fn server_with_alpn(
    ca: &Path,
    cert: &Path,
    private_key: &Path,
    alpn: &[u8],
) -> Result<TlsAcceptor, String> {
    let provider = Arc::new(ring::default_provider());
    let verifier =
        WebPkiClientVerifier::builder_with_provider(Arc::new(roots(ca)?), provider.clone())
            .build()
            .map_err(|error| error.to_string())?;
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|error| error.to_string())?
        .with_client_cert_verifier(verifier)
        .with_single_cert(certificates(cert)?, key(private_key)?)
        .map_err(|error| error.to_string())?;
    config.alpn_protocols = vec![alpn.to_vec()];
    Ok(TlsAcceptor::from(Arc::new(config)))
}

pub fn client(ca: &Path, cert: &Path, private_key: &Path) -> Result<TlsConnector, String> {
    let mut config = ClientConfig::builder_with_provider(Arc::new(ring::default_provider()))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|error| error.to_string())?
        .with_root_certificates(roots(ca)?)
        .with_client_auth_cert(certificates(cert)?, key(private_key)?)
        .map_err(|error| error.to_string())?;
    config.alpn_protocols = vec![ALPN.to_vec()];
    Ok(TlsConnector::from(Arc::new(config)))
}
