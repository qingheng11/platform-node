use std::time::Duration;

use ring::signature::{UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::local_node::{Output, Run};

pub const ALPN: &str = "/compute/scheduled-experience/1";
pub const FRAME_MAX: usize = 64 * 1024;
pub const RESULT_TTL: Duration = Duration::from_secs(120);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub endpoint_id: String,
    pub relay_url: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registration {
    pub target: Target,
    pub signature: Vec<u8>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub browser_endpoint_id: String,
    pub ticket: Uuid,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attach {
    pub task_id: Uuid,
    pub attempt_id: Uuid,
    pub ticket: Uuid,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Result { attempt_id: Uuid, output: Output },
    Error { code: String },
}

pub fn endpoint_bytes(id: &str) -> Option<[u8; 32]> {
    if id.len() != 64
        || !id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&id[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

pub fn valid_relay(url: &str) -> bool {
    url.len() <= 256
        && url.starts_with("https://")
        && url.len() > 8
        && !url
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b"@?#\\".contains(&b))
}

impl Target {
    pub fn signing_bytes(&self, run: &Run) -> Vec<u8> {
        let mut bytes = b"compute:scheduled-experience:endpoint:v1\0".to_vec();
        bytes.extend(run.installation_id.as_bytes());
        bytes.extend(run.run_id.as_bytes());
        bytes.extend(run.generation.to_be_bytes());
        bytes.extend(self.endpoint_id.as_bytes());
        bytes.extend(self.relay_url.as_bytes());
        bytes
    }
}

impl Registration {
    pub fn verify(&self, run: &Run, relay: &str) -> bool {
        self.target.relay_url == relay
            && self.signature.len() == 64
            && endpoint_bytes(&self.target.endpoint_id).is_some_and(|key| {
                UnparsedPublicKey::new(&ED25519, key)
                    .verify(&self.target.signing_bytes(run), &self.signature)
                    .is_ok()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};

    #[test]
    fn registration_signature_binds_installation_generation_endpoint_and_relay() {
        let pair = Ed25519KeyPair::from_seed_unchecked(&[7; 32]).unwrap();
        let mut run = Run {
            installation_id: Uuid::from_u128(1),
            run_id: Uuid::from_u128(2),
            generation: 3,
        };
        let target = Target {
            endpoint_id: pair
                .public_key()
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            relay_url: "https://relay.example.invalid/".into(),
        };
        let signature = pair.sign(&target.signing_bytes(&run)).as_ref().to_vec();
        let registration = Registration { target, signature };
        assert!(registration.verify(&run, &registration.target.relay_url));
        assert!(!registration.verify(&run, "https://another.example.invalid/"));
        run.generation += 1;
        assert!(!registration.verify(&run, &registration.target.relay_url));
    }
}
