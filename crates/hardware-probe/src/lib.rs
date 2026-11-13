//! Cross-component hardware discovery for Platform Node.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nvml_wrapper::{
    cuda_driver_version_major, cuda_driver_version_minor, enum_wrappers::device::TemperatureSensor,
    Nvml,
};
use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};
use thiserror::Error;

/// Version of the serialized hardware contract.
pub const HARDWARE_SCHEMA_VERSION: u32 = 1;

/// Version of the immutable device capability contract sent to Platform.
pub const DEVICE_CAPABILITY_SCHEMA_VERSION: u32 = 1;

/// Stable identifier of the volume containing the configured model directory.
pub const MODEL_DISK_ID: &str = "disk_models";

const MAX_DEVICE_GPUS: usize = 16;
const MAX_DEVICE_DISKS: usize = 32;
const MAX_DEVICE_ERROR_CODES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareSnapshotV1 {
    pub schema_version: u32,
    pub captured_at_unix_ms: u64,
    pub os: OsInfo,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpus: Vec<GpuInfo>,
    pub warnings: Vec<HardwareWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OsInfo {
    pub family: String,
    pub version: String,
    pub arch: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuVendor {
    Intel,
    Amd,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub vendor: CpuVendor,
    pub vendor_id: String,
    pub model: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub frequency_mhz: u64,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
    pub index: u32,
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub utilization_percent: Option<u32>,
    pub temperature_celsius: Option<u32>,
    pub driver_version: Option<String>,
    pub cuda_driver_version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareWarningCode {
    PhysicalCoreCountUnavailable,
    NvmlUnavailable,
    NvidiaDeviceCountUnavailable,
    NvidiaDeviceUnavailable,
    NvidiaDeviceNameUnavailable,
    NvidiaDeviceMemoryUnavailable,
    NvidiaDeviceUtilizationUnavailable,
    NvidiaDeviceTemperatureUnavailable,
    NvidiaDriverVersionUnavailable,
    NvidiaCudaDriverVersionUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareWarning {
    pub code: HardwareWarningCode,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HardwareProbeError {
    #[error("no CPU information is available from the operating system")]
    CpuUnavailable,
    #[error("system memory information is unavailable")]
    MemoryUnavailable,
}

/// Confidence assigned to a Platform-facing hardware projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DeviceProbeStatus {
    Verified,
    Partial,
    Unverified,
}

/// Stable, machine-readable probe errors accepted by the device API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DeviceProbeErrorCode {
    PhysicalCoreCountUnavailable,
    NvmlUnavailable,
    NvidiaDeviceCountUnavailable,
    NvidiaDeviceUnavailable,
    NvidiaDeviceNameUnavailable,
    NvidiaDeviceMemoryUnavailable,
    NvidiaDriverVersionUnavailable,
    NvidiaCudaDriverVersionUnavailable,
    NvidiaComputeCapabilityUnavailable,
    NvidiaPciInfoUnavailable,
    NvidiaDeviceLimitExceeded,
}

impl DeviceProbeErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PhysicalCoreCountUnavailable => "PHYSICAL_CORE_COUNT_UNAVAILABLE",
            Self::NvmlUnavailable => "NVML_UNAVAILABLE",
            Self::NvidiaDeviceCountUnavailable => "NVIDIA_DEVICE_COUNT_UNAVAILABLE",
            Self::NvidiaDeviceUnavailable => "NVIDIA_DEVICE_UNAVAILABLE",
            Self::NvidiaDeviceNameUnavailable => "NVIDIA_DEVICE_NAME_UNAVAILABLE",
            Self::NvidiaDeviceMemoryUnavailable => "NVIDIA_DEVICE_MEMORY_UNAVAILABLE",
            Self::NvidiaDriverVersionUnavailable => "NVIDIA_DRIVER_VERSION_UNAVAILABLE",
            Self::NvidiaCudaDriverVersionUnavailable => "NVIDIA_CUDA_DRIVER_VERSION_UNAVAILABLE",
            Self::NvidiaComputeCapabilityUnavailable => "NVIDIA_COMPUTE_CAPABILITY_UNAVAILABLE",
            Self::NvidiaPciInfoUnavailable => "NVIDIA_PCI_INFO_UNAVAILABLE",
            Self::NvidiaDeviceLimitExceeded => "NVIDIA_DEVICE_LIMIT_EXCEEDED",
        }
    }
}

impl fmt::Display for DeviceProbeErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceOperatingSystemFamily {
    Windows,
    Linux,
    Macos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceArchitecture {
    X86_64,
    Arm64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DeviceGpuBackend {
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceOperatingSystem {
    pub family: DeviceOperatingSystemFamily,
    pub distribution: Option<String>,
    pub version: String,
    pub architecture: DeviceArchitecture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCpuCapability {
    pub vendor: Option<String>,
    pub model: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemoryCapability {
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceGpuCapability {
    pub id: String,
    pub vendor: String,
    pub name: String,
    pub vram_total_bytes: u64,
    pub backend: DeviceGpuBackend,
    pub runtime: Option<String>,
    pub runtime_version: Option<String>,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
    pub verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceDiskCapability {
    pub id: String,
    pub label: String,
    pub file_system: Option<String>,
    pub total_bytes: u64,
}

/// Static, Platform-facing hardware facts. The timestamp is intentionally kept
/// as Unix milliseconds so the HTTP owner can format it as RFC 3339 UTC.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCapabilitySnapshot {
    pub schema_version: u32,
    pub captured_at_unix_ms: u64,
    pub probe_status: DeviceProbeStatus,
    pub operating_system: DeviceOperatingSystem,
    pub cpu: DeviceCpuCapability,
    pub memory: DeviceMemoryCapability,
    pub gpus: Vec<DeviceGpuCapability>,
    pub disks: Vec<DeviceDiskCapability>,
    pub error_codes: Vec<DeviceProbeErrorCode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceGpuResource {
    pub id: String,
    pub vram_free_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceDiskResource {
    pub id: String,
    pub free_bytes: u64,
}

/// Dynamic resource availability paired with a capability snapshot. Probe
/// diagnostics are local metadata; the HTTP owner maps only resource fields to
/// the heartbeat contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceResourceSnapshot {
    pub captured_at_unix_ms: u64,
    pub memory_available_bytes: u64,
    pub gpus: Vec<DeviceGpuResource>,
    pub disks: Vec<DeviceDiskResource>,
    pub probe_status: DeviceProbeStatus,
    pub error_codes: Vec<DeviceProbeErrorCode>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeviceProbeError {
    #[error(transparent)]
    Hardware(#[from] HardwareProbeError),
    #[error("the current operating system is not supported by the device API")]
    UnsupportedOperatingSystem,
    #[error("the current CPU architecture is not supported by the device API")]
    UnsupportedArchitecture,
    #[error("the volume containing the configured model directory is unavailable")]
    ModelDiskUnavailable,
    #[error("the configured model directory cannot be resolved to an absolute path")]
    ModelDirectoryUnavailable,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeviceSnapshotValidationError {
    #[error("the capability snapshot exceeds the GPU limit")]
    CapabilityGpuLimitExceeded,
    #[error("the capability snapshot disk count is outside the API boundary")]
    CapabilityDiskCountInvalid,
    #[error("the resource snapshot exceeds the GPU limit")]
    ResourceGpuLimitExceeded,
    #[error("the resource snapshot exceeds the disk limit")]
    ResourceDiskLimitExceeded,
    #[error("the capability snapshot contains a duplicate GPU id")]
    DuplicateCapabilityGpuId,
    #[error("the capability snapshot contains a duplicate disk id")]
    DuplicateCapabilityDiskId,
    #[error("the resource snapshot contains a duplicate GPU id")]
    DuplicateResourceGpuId,
    #[error("the resource snapshot contains a duplicate disk id")]
    DuplicateResourceDiskId,
    #[error("resource GPU ids do not match the capability snapshot")]
    GpuIdsMismatch,
    #[error("resource disk ids do not match the capability snapshot")]
    DiskIdsMismatch,
    #[error("available memory exceeds the capability total")]
    MemoryAvailableExceedsTotal,
    #[error("free VRAM exceeds the corresponding capability total for {0}")]
    GpuFreeExceedsTotal(String),
    #[error("free disk space exceeds the corresponding capability total for {0}")]
    DiskFreeExceedsTotal(String),
}

#[derive(Debug)]
struct SystemFacts {
    os: OsInfo,
    cpu: CpuInfo,
    memory: MemoryInfo,
    warnings: Vec<HardwareWarningCode>,
}

#[derive(Debug, Default)]
struct NvidiaProbeResult {
    gpus: Vec<GpuInfo>,
    warnings: Vec<HardwareWarningCode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PciSortKey {
    domain: u32,
    bus: u32,
    device: u32,
}

#[derive(Debug)]
struct StaticGpuCandidate {
    nvml_index: u32,
    pci_sort_key: Option<PciSortKey>,
    name: String,
    vram_total_bytes: u64,
    driver_version: Option<String>,
    runtime_version: Option<String>,
    compute_capability: Option<String>,
    verified: bool,
}

#[derive(Debug)]
struct ResourceGpuCandidate {
    nvml_index: u32,
    pci_sort_key: Option<PciSortKey>,
    vram_free_bytes: u64,
}

#[derive(Debug, Default)]
struct DeviceStaticNvidiaProbe {
    gpus: Vec<DeviceGpuCapability>,
    error_codes: Vec<DeviceProbeErrorCode>,
}

#[derive(Debug, Default)]
struct DeviceResourceNvidiaProbe {
    gpus: Vec<DeviceGpuResource>,
    error_codes: Vec<DeviceProbeErrorCode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NvidiaAdapterPresence {
    Present,
    Absent,
    Unknown,
}

#[derive(Debug, Clone)]
struct DiskObservation {
    mount_point: PathBuf,
    file_system: Option<String>,
    total_bytes: u64,
    free_bytes: u64,
}

/// Collects a point-in-time view of the local CPU, memory, and supported GPUs.
///
/// NVIDIA discovery is best-effort: systems without NVML or an NVIDIA driver
/// still return a successful CPU/memory snapshot with an `nvml_unavailable`
/// warning. No external programs or shell commands are invoked.
pub fn probe_hardware() -> Result<HardwareSnapshotV1, HardwareProbeError> {
    let system = collect_system_facts()?;
    let nvidia = probe_nvidia();

    Ok(assemble_snapshot(
        unix_timestamp_ms(SystemTime::now()),
        system,
        nvidia,
    ))
}

/// Probes immutable device capability data suitable for the Platform device
/// capability endpoint.
///
/// Only the volume containing `model_directory` is reported, under the stable
/// id [`MODEL_DISK_ID`]. Hardware serial numbers, PCI addresses and mount paths
/// never enter the returned projection.
pub fn probe_static_capability(
    model_directory: &Path,
) -> Result<DeviceCapabilitySnapshot, DeviceProbeError> {
    let system = collect_system_facts()?;
    let operating_system = device_operating_system(&system.os)?;
    let disk = probe_model_disk(model_directory)?;
    let nvidia = probe_device_nvidia_static();

    let mut error_codes = system
        .warnings
        .iter()
        .filter_map(|warning| map_system_warning(*warning))
        .chain(nvidia.error_codes)
        .collect::<Vec<_>>();
    normalize_error_codes(&mut error_codes);

    let logical_cores = system.cpu.logical_cores.clamp(1, 4096);
    let physical_cores = system.cpu.physical_cores.clamp(1, 1024).min(logical_cores);
    let mut features = system
        .cpu
        .features
        .iter()
        .map(|feature| truncate_chars(&feature.to_ascii_uppercase(), 40))
        .filter(|feature| !feature.is_empty())
        .collect::<Vec<_>>();
    features.sort_unstable();
    features.dedup();
    features.truncate(128);

    let vendor_id = system.cpu.vendor_id.trim();
    let vendor = (!vendor_id.is_empty() && !vendor_id.eq_ignore_ascii_case("unknown"))
        .then(|| truncate_chars(vendor_id, 120));

    Ok(DeviceCapabilitySnapshot {
        schema_version: DEVICE_CAPABILITY_SCHEMA_VERSION,
        captured_at_unix_ms: unix_timestamp_ms(SystemTime::now()),
        probe_status: probe_status(&error_codes),
        operating_system,
        cpu: DeviceCpuCapability {
            vendor,
            model: nonempty_truncated(&system.cpu.model, "Unknown CPU", 200),
            physical_cores,
            logical_cores,
            features,
        },
        memory: DeviceMemoryCapability {
            total_bytes: api_int64(system.memory.total_bytes).max(1),
        },
        gpus: nvidia.gpus,
        disks: vec![DeviceDiskCapability {
            id: MODEL_DISK_ID.to_owned(),
            label: "模型盘".to_owned(),
            file_system: disk.file_system,
            total_bytes: api_int64(disk.total_bytes).max(1),
        }],
        error_codes,
    })
}

/// Probes the latest free memory, VRAM and model-volume space. Call
/// [`validate_resource_snapshot`] against the active capability before sending
/// a heartbeat.
pub fn probe_dynamic_resources(
    model_directory: &Path,
) -> Result<DeviceResourceSnapshot, DeviceProbeError> {
    let system = System::new_all();
    let total_memory = system.total_memory();
    if total_memory == 0 {
        return Err(HardwareProbeError::MemoryUnavailable.into());
    }

    let disk = probe_model_disk(model_directory)?;
    let nvidia = probe_device_nvidia_resources();
    let mut error_codes = nvidia.error_codes;
    normalize_error_codes(&mut error_codes);

    Ok(DeviceResourceSnapshot {
        captured_at_unix_ms: unix_timestamp_ms(SystemTime::now()),
        memory_available_bytes: api_int64(system.available_memory().min(total_memory)),
        gpus: nvidia.gpus,
        disks: vec![DeviceDiskResource {
            id: MODEL_DISK_ID.to_owned(),
            free_bytes: api_int64(disk.free_bytes.min(disk.total_bytes)),
        }],
        probe_status: probe_status(&error_codes),
        error_codes,
    })
}

/// Verifies that dynamic IDs and free values are valid for the immutable
/// capability snapshot currently referenced by a heartbeat.
pub fn validate_resource_snapshot(
    capability: &DeviceCapabilitySnapshot,
    resources: &DeviceResourceSnapshot,
) -> Result<(), DeviceSnapshotValidationError> {
    if capability.gpus.len() > MAX_DEVICE_GPUS {
        return Err(DeviceSnapshotValidationError::CapabilityGpuLimitExceeded);
    }
    if capability.disks.is_empty() || capability.disks.len() > MAX_DEVICE_DISKS {
        return Err(DeviceSnapshotValidationError::CapabilityDiskCountInvalid);
    }
    if resources.gpus.len() > MAX_DEVICE_GPUS {
        return Err(DeviceSnapshotValidationError::ResourceGpuLimitExceeded);
    }
    if resources.disks.len() > MAX_DEVICE_DISKS {
        return Err(DeviceSnapshotValidationError::ResourceDiskLimitExceeded);
    }
    if resources.memory_available_bytes > capability.memory.total_bytes {
        return Err(DeviceSnapshotValidationError::MemoryAvailableExceedsTotal);
    }

    let gpu_totals = collect_unique_totals(
        capability
            .gpus
            .iter()
            .map(|gpu| (gpu.id.as_str(), gpu.vram_total_bytes)),
        DeviceSnapshotValidationError::DuplicateCapabilityGpuId,
    )?;
    let resource_gpu_values = collect_unique_totals(
        resources
            .gpus
            .iter()
            .map(|gpu| (gpu.id.as_str(), gpu.vram_free_bytes)),
        DeviceSnapshotValidationError::DuplicateResourceGpuId,
    )?;
    if gpu_totals.keys().collect::<HashSet<_>>()
        != resource_gpu_values.keys().collect::<HashSet<_>>()
    {
        return Err(DeviceSnapshotValidationError::GpuIdsMismatch);
    }
    for (id, free_bytes) in resource_gpu_values {
        if free_bytes > gpu_totals[id] {
            return Err(DeviceSnapshotValidationError::GpuFreeExceedsTotal(
                id.to_owned(),
            ));
        }
    }

    let disk_totals = collect_unique_totals(
        capability
            .disks
            .iter()
            .map(|disk| (disk.id.as_str(), disk.total_bytes)),
        DeviceSnapshotValidationError::DuplicateCapabilityDiskId,
    )?;
    let resource_disk_values = collect_unique_totals(
        resources
            .disks
            .iter()
            .map(|disk| (disk.id.as_str(), disk.free_bytes)),
        DeviceSnapshotValidationError::DuplicateResourceDiskId,
    )?;
    if disk_totals.keys().collect::<HashSet<_>>()
        != resource_disk_values.keys().collect::<HashSet<_>>()
    {
        return Err(DeviceSnapshotValidationError::DiskIdsMismatch);
    }
    for (id, free_bytes) in resource_disk_values {
        if free_bytes > disk_totals[id] {
            return Err(DeviceSnapshotValidationError::DiskFreeExceedsTotal(
                id.to_owned(),
            ));
        }
    }

    Ok(())
}

/// Classifies the stable CPUID vendor string without relying on model names.
pub fn classify_cpu_vendor(vendor_id: &str) -> CpuVendor {
    let normalized = vendor_id.trim().to_ascii_lowercase();
    if normalized == "genuineintel" || normalized.contains("intel") {
        CpuVendor::Intel
    } else if normalized == "authenticamd" || normalized.contains("amd") {
        CpuVendor::Amd
    } else {
        CpuVendor::Other
    }
}

fn collect_system_facts() -> Result<SystemFacts, HardwareProbeError> {
    let system = System::new_all();
    let cpus = system.cpus();
    let first_cpu = cpus.first().ok_or(HardwareProbeError::CpuUnavailable)?;

    let total_memory = system.total_memory();
    if total_memory == 0 {
        return Err(HardwareProbeError::MemoryUnavailable);
    }

    let (cpuid_vendor, features) = cpuid_facts();
    let vendor_id = cpuid_vendor
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| first_cpu.vendor_id().trim().to_owned());

    let logical_cores = usize_to_u32(cpus.len());
    let (physical_cores, warnings) = match System::physical_core_count() {
        Some(value) if value > 0 => (usize_to_u32(value), Vec::new()),
        _ => (
            logical_cores,
            vec![HardwareWarningCode::PhysicalCoreCountUnavailable],
        ),
    };

    let frequency_mhz = cpus.iter().map(|cpu| cpu.frequency()).max().unwrap_or(0);
    let model = first_cpu.brand().trim();

    Ok(SystemFacts {
        os: OsInfo {
            family: System::name().unwrap_or_else(|| std::env::consts::OS.to_owned()),
            version: System::os_version().unwrap_or_else(|| "unknown".to_owned()),
            arch: nonempty_or(System::cpu_arch(), std::env::consts::ARCH),
        },
        cpu: CpuInfo {
            vendor: classify_cpu_vendor(&vendor_id),
            vendor_id: nonempty_or(vendor_id, "unknown"),
            model: if model.is_empty() {
                "Unknown CPU".to_owned()
            } else {
                model.to_owned()
            },
            physical_cores,
            logical_cores,
            frequency_mhz,
            features,
        },
        memory: MemoryInfo {
            total_bytes: total_memory,
            available_bytes: system.available_memory(),
        },
        warnings,
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuid_facts() -> (Option<String>, Vec<String>) {
    let cpuid = raw_cpuid::CpuId::new();
    let vendor = cpuid.get_vendor_info().map(|info| info.as_str().to_owned());
    let mut features = Vec::with_capacity(4);

    if let Some(info) = cpuid.get_feature_info() {
        if info.has_sse42() {
            features.push("sse4.2".to_owned());
        }
        if info.has_avx() {
            features.push("avx".to_owned());
        }
        if info.has_fma() {
            features.push("fma".to_owned());
        }
    }
    if cpuid
        .get_extended_feature_info()
        .is_some_and(|info| info.has_avx2())
    {
        features.push("avx2".to_owned());
    }

    features.sort_unstable();
    (vendor, features)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpuid_facts() -> (Option<String>, Vec<String>) {
    (None, Vec::new())
}

fn probe_nvidia() -> NvidiaProbeResult {
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(_) => {
            return NvidiaProbeResult {
                gpus: Vec::new(),
                warnings: match detect_nvidia_adapter_presence() {
                    NvidiaAdapterPresence::Absent => Vec::new(),
                    NvidiaAdapterPresence::Present | NvidiaAdapterPresence::Unknown => {
                        vec![HardwareWarningCode::NvmlUnavailable]
                    }
                },
            };
        }
    };

    let mut warnings = Vec::new();
    let driver_version = nvml.sys_driver_version().map_or_else(
        |_| {
            warnings.push(HardwareWarningCode::NvidiaDriverVersionUnavailable);
            None
        },
        Some,
    );
    let cuda_driver_version = nvml.sys_cuda_driver_version().map_or_else(
        |_| {
            warnings.push(HardwareWarningCode::NvidiaCudaDriverVersionUnavailable);
            None
        },
        |version| Some(format_cuda_driver_version(version)),
    );

    let device_count = match nvml.device_count() {
        Ok(count) => count,
        Err(_) => {
            warnings.push(HardwareWarningCode::NvidiaDeviceCountUnavailable);
            return NvidiaProbeResult {
                gpus: Vec::new(),
                warnings,
            };
        }
    };

    let mut gpus = Vec::with_capacity(device_count as usize);
    for index in 0..device_count {
        let device = match nvml.device_by_index(index) {
            Ok(device) => device,
            Err(_) => {
                warnings.push(HardwareWarningCode::NvidiaDeviceUnavailable);
                continue;
            }
        };
        let name = device.name().unwrap_or_else(|_| {
            warnings.push(HardwareWarningCode::NvidiaDeviceNameUnavailable);
            format!("NVIDIA GPU {index}")
        });
        let uuid = device.uuid().ok();
        let memory = match device.memory_info() {
            Ok(memory) => memory,
            Err(_) => {
                warnings.push(HardwareWarningCode::NvidiaDeviceMemoryUnavailable);
                continue;
            }
        };
        let utilization_percent = device.utilization_rates().map_or_else(
            |_| {
                warnings.push(HardwareWarningCode::NvidiaDeviceUtilizationUnavailable);
                None
            },
            |utilization| Some(utilization.gpu),
        );
        let temperature_celsius = device.temperature(TemperatureSensor::Gpu).map_or_else(
            |_| {
                warnings.push(HardwareWarningCode::NvidiaDeviceTemperatureUnavailable);
                None
            },
            Some,
        );

        gpus.push(GpuInfo {
            id: format!("nvidia:{index}"),
            uuid,
            index,
            vendor: GpuVendor::Nvidia,
            name,
            vram_total_bytes: memory.total,
            vram_free_bytes: memory.free,
            utilization_percent,
            temperature_celsius,
            driver_version: driver_version.clone(),
            cuda_driver_version: cuda_driver_version.clone(),
        });
    }

    NvidiaProbeResult { gpus, warnings }
}

fn probe_device_nvidia_static() -> DeviceStaticNvidiaProbe {
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(_) => {
            return DeviceStaticNvidiaProbe {
                gpus: Vec::new(),
                error_codes: nvml_unavailable_errors(detect_nvidia_adapter_presence()),
            };
        }
    };

    let mut error_codes = Vec::new();
    let driver_version = nvml.sys_driver_version().map_or_else(
        |_| {
            error_codes.push(DeviceProbeErrorCode::NvidiaDriverVersionUnavailable);
            None
        },
        |value| Some(truncate_chars(&value, 80)),
    );
    let runtime_version = nvml.sys_cuda_driver_version().map_or_else(
        |_| {
            error_codes.push(DeviceProbeErrorCode::NvidiaCudaDriverVersionUnavailable);
            None
        },
        |value| Some(format_cuda_driver_version(value)),
    );
    let device_count = match nvml.device_count() {
        Ok(count) => count,
        Err(_) => {
            error_codes.push(DeviceProbeErrorCode::NvidiaDeviceCountUnavailable);
            normalize_error_codes(&mut error_codes);
            return DeviceStaticNvidiaProbe {
                gpus: Vec::new(),
                error_codes,
            };
        }
    };

    let mut candidates = Vec::with_capacity((device_count as usize).min(MAX_DEVICE_GPUS));
    for nvml_index in 0..device_count {
        let device = match nvml.device_by_index(nvml_index) {
            Ok(device) => device,
            Err(_) => {
                error_codes.push(DeviceProbeErrorCode::NvidiaDeviceUnavailable);
                continue;
            }
        };

        let (name, name_verified) = device.name().map_or_else(
            |_| {
                error_codes.push(DeviceProbeErrorCode::NvidiaDeviceNameUnavailable);
                (format!("NVIDIA GPU {nvml_index}"), false)
            },
            |name| (nonempty_truncated(&name, "NVIDIA GPU", 200), true),
        );
        let memory = match device.memory_info() {
            Ok(memory) if memory.total > 0 => memory,
            _ => {
                error_codes.push(DeviceProbeErrorCode::NvidiaDeviceMemoryUnavailable);
                continue;
            }
        };
        let compute_capability = device.cuda_compute_capability().map_or_else(
            |_| {
                error_codes.push(DeviceProbeErrorCode::NvidiaComputeCapabilityUnavailable);
                None
            },
            |capability| Some(format!("{}.{}", capability.major, capability.minor)),
        );
        let pci_sort_key = device.pci_info().map_or_else(
            |_| {
                error_codes.push(DeviceProbeErrorCode::NvidiaPciInfoUnavailable);
                None
            },
            |pci| {
                Some(PciSortKey {
                    domain: pci.domain,
                    bus: pci.bus,
                    device: pci.device,
                })
            },
        );
        let verified = name_verified
            && driver_version.is_some()
            && runtime_version.is_some()
            && compute_capability.is_some()
            && pci_sort_key.is_some();

        candidates.push(StaticGpuCandidate {
            nvml_index,
            pci_sort_key,
            name,
            vram_total_bytes: api_int64(memory.total),
            driver_version: driver_version.clone(),
            runtime_version: runtime_version.clone(),
            compute_capability,
            verified,
        });
    }

    sort_gpu_candidates(&mut candidates, |candidate| {
        (candidate.pci_sort_key, candidate.nvml_index)
    });
    if candidates.len() > MAX_DEVICE_GPUS {
        candidates.truncate(MAX_DEVICE_GPUS);
        error_codes.push(DeviceProbeErrorCode::NvidiaDeviceLimitExceeded);
    }
    let gpus = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| DeviceGpuCapability {
            id: format!("gpu_{index}"),
            vendor: "NVIDIA".to_owned(),
            name: candidate.name,
            vram_total_bytes: candidate.vram_total_bytes,
            backend: DeviceGpuBackend::Cuda,
            runtime: Some("CUDA".to_owned()),
            runtime_version: candidate.runtime_version,
            driver_version: candidate.driver_version,
            compute_capability: candidate.compute_capability,
            verified: candidate.verified,
        })
        .collect();
    normalize_error_codes(&mut error_codes);

    DeviceStaticNvidiaProbe { gpus, error_codes }
}

fn probe_device_nvidia_resources() -> DeviceResourceNvidiaProbe {
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(_) => {
            return DeviceResourceNvidiaProbe {
                gpus: Vec::new(),
                error_codes: nvml_unavailable_errors(detect_nvidia_adapter_presence()),
            };
        }
    };
    let mut error_codes = Vec::new();
    let device_count = match nvml.device_count() {
        Ok(count) => count,
        Err(_) => {
            return DeviceResourceNvidiaProbe {
                gpus: Vec::new(),
                error_codes: vec![DeviceProbeErrorCode::NvidiaDeviceCountUnavailable],
            };
        }
    };

    let mut candidates = Vec::with_capacity((device_count as usize).min(MAX_DEVICE_GPUS));
    for nvml_index in 0..device_count {
        let device = match nvml.device_by_index(nvml_index) {
            Ok(device) => device,
            Err(_) => {
                error_codes.push(DeviceProbeErrorCode::NvidiaDeviceUnavailable);
                continue;
            }
        };
        let memory = match device.memory_info() {
            Ok(memory) if memory.total > 0 => memory,
            _ => {
                error_codes.push(DeviceProbeErrorCode::NvidiaDeviceMemoryUnavailable);
                continue;
            }
        };
        let pci_sort_key = device.pci_info().map_or_else(
            |_| {
                error_codes.push(DeviceProbeErrorCode::NvidiaPciInfoUnavailable);
                None
            },
            |pci| {
                Some(PciSortKey {
                    domain: pci.domain,
                    bus: pci.bus,
                    device: pci.device,
                })
            },
        );
        candidates.push(ResourceGpuCandidate {
            nvml_index,
            pci_sort_key,
            vram_free_bytes: api_int64(memory.free.min(memory.total)),
        });
    }

    sort_gpu_candidates(&mut candidates, |candidate| {
        (candidate.pci_sort_key, candidate.nvml_index)
    });
    if candidates.len() > MAX_DEVICE_GPUS {
        candidates.truncate(MAX_DEVICE_GPUS);
        error_codes.push(DeviceProbeErrorCode::NvidiaDeviceLimitExceeded);
    }
    let gpus = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| DeviceGpuResource {
            id: format!("gpu_{index}"),
            vram_free_bytes: candidate.vram_free_bytes,
        })
        .collect();
    normalize_error_codes(&mut error_codes);

    DeviceResourceNvidiaProbe { gpus, error_codes }
}

fn sort_gpu_candidates<T, F>(candidates: &mut [T], key: F)
where
    F: Fn(&T) -> (Option<PciSortKey>, u32),
{
    candidates.sort_by(|left, right| {
        let (left_pci, left_index) = key(left);
        let (right_pci, right_index) = key(right);
        match (left_pci, right_pci) {
            (Some(left), Some(right)) => left.cmp(&right).then(left_index.cmp(&right_index)),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left_index.cmp(&right_index),
        }
    });
}

fn probe_model_disk(model_directory: &Path) -> Result<DiskObservation, DeviceProbeError> {
    let absolute_model_directory = absolute_path(model_directory)?;
    let disks = Disks::new_with_refreshed_list();
    let observations = disks
        .list()
        .iter()
        .filter(|disk| disk.total_space() > 0)
        .map(|disk| DiskObservation {
            mount_point: disk.mount_point().to_path_buf(),
            file_system: optional_truncated(&disk.file_system().to_string_lossy(), 40),
            total_bytes: disk.total_space(),
            free_bytes: disk.available_space(),
        })
        .collect::<Vec<_>>();

    select_model_disk(&absolute_model_directory, observations.iter())
        .cloned()
        .ok_or(DeviceProbeError::ModelDiskUnavailable)
}

fn select_model_disk<'a>(
    model_directory: &Path,
    disks: impl Iterator<Item = &'a DiskObservation>,
) -> Option<&'a DiskObservation> {
    disks
        .filter(|disk| path_is_within(model_directory, &disk.mount_point))
        .max_by_key(|disk| normalized_path(&disk.mount_point).len())
}

fn absolute_path(path: &Path) -> Result<PathBuf, DeviceProbeError> {
    let absolute = if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|_| DeviceProbeError::ModelDirectoryUnavailable)
    }?;

    Ok(std::fs::canonicalize(&absolute).unwrap_or(absolute))
}

fn path_is_within(path: &Path, mount_point: &Path) -> bool {
    let path = normalized_path(path);
    let mount = normalized_path(mount_point);
    if path == mount {
        return true;
    }

    let prefix = if mount.ends_with(std::path::MAIN_SEPARATOR) {
        mount
    } else {
        format!("{mount}{}", std::path::MAIN_SEPARATOR)
    };
    path.starts_with(&prefix)
}

fn normalized_path(path: &Path) -> String {
    let value = path.to_string_lossy().replace(
        if std::path::MAIN_SEPARATOR == '/' {
            '\\'
        } else {
            '/'
        },
        std::path::MAIN_SEPARATOR_STR,
    );
    let trimmed = value.trim_end_matches(std::path::MAIN_SEPARATOR);
    let normalized = if trimmed.is_empty() { &value } else { trimmed };
    if cfg!(windows) {
        let normalized = normalized.to_ascii_lowercase();
        if let Some(path) = normalized.strip_prefix(r"\\?\unc\") {
            format!(r"\\{path}")
        } else if let Some(path) = normalized.strip_prefix(r"\\?\") {
            path.to_owned()
        } else {
            normalized
        }
    } else {
        normalized.to_owned()
    }
}

fn device_operating_system(os: &OsInfo) -> Result<DeviceOperatingSystem, DeviceProbeError> {
    let family = match std::env::consts::OS {
        "windows" => DeviceOperatingSystemFamily::Windows,
        "linux" => DeviceOperatingSystemFamily::Linux,
        "macos" => DeviceOperatingSystemFamily::Macos,
        _ => return Err(DeviceProbeError::UnsupportedOperatingSystem),
    };
    let architecture = match std::env::consts::ARCH {
        "x86_64" => DeviceArchitecture::X86_64,
        "aarch64" => DeviceArchitecture::Arm64,
        _ => return Err(DeviceProbeError::UnsupportedArchitecture),
    };
    let distribution = if family == DeviceOperatingSystemFamily::Linux {
        optional_truncated(&System::distribution_id(), 64)
    } else {
        None
    };

    Ok(DeviceOperatingSystem {
        family,
        distribution,
        version: nonempty_truncated(&os.version, "unknown", 120),
        architecture,
    })
}

fn map_system_warning(code: HardwareWarningCode) -> Option<DeviceProbeErrorCode> {
    match code {
        HardwareWarningCode::PhysicalCoreCountUnavailable => {
            Some(DeviceProbeErrorCode::PhysicalCoreCountUnavailable)
        }
        _ => None,
    }
}

fn probe_status(error_codes: &[DeviceProbeErrorCode]) -> DeviceProbeStatus {
    if error_codes.is_empty() {
        DeviceProbeStatus::Verified
    } else if error_codes.iter().any(|code| {
        matches!(
            code,
            DeviceProbeErrorCode::NvmlUnavailable
                | DeviceProbeErrorCode::NvidiaDeviceCountUnavailable
        )
    }) {
        DeviceProbeStatus::Unverified
    } else {
        DeviceProbeStatus::Partial
    }
}

fn nvml_unavailable_errors(presence: NvidiaAdapterPresence) -> Vec<DeviceProbeErrorCode> {
    match presence {
        NvidiaAdapterPresence::Absent => Vec::new(),
        NvidiaAdapterPresence::Present | NvidiaAdapterPresence::Unknown => {
            vec![DeviceProbeErrorCode::NvmlUnavailable]
        }
    }
}

#[cfg(target_os = "windows")]
fn detect_nvidia_adapter_presence() -> NvidiaAdapterPresence {
    #[allow(non_snake_case)]
    #[repr(C)]
    struct DisplayDeviceW {
        cb: u32,
        DeviceName: [u16; 32],
        DeviceString: [u16; 128],
        StateFlags: u32,
        DeviceID: [u16; 128],
        DeviceKey: [u16; 128],
    }

    #[link(name = "user32")]
    extern "system" {
        fn EnumDisplayDevicesW(
            device: *const u16,
            device_number: u32,
            display_device: *mut DisplayDeviceW,
            flags: u32,
        ) -> i32;
    }

    fn wide_text(value: &[u16]) -> String {
        let length = value
            .iter()
            .position(|code| *code == 0)
            .unwrap_or(value.len());
        String::from_utf16_lossy(&value[..length])
    }

    let mut found_adapter = false;
    for index in 0..256 {
        let mut adapter = DisplayDeviceW {
            cb: std::mem::size_of::<DisplayDeviceW>() as u32,
            DeviceName: [0; 32],
            DeviceString: [0; 128],
            StateFlags: 0,
            DeviceID: [0; 128],
            DeviceKey: [0; 128],
        };
        // SAFETY: `adapter` has the documented DISPLAY_DEVICEW layout and remains
        // writable for the duration of this synchronous Win32 call.
        let present = unsafe { EnumDisplayDevicesW(std::ptr::null(), index, &mut adapter, 0) != 0 };
        if !present {
            break;
        }
        found_adapter = true;
        let device_id = wide_text(&adapter.DeviceID).to_ascii_uppercase();
        let display_name = wide_text(&adapter.DeviceString).to_ascii_uppercase();
        if device_id.contains("VEN_10DE") || display_name.contains("NVIDIA") {
            return NvidiaAdapterPresence::Present;
        }
    }

    if found_adapter {
        NvidiaAdapterPresence::Absent
    } else {
        NvidiaAdapterPresence::Unknown
    }
}

#[cfg(target_os = "linux")]
fn detect_nvidia_adapter_presence() -> NvidiaAdapterPresence {
    linux_nvidia_adapter_presence(
        Path::new("/sys/bus/pci/devices"),
        Path::new("/dev/dxg").exists(),
    )
}

#[cfg(target_os = "linux")]
fn linux_nvidia_adapter_presence(pci_directory: &Path, virtual_gpu: bool) -> NvidiaAdapterPresence {
    let Ok(entries) = std::fs::read_dir(pci_directory) else {
        return NvidiaAdapterPresence::Unknown;
    };
    let mut observed_device = false;
    let mut incomplete = false;
    for entry in entries {
        let Ok(entry) = entry else {
            incomplete = true;
            continue;
        };
        let vendor = std::fs::read_to_string(entry.path().join("vendor"))
            .ok()
            .and_then(|value| u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok());
        let class = std::fs::read_to_string(entry.path().join("class"))
            .ok()
            .and_then(|value| u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok());
        let (Some(vendor), Some(class)) = (vendor, class) else {
            incomplete = true;
            continue;
        };
        observed_device = true;
        // Display controllers and processing accelerators; NVIDIA networking
        // hardware alone is not evidence that a CUDA GPU is present.
        if vendor == 0x10de && matches!(class >> 16, 0x03 | 0x12) {
            return NvidiaAdapterPresence::Present;
        }
    }
    // WSL may expose a GPU through dxg instead of the guest PCI inventory.
    if incomplete || virtual_gpu || !observed_device {
        NvidiaAdapterPresence::Unknown
    } else {
        NvidiaAdapterPresence::Absent
    }
}

#[cfg(all(test, target_os = "linux"))]
mod linux_adapter_tests {
    use super::*;

    fn pci_device(root: &Path, name: &str, vendor: &str, class: &str) {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("vendor"), vendor).unwrap();
        std::fs::write(directory.join("class"), class).unwrap();
    }

    #[test]
    fn nvidia_networking_is_not_mistaken_for_a_compute_gpu() {
        let directory = tempfile::tempdir().unwrap();
        pci_device(directory.path(), "network", "0x10de\n", "0x020000\n");
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), false),
            NvidiaAdapterPresence::Absent
        );
        pci_device(directory.path(), "display", "0x10de\n", "0x030000\n");
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), false),
            NvidiaAdapterPresence::Present
        );
    }

    #[test]
    fn empty_incomplete_and_virtual_inventories_remain_unknown() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), false),
            NvidiaAdapterPresence::Unknown
        );
        pci_device(directory.path(), "host", "0x8086\n", "0x060000\n");
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), false),
            NvidiaAdapterPresence::Absent
        );
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), true),
            NvidiaAdapterPresence::Unknown
        );
        pci_device(directory.path(), "incomplete", "invalid", "0x030000");
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), false),
            NvidiaAdapterPresence::Unknown
        );
    }

    #[test]
    fn nvidia_processing_accelerators_are_detected() {
        let directory = tempfile::tempdir().unwrap();
        pci_device(directory.path(), "accelerator", "0x10de\n", "0x120000\n");
        assert_eq!(
            linux_nvidia_adapter_presence(directory.path(), false),
            NvidiaAdapterPresence::Present
        );
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn detect_nvidia_adapter_presence() -> NvidiaAdapterPresence {
    NvidiaAdapterPresence::Unknown
}

fn normalize_error_codes(error_codes: &mut Vec<DeviceProbeErrorCode>) {
    let mut seen = HashSet::new();
    error_codes.retain(|code| seen.insert(*code));
    error_codes.truncate(MAX_DEVICE_ERROR_CODES);
}

fn collect_unique_totals<'a>(
    values: impl Iterator<Item = (&'a str, u64)>,
    duplicate_error: DeviceSnapshotValidationError,
) -> Result<HashMap<&'a str, u64>, DeviceSnapshotValidationError> {
    let mut result = HashMap::new();
    for (id, value) in values {
        if result.insert(id, value).is_some() {
            return Err(duplicate_error);
        }
    }
    Ok(result)
}

fn api_int64(value: u64) -> u64 {
    value.min(i64::MAX as u64)
}

fn optional_truncated(value: &str, max_chars: usize) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| truncate_chars(value, max_chars))
}

fn nonempty_truncated(value: &str, fallback: &str, max_chars: usize) -> String {
    optional_truncated(value, max_chars).unwrap_or_else(|| fallback.to_owned())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn assemble_snapshot(
    captured_at_unix_ms: u64,
    system: SystemFacts,
    nvidia: NvidiaProbeResult,
) -> HardwareSnapshotV1 {
    let mut seen = HashSet::new();
    let warnings = system
        .warnings
        .into_iter()
        .chain(nvidia.warnings)
        .filter(|code| seen.insert(*code))
        .map(|code| HardwareWarning { code })
        .collect();

    HardwareSnapshotV1 {
        schema_version: HARDWARE_SCHEMA_VERSION,
        captured_at_unix_ms,
        os: system.os,
        cpu: system.cpu,
        memory: system.memory,
        gpus: nvidia.gpus,
        warnings,
    }
}

fn format_cuda_driver_version(version: i32) -> String {
    format!(
        "{}.{}",
        cuda_driver_version_major(version),
        cuda_driver_version_minor(version)
    )
}

fn nonempty_or(value: String, fallback: &str) -> String {
    if value.trim().is_empty() {
        fallback.to_owned()
    } else {
        value
    }
}

fn unix_timestamp_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

fn usize_to_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_system(vendor_id: &str, warnings: Vec<HardwareWarningCode>) -> SystemFacts {
        SystemFacts {
            os: OsInfo {
                family: "Windows".to_owned(),
                version: "11".to_owned(),
                arch: "x86_64".to_owned(),
            },
            cpu: CpuInfo {
                vendor: classify_cpu_vendor(vendor_id),
                vendor_id: vendor_id.to_owned(),
                model: "Test CPU".to_owned(),
                physical_cores: 8,
                logical_cores: 16,
                frequency_mhz: 4_200,
                features: vec!["avx2".to_owned()],
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 8 * 1024 * 1024 * 1024,
            },
            warnings,
        }
    }

    fn sample_device_capability() -> DeviceCapabilitySnapshot {
        DeviceCapabilitySnapshot {
            schema_version: DEVICE_CAPABILITY_SCHEMA_VERSION,
            captured_at_unix_ms: 100,
            probe_status: DeviceProbeStatus::Verified,
            operating_system: DeviceOperatingSystem {
                family: DeviceOperatingSystemFamily::Windows,
                distribution: None,
                version: "11".to_owned(),
                architecture: DeviceArchitecture::X86_64,
            },
            cpu: DeviceCpuCapability {
                vendor: Some("GenuineIntel".to_owned()),
                model: "Test CPU".to_owned(),
                physical_cores: 8,
                logical_cores: 16,
                features: vec!["AVX2".to_owned()],
            },
            memory: DeviceMemoryCapability { total_bytes: 64 },
            gpus: vec![DeviceGpuCapability {
                id: "gpu_0".to_owned(),
                vendor: "NVIDIA".to_owned(),
                name: "Test GPU".to_owned(),
                vram_total_bytes: 16,
                backend: DeviceGpuBackend::Cuda,
                runtime: Some("CUDA".to_owned()),
                runtime_version: Some("12.8".to_owned()),
                driver_version: Some("570.86".to_owned()),
                compute_capability: Some("8.9".to_owned()),
                verified: true,
            }],
            disks: vec![DeviceDiskCapability {
                id: MODEL_DISK_ID.to_owned(),
                label: "模型盘".to_owned(),
                file_system: Some("NTFS".to_owned()),
                total_bytes: 128,
            }],
            error_codes: Vec::new(),
        }
    }

    fn sample_device_resources() -> DeviceResourceSnapshot {
        DeviceResourceSnapshot {
            captured_at_unix_ms: 101,
            memory_available_bytes: 32,
            gpus: vec![DeviceGpuResource {
                id: "gpu_0".to_owned(),
                vram_free_bytes: 8,
            }],
            disks: vec![DeviceDiskResource {
                id: MODEL_DISK_ID.to_owned(),
                free_bytes: 64,
            }],
            probe_status: DeviceProbeStatus::Verified,
            error_codes: Vec::new(),
        }
    }

    #[test]
    fn classifies_intel_vendor_ids() {
        assert_eq!(classify_cpu_vendor("GenuineIntel"), CpuVendor::Intel);
        assert_eq!(classify_cpu_vendor(" Intel Corporation "), CpuVendor::Intel);
    }

    #[test]
    fn classifies_amd_vendor_ids() {
        assert_eq!(classify_cpu_vendor("AuthenticAMD"), CpuVendor::Amd);
        assert_eq!(classify_cpu_vendor("AMD"), CpuVendor::Amd);
    }

    #[test]
    fn classifies_unknown_vendor_ids_without_guessing() {
        assert_eq!(classify_cpu_vendor("CentaurHauls"), CpuVendor::Other);
        assert_eq!(classify_cpu_vendor(""), CpuVendor::Other);
    }

    #[test]
    fn nvml_failure_is_a_warning_not_a_snapshot_error() {
        let snapshot = assemble_snapshot(
            123,
            sample_system("AuthenticAMD", Vec::new()),
            NvidiaProbeResult {
                gpus: Vec::new(),
                warnings: vec![HardwareWarningCode::NvmlUnavailable],
            },
        );

        assert_eq!(snapshot.captured_at_unix_ms, 123);
        assert_eq!(snapshot.cpu.vendor, CpuVendor::Amd);
        assert!(snapshot.gpus.is_empty());
        assert_eq!(
            snapshot.warnings,
            vec![HardwareWarning {
                code: HardwareWarningCode::NvmlUnavailable
            }]
        );
    }

    #[test]
    fn duplicate_warning_codes_are_collapsed() {
        let snapshot = assemble_snapshot(
            123,
            sample_system(
                "GenuineIntel",
                vec![HardwareWarningCode::PhysicalCoreCountUnavailable],
            ),
            NvidiaProbeResult {
                gpus: Vec::new(),
                warnings: vec![
                    HardwareWarningCode::NvidiaDeviceUnavailable,
                    HardwareWarningCode::NvidiaDeviceUnavailable,
                ],
            },
        );

        assert_eq!(snapshot.warnings.len(), 2);
    }

    #[test]
    fn serializes_the_versioned_snake_case_contract() {
        let snapshot = assemble_snapshot(
            123,
            sample_system("GenuineIntel", Vec::new()),
            NvidiaProbeResult {
                gpus: Vec::new(),
                warnings: vec![HardwareWarningCode::NvmlUnavailable],
            },
        );
        let json = serde_json::to_value(snapshot).expect("snapshot should serialize");

        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["cpu"]["vendor"], "intel");
        assert_eq!(json["warnings"][0]["code"], "nvml_unavailable");
    }

    #[test]
    fn formats_nvml_cuda_versions_for_display() {
        assert_eq!(format_cuda_driver_version(12_080), "12.8");
        assert_eq!(format_cuda_driver_version(11_040), "11.4");
    }

    #[test]
    fn serializes_optional_gpu_telemetry_and_warning_codes() {
        let snapshot = assemble_snapshot(
            123,
            sample_system("GenuineIntel", Vec::new()),
            NvidiaProbeResult {
                gpus: vec![GpuInfo {
                    id: "nvidia:0".to_owned(),
                    uuid: Some("GPU-test-uuid".to_owned()),
                    index: 0,
                    vendor: GpuVendor::Nvidia,
                    name: "NVIDIA Test GPU".to_owned(),
                    vram_total_bytes: 8 * 1024 * 1024 * 1024,
                    vram_free_bytes: 6 * 1024 * 1024 * 1024,
                    utilization_percent: Some(38),
                    temperature_celsius: None,
                    driver_version: Some("576.80".to_owned()),
                    cuda_driver_version: Some("12.9".to_owned()),
                }],
                warnings: vec![
                    HardwareWarningCode::NvidiaDeviceUtilizationUnavailable,
                    HardwareWarningCode::NvidiaDeviceTemperatureUnavailable,
                ],
            },
        );
        let json = serde_json::to_value(snapshot).expect("snapshot should serialize");

        assert_eq!(json["gpus"][0]["utilization_percent"], 38);
        assert!(json["gpus"][0]["temperature_celsius"].is_null());
        assert_eq!(
            json["warnings"][0]["code"],
            "nvidia_device_utilization_unavailable"
        );
        assert_eq!(
            json["warnings"][1]["code"],
            "nvidia_device_temperature_unavailable"
        );
    }

    #[test]
    fn timestamps_before_unix_epoch_are_safely_clamped() {
        assert_eq!(
            unix_timestamp_ms(UNIX_EPOCH - std::time::Duration::from_millis(1)),
            0
        );
    }

    #[test]
    fn serializes_platform_probe_enums_with_contract_casing() {
        assert_eq!(
            serde_json::to_value(DeviceProbeStatus::Unverified).unwrap(),
            "UNVERIFIED"
        );
        assert_eq!(
            serde_json::to_value(DeviceProbeErrorCode::NvidiaPciInfoUnavailable).unwrap(),
            "NVIDIA_PCI_INFO_UNAVAILABLE"
        );
        assert_eq!(
            DeviceProbeErrorCode::NvidiaPciInfoUnavailable.to_string(),
            "NVIDIA_PCI_INFO_UNAVAILABLE"
        );
        assert_eq!(
            serde_json::to_value(DeviceOperatingSystemFamily::Windows).unwrap(),
            "windows"
        );
        assert_eq!(
            serde_json::to_value(DeviceArchitecture::X86_64).unwrap(),
            "x86_64"
        );
        assert_eq!(
            serde_json::to_value(DeviceGpuBackend::Cuda).unwrap(),
            "CUDA"
        );
    }

    #[test]
    fn gpu_ids_follow_pci_order_and_do_not_expose_pci_data() {
        let mut candidates = vec![
            StaticGpuCandidate {
                nvml_index: 0,
                pci_sort_key: Some(PciSortKey {
                    domain: 0,
                    bus: 8,
                    device: 0,
                }),
                name: "GPU at bus 8".to_owned(),
                vram_total_bytes: 8,
                driver_version: None,
                runtime_version: None,
                compute_capability: None,
                verified: false,
            },
            StaticGpuCandidate {
                nvml_index: 1,
                pci_sort_key: Some(PciSortKey {
                    domain: 0,
                    bus: 2,
                    device: 0,
                }),
                name: "GPU at bus 2".to_owned(),
                vram_total_bytes: 8,
                driver_version: None,
                runtime_version: None,
                compute_capability: None,
                verified: false,
            },
            StaticGpuCandidate {
                nvml_index: 2,
                pci_sort_key: None,
                name: "GPU without PCI".to_owned(),
                vram_total_bytes: 8,
                driver_version: None,
                runtime_version: None,
                compute_capability: None,
                verified: false,
            },
        ];

        sort_gpu_candidates(&mut candidates, |candidate| {
            (candidate.pci_sort_key, candidate.nvml_index)
        });
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.name.as_str())
                .collect::<Vec<_>>(),
            vec!["GPU at bus 2", "GPU at bus 8", "GPU without PCI"]
        );

        let serialized = serde_json::to_string(&sample_device_capability()).unwrap();
        assert!(!serialized.contains("pci"));
        assert!(!serialized.contains("bus"));
        assert!(!serialized.contains("temperature"));
        assert!(!serialized.contains("utilization"));
    }

    #[test]
    fn model_disk_selection_prefers_the_deepest_containing_mount() {
        let root = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
        let models_mount = root.join("var").join("compute-models");
        let model_directory = models_mount.join("cache");
        let observations = [
            DiskObservation {
                mount_point: root,
                file_system: Some("rootfs".to_owned()),
                total_bytes: 100,
                free_bytes: 50,
            },
            DiskObservation {
                mount_point: models_mount.clone(),
                file_system: Some("modelsfs".to_owned()),
                total_bytes: 200,
                free_bytes: 150,
            },
        ];

        let selected = select_model_disk(&model_directory, observations.iter()).unwrap();
        assert_eq!(selected.mount_point, models_mount);
        assert_eq!(selected.total_bytes, 200);
    }

    #[test]
    fn validates_matching_resource_ids_and_bounds() {
        assert_eq!(
            validate_resource_snapshot(&sample_device_capability(), &sample_device_resources()),
            Ok(())
        );
    }

    #[test]
    fn rejects_resources_that_exceed_static_totals() {
        let capability = sample_device_capability();
        let mut resources = sample_device_resources();
        resources.memory_available_bytes = 65;
        assert_eq!(
            validate_resource_snapshot(&capability, &resources),
            Err(DeviceSnapshotValidationError::MemoryAvailableExceedsTotal)
        );

        resources.memory_available_bytes = 32;
        resources.gpus[0].vram_free_bytes = 17;
        assert_eq!(
            validate_resource_snapshot(&capability, &resources),
            Err(DeviceSnapshotValidationError::GpuFreeExceedsTotal(
                "gpu_0".to_owned()
            ))
        );

        resources.gpus[0].vram_free_bytes = 8;
        resources.disks[0].free_bytes = 129;
        assert_eq!(
            validate_resource_snapshot(&capability, &resources),
            Err(DeviceSnapshotValidationError::DiskFreeExceedsTotal(
                MODEL_DISK_ID.to_owned()
            ))
        );
    }

    #[test]
    fn rejects_resource_id_drift_and_duplicates() {
        let capability = sample_device_capability();
        let mut resources = sample_device_resources();
        resources.gpus[0].id = "gpu_1".to_owned();
        assert_eq!(
            validate_resource_snapshot(&capability, &resources),
            Err(DeviceSnapshotValidationError::GpuIdsMismatch)
        );

        let mut resources = sample_device_resources();
        resources.gpus.push(resources.gpus[0].clone());
        assert_eq!(
            validate_resource_snapshot(&capability, &resources),
            Err(DeviceSnapshotValidationError::DuplicateResourceGpuId)
        );

        let mut resources = sample_device_resources();
        resources.disks[0].id = "disk_other".to_owned();
        assert_eq!(
            validate_resource_snapshot(&capability, &resources),
            Err(DeviceSnapshotValidationError::DiskIdsMismatch)
        );
    }

    #[test]
    fn enforces_device_api_collection_boundaries() {
        let mut capability = sample_device_capability();
        capability.gpus = (0..=MAX_DEVICE_GPUS)
            .map(|index| DeviceGpuCapability {
                id: format!("gpu_{index}"),
                ..capability.gpus[0].clone()
            })
            .collect();
        assert_eq!(
            validate_resource_snapshot(&capability, &sample_device_resources()),
            Err(DeviceSnapshotValidationError::CapabilityGpuLimitExceeded)
        );

        let mut capability = sample_device_capability();
        capability.disks.clear();
        assert_eq!(
            validate_resource_snapshot(&capability, &sample_device_resources()),
            Err(DeviceSnapshotValidationError::CapabilityDiskCountInvalid)
        );
    }

    #[test]
    fn maps_probe_errors_to_conservative_statuses() {
        assert_eq!(probe_status(&[]), DeviceProbeStatus::Verified);
        assert_eq!(
            probe_status(&[DeviceProbeErrorCode::NvidiaPciInfoUnavailable]),
            DeviceProbeStatus::Partial
        );
        assert_eq!(
            probe_status(&[DeviceProbeErrorCode::NvmlUnavailable]),
            DeviceProbeStatus::Unverified
        );
    }

    #[test]
    fn cpu_only_hosts_do_not_report_nvml_as_a_capability_failure() {
        let errors = nvml_unavailable_errors(NvidiaAdapterPresence::Absent);
        assert!(errors.is_empty());
        assert_eq!(probe_status(&errors), DeviceProbeStatus::Verified);
    }

    #[test]
    fn nvidia_or_unknown_adapter_presence_keeps_nvml_failure_conservative() {
        for presence in [
            NvidiaAdapterPresence::Present,
            NvidiaAdapterPresence::Unknown,
        ] {
            let errors = nvml_unavailable_errors(presence);
            assert_eq!(errors, vec![DeviceProbeErrorCode::NvmlUnavailable]);
            assert_eq!(probe_status(&errors), DeviceProbeStatus::Unverified);
        }
    }

    #[test]
    fn live_device_probes_use_matching_platform_ids() {
        let model_directory = std::env::current_dir().expect("working directory should exist");
        let capability = probe_static_capability(&model_directory)
            .expect("host capability should be discoverable");
        let resources = probe_dynamic_resources(&model_directory)
            .expect("host resources should be discoverable");

        assert_eq!(capability.schema_version, DEVICE_CAPABILITY_SCHEMA_VERSION);
        assert_eq!(capability.disks[0].id, MODEL_DISK_ID);
        assert_eq!(resources.disks[0].id, MODEL_DISK_ID);
        validate_resource_snapshot(&capability, &resources)
            .expect("back-to-back probes should share ids and resource boundaries");
    }

    #[test]
    fn live_probe_returns_real_cpu_and_memory() {
        let snapshot = probe_hardware().expect("host CPU and memory should be discoverable");

        eprintln!(
            "hardware smoke test: cpu={} vendor={:?} memory={} gpus={:?} warnings={:?}",
            snapshot.cpu.model,
            snapshot.cpu.vendor,
            snapshot.memory.total_bytes,
            snapshot
                .gpus
                .iter()
                .map(|gpu| (
                    &gpu.name,
                    gpu.vram_total_bytes,
                    gpu.utilization_percent,
                    gpu.temperature_celsius,
                ))
                .collect::<Vec<_>>(),
            snapshot.warnings
        );
        assert_eq!(snapshot.schema_version, HARDWARE_SCHEMA_VERSION);
        assert!(!snapshot.cpu.model.trim().is_empty());
        assert!(snapshot.cpu.logical_cores > 0);
        assert!(snapshot.memory.total_bytes > 0);
        assert!(snapshot.memory.available_bytes <= snapshot.memory.total_bytes);
    }
}
