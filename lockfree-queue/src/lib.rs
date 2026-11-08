//! # lockfree-queue
//!
//! A lock-free multi-producer multi-consumer (MPMC) queue based on the
//! Michael-Scott algorithm with epoch-based memory reclamation.
//!
//! ## Algorithm
//!
//! This implements the classic Michael & Scott (1996) two-lock / lock-free queue
//! adapted for Rust's ownership model. Memory reclamation uses a simplified
//! epoch-based scheme inspired by Keir Fraser's epoch-based reclamation and
//! crossbeam-epoch.
//!
//! ## Safety
//!
//! This crate uses `unsafe` for atomic pointer manipulation. Correctness is
//! validated through:
//! - Loom exhaustive interleaving tests
//! - A Python model checker for linearizability
//! - Stress tests with millions of operations
//! - Mathematical invariant documentation

pub mod queue;
pub mod epoch;

#[cfg(not(feature = "loom"))]
pub(crate) mod sync {
    pub use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering, fence};
    pub use std::sync::Arc;
    pub use std::thread;
}

#[cfg(feature = "loom")]
pub(crate) mod sync {
    pub use loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering, fence};
    pub use loom::sync::Arc;
    pub use loom::thread;
}

pub use queue::MsQueue;
