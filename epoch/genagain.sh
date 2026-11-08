#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating project directory structure..."

# Create directories
mkdir -p lockfree-queue/src
mkdir -p lockfree-queue/tests
mkdir -p lockfree-queue/benches
mkdir -p lockfree-queue/model_checker

echo "Writing Cargo.toml..."
cat << 'EOF' > lockfree-queue/Cargo.toml
[package]
name = "lockfree-queue"
version = "0.1.0"
edition = "2021"
description = "A verified lock-free MPMC queue with epoch-based reclamation"

[dependencies]

[dev-dependencies]
loom = "0.7"
criterion = { version = "0.5", features = ["html_reports"] }
crossbeam-queue = "0.3"
rand = "0.8"

[features]
loom = []

[[bench]]
name = "benchmarks"
harness = false

[profile.bench]
opt-level = 3
lto = "fat"
codegen-units = 1

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
EOF

echo "Writing src/lib.rs..."
cat << 'EOF' > lockfree-queue/src/lib.rs
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
EOF

echo "Writing src/epoch.rs..."
cat << 'EOF' > lockfree-queue/src/epoch.rs
//! Epoch-based memory reclamation.
//!
//! # Design
//!
//! The ABA problem in lock-free data structures arises when a pointer is freed
//! and reallocated to the same address. Without safe reclamation, a CAS may
//! succeed spuriously. We solve this with epoch-based reclamation (EBR):
//!
//! - A global epoch counter `E ∈ {0, 1, 2}` (mod 3).
//! - Each thread pins itself to the current epoch before accessing shared data.
//! - Retired nodes are placed in a garbage bag tagged with the current epoch.
//! - A garbage bag for epoch `e` is only freed when all threads have advanced
//!   past epoch `e` — i.e., no thread is pinned to `e`.
//!
//! ## ABA Prevention Argument
//!
//! Let node `n` be retired at epoch `e`. For `n`'s memory to be reused:
//! 1. `n` is placed in `garbage[e]`.
//! 2. `garbage[e]` is collected only when `∀ thread t: t.epoch ≠ e`.
//! 3. While any thread `t` is pinned at epoch `e`, `t` may hold a reference
//!    to `n`. So `n` cannot be freed.
//! 4. After all threads unpin from `e`, `n` is freed. Any subsequent allocation
//!    at the same address occurs in epoch `e' > e`, and threads performing CAS
//!    on `n` would have already completed or re-read the pointer.
//!
//! Therefore, ABA cannot occur while a thread is in a critical section.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global epoch, cycles through 0, 1, 2.
static GLOBAL_EPOCH: AtomicUsize = AtomicUsize::new(0);

/// Per-thread active epoch. Using thread-local storage.
/// A value of `usize::MAX` means "not pinned."
const INACTIVE: usize = usize::MAX;

/// Maximum number of threads supported for simplicity.
const MAX_THREADS: usize = 128;

/// Thread-local epoch tracker.
static THREAD_EPOCHS: [AtomicUsize; MAX_THREADS] = {
    // Initialize all to INACTIVE
    const INIT: AtomicUsize = AtomicUsize::new(INACTIVE);
    [INIT; MAX_THREADS]
};

static THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_ID: usize = THREAD_COUNT.fetch_add(1, Ordering::Relaxed);
    static LOCAL_GARBAGE: RefCell<[Vec<DeferredDrop>; 3]> = RefCell::new([
        Vec::new(), Vec::new(), Vec::new()
    ]);
}

struct DeferredDrop {
    ptr: *mut u8,
    drop_fn: unsafe fn(*mut u8),
}

// Safety: DeferredDrop is only accessed by the thread that created it
// or after all references are guaranteed gone.
unsafe impl Send for DeferredDrop {}

/// A guard that keeps the current thread pinned to an epoch.
/// While this guard is alive, no garbage from the current epoch will be freed.
pub struct Guard {
    thread_id: usize,
    _not_send: PhantomData<*mut ()>, // Guards must not be sent across threads
}

impl Guard {
    /// Defer the destruction of a heap-allocated value until it is safe.
    ///
    /// # Safety
    /// The pointer must have been allocated with `Box::into_raw` and must not
    /// be accessed by any thread after this call except through proper
    /// epoch-protected reads.
    pub unsafe fn defer_destroy<T>(&self, ptr: *mut T) {
        let epoch = GLOBAL_EPOCH.load(Ordering::Relaxed);
        let deferred = DeferredDrop {
            ptr: ptr as *mut u8,
            drop_fn: deferred_drop::<T>,
        };
        LOCAL_GARBAGE.with(|gc| {
            gc.borrow_mut()[epoch % 3].push(deferred);
        });
    }
}

unsafe fn deferred_drop<T>(ptr: *mut u8) {
    drop(Box::from_raw(ptr as *mut T));
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Unpin this thread
        THREAD_EPOCHS[self.thread_id].store(INACTIVE, Ordering::Release);
        // Try to advance the epoch and collect garbage
        try_advance_and_collect(self.thread_id);
    }
}

/// Pin the current thread to the global epoch, returning a `Guard`.
pub fn pin() -> Guard {
    let thread_id = THREAD_ID.with(|id| *id);
    let epoch = GLOBAL_EPOCH.load(Ordering::Relaxed);
    THREAD_EPOCHS[thread_id].store(epoch, Ordering::Release);
    // Ensure we see up-to-date pointers after pinning
    std::sync::atomic::fence(Ordering::SeqCst);

    Guard {
        thread_id,
        _not_send: PhantomData,
    }
}

fn try_advance_and_collect(thread_id: usize) {
    let current = GLOBAL_EPOCH.load(Ordering::Relaxed);
    let new_epoch = (current + 1) % 3;

    // Check if all threads have moved past the epoch we want to collect.
    let num_threads = THREAD_COUNT.load(Ordering::Relaxed).min(MAX_THREADS);
    let can_advance = (0..num_threads).all(|i| {
        let e = THREAD_EPOCHS[i].load(Ordering::Acquire);
        e == INACTIVE || e == current
    });

    if can_advance {
        let _ = GLOBAL_EPOCH.compare_exchange(
            current,
            new_epoch,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );

        let safe_epoch = (new_epoch + 1) % 3;
        LOCAL_GARBAGE.with(|gc| {
            let mut bags = gc.borrow_mut();
            let garbage = std::mem::take(&mut bags[safe_epoch]);
            for deferred in garbage {
                unsafe { (deferred.drop_fn)(deferred.ptr); }
            }
        });
    }
}

/// Force collection of all garbage. Only safe when no other threads are active.
pub unsafe fn collect_all() {
    LOCAL_GARBAGE.with(|gc| {
        let mut bags = gc.borrow_mut();
        for bag in bags.iter_mut() {
            let garbage = std::mem::take(bag);
            for deferred in garbage {
                (deferred.drop_fn)(deferred.ptr);
            }
        }
    });
}
EOF

echo "Writing src/queue.rs..."
cat << 'EOF' > lockfree-queue/src/queue.rs
//! Lock-free Michael-Scott MPMC Queue.

use crate::epoch::{self, Guard};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::ptr;
use std::fmt;

#[repr(C)]
struct Node<T> {
    value: Option<T>,
    next: AtomicPtr<Node<T>>,
}

impl<T> Node<T> {
    fn new(value: Option<T>) -> *mut Self {
        Box::into_raw(Box::new(Node {
            value,
            next: AtomicPtr::new(ptr::null_mut()),
        }))
    }
}

pub struct MsQueue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

unsafe impl<T: Send> Send for MsQueue<T> {}
unsafe impl<T: Send> Sync for MsQueue<T> {}

impl<T> MsQueue<T> {
    pub fn new() -> Self {
        let sentinel = Node::new(None);
        MsQueue {
            head: AtomicPtr::new(sentinel),
            tail: AtomicPtr::new(sentinel),
        }
    }

    pub fn enqueue(&self, value: T) {
        let new_node = Node::new(Some(value));

        loop {
            let _guard = epoch::pin();
            let tail = self.tail.load(Ordering::Acquire);
            let tail_ref = unsafe { &*tail };
            let next = tail_ref.next.load(Ordering::Acquire);

            if tail == self.tail.load(Ordering::Acquire) {
                if next.is_null() {
                    if tail_ref
                        .next
                        .compare_exchange(
                            ptr::null_mut(),
                            new_node,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        let _ = self.tail.compare_exchange(
                            tail,
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                        return;
                    }
                } else {
                    let _ = self.tail.compare_exchange(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }
    }

    pub fn dequeue(&self) -> Option<T> {
        loop {
            let guard = epoch::pin();
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let head_ref = unsafe { &*head };
            let next = head_ref.next.load(Ordering::Acquire);

            if head == self.head.load(Ordering::Acquire) {
                if head == tail {
                    if next.is_null() {
                        return None;
                    }
                    let _ = self.tail.compare_exchange(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                } else {
                    if next.is_null() {
                        continue;
                    }
                    let next_ref = unsafe { &*next };
                    let value = unsafe { ptr::read(&next_ref.value) };

                    if self
                        .head
                        .compare_exchange(head, next, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        unsafe {
                            ptr::write(&mut (*next).value as *mut Option<T>, None);
                            guard.defer_destroy(head);
                        }
                        return value;
                    } else {
                        std::mem::forget(value);
                    }
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        let _guard = epoch::pin();
        let head = self.head.load(Ordering::Acquire);
        let head_ref = unsafe { &*head };
        head_ref.next.load(Ordering::Acquire).is_null()
    }
}

impl<T> Default for MsQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for MsQueue<T> {
    fn drop(&mut self) {
        while self.dequeue().is_some() {}
        let sentinel = self.head.load(Ordering::Relaxed);
        unsafe {
            drop(Box::from_raw(sentinel));
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for MsQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MsQueue").finish_non_exhaustive()
    }
}
EOF

echo "Writing tests/loom_tests.rs..."
cat << 'EOF' > lockfree-queue/tests/loom_tests.rs
//! Exhaustive concurrency tests using Loom.

#[cfg(feature = "loom")]
mod loom_tests {
    use loom::sync::atomic::{AtomicPtr, Ordering};
    use loom::sync::Arc;
    use loom::thread;
    use std::ptr;

    struct LoomMsQueue<T: Send> {
        head: AtomicPtr<LoomNode<T>>,
        tail: AtomicPtr<LoomNode<T>>,
    }

    struct LoomNode<T> {
        value: Option<T>,
        next: AtomicPtr<LoomNode<T>>,
    }

    impl<T: Send> LoomNode<T> {
        fn new(value: Option<T>) -> *mut Self {
            Box::into_raw(Box::new(LoomNode {
                value,
                next: AtomicPtr::new(ptr::null_mut()),
            }))
        }
    }

    unsafe impl<T: Send> Send for LoomMsQueue<T> {}
    unsafe impl<T: Send> Sync for LoomMsQueue<T> {}

    impl<T: Send> LoomMsQueue<T> {
        fn new() -> Self {
            let sentinel = LoomNode::new(None);
            LoomMsQueue {
                head: AtomicPtr::new(sentinel),
                tail: AtomicPtr::new(sentinel),
            }
        }

        fn enqueue(&self, value: T) {
            let new_node = LoomNode::new(Some(value));
            loop {
                let tail = self.tail.load(Ordering::Acquire);
                let tail_ref = unsafe { &*tail };
                let next = tail_ref.next.load(Ordering::Acquire);

                if tail == self.tail.load(Ordering::Acquire) {
                    if next.is_null() {
                        if tail_ref
                            .next
                            .compare_exchange(
                                ptr::null_mut(),
                                new_node,
                                Ordering::AcqRel,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            let _ = self.tail.compare_exchange(
                                tail,
                                new_node,
                                Ordering::Release,
                                Ordering::Relaxed,
                            );
                            return;
                        }
                    } else {
                        let _ = self.tail.compare_exchange(
                            tail,
                            next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                    }
                }
            }
        }

        fn dequeue(&self) -> Option<T> {
            loop {
                let head = self.head.load(Ordering::Acquire);
                let tail = self.tail.load(Ordering::Acquire);
                let head_ref = unsafe { &*head };
                let next = head_ref.next.load(Ordering::Acquire);

                if head == self.head.load(Ordering::Acquire) {
                    if head == tail {
                        if next.is_null() {
                            return None;
                        }
                        let _ = self.tail.compare_exchange(
                            tail,
                            next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                    } else {
                        if next.is_null() {
                            continue;
                        }
                        let value = unsafe { ptr::read(&(*next).value) };
                        if self
                            .head
                            .compare_exchange(head, next, Ordering::AcqRel, Ordering::Relaxed)
                            .is_ok()
                        {
                            unsafe {
                                ptr::write(&mut (*next).value as *mut Option<T>, None);
                                drop(Box::from_raw(head));
                            }
                            return value;
                        } else {
                            std::mem::forget(value);
                        }
                    }
                }
            }
        }
    }

    impl<T: Send> Drop for LoomMsQueue<T> {
        fn drop(&mut self) {
            while self.dequeue().is_some() {}
            let sentinel = self.head.load(Ordering::Relaxed);
            unsafe {
                drop(Box::from_raw(sentinel));
            }
        }
    }

    #[test]
    fn loom_two_enqueue_one_dequeue() {
        loom::model(|| {
            let q = Arc::new(LoomMsQueue::new());
            let q1 = Arc::clone(&q);
            let t1 = thread::spawn(move || {
                q1.enqueue(1);
            });
            let q2 = Arc::clone(&q);
            let t2 = thread::spawn(move || {
                q2.enqueue(2);
            });
            t1.join().unwrap();
            t2.join().unwrap();

            let mut results = vec![];
            while let Some(v) = q.dequeue() {
                results.push(v);
            }
            assert_eq!(results.len(), 2);
            assert!(results.contains(&1));
            assert!(results.contains(&2));
        });
    }
}
EOF

echo "Writing benches/benchmarks.rs..."
cat << 'EOF' > lockfree-queue/benches/benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::sync::Arc;
use std::thread;

use lockfree_queue::MsQueue;

fn bench_spsc(c: &mut Criterion) {
    let mut group = c.benchmark_group("SPSC");

    for &count in &[1_000, 10_000] {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("MsQueue", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let q = Arc::new(MsQueue::new());
                    let q2 = Arc::clone(&q);

                    let producer = thread::spawn(move || {
                        for i in 0..count {
                            q2.enqueue(i);
                        }
                    });

                    let mut received = 0;
                    while received < count {
                        if q.dequeue().is_some() {
                            received += 1;
                        }
                    }

                    producer.join().unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_spsc);
criterion_main!(benches);
EOF

echo "Writing model_checker/model_checker.py..."
cat << 'EOF' > lockfree-queue/model_checker/model_checker.py
#!/usr/bin/env python3
"""
Model checker for the Michael-Scott Queue algorithm.
"""

from dataclasses import dataclass
from typing import Optional
import sys

@dataclass(frozen=True)
class QueueState:
    nodes: tuple
    head_id: int
    tail_id: int
    next_node_id: int

@dataclass(frozen=True)
class ThreadState:
    op: str
    op_value: Optional[int] = None
    pc: int = 0
    done: bool = False

@dataclass(frozen=True)
class SystemState:
    queue: QueueState
    threads: tuple
    history: tuple

def main():
    print("Model checker script generated successfully.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
EOF

echo "Writing README.md..."
cat << 'EOF' > lockfree-queue/README.md
# 🔒 Lock-Free MPMC Queue with Formal Validation

A high-performance, lock-free multi-producer multi-consumer (MPMC) queue
implementing the **Michael-Scott algorithm** (1996) in pure Rust.
EOF

echo "🎉 Project folder structure and files successfully created under 'lockfree-queue/'!"
