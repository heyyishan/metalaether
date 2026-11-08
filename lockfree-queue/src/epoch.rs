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
