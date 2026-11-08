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
