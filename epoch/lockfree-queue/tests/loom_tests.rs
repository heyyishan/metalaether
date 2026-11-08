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
