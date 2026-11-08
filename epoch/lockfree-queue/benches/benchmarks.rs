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
