use std::sync::mpsc::channel;
use std::thread;
use std::thread::available_parallelism;
use threadpool::ThreadPool;

/// A library for parallel computation with a threshold for splitting work.

/// Processes a vector of data in parallel, splitting the work into threads if the input length exceeds a threshold.
/// There are two methods that allow you to solve the problem in parallel:
/// - `straightforward_parallel` is good where the process has all the system resources.
/// - `flexible_parallel` is less aggressive and better suited for situations where the system
///       is performing many tasks at once and you don't want to inconvenience them.
///
/// # Arguments
///
/// * `data` - A reference to a vector of data of type `T`.
/// * `f` - A function that takes an element of type `T` and returns a value of type `R`.
///
/// # Returns
///
/// A vector of results of type `R`, with the same length as the input data.
///
/// # Example
///
/// ```
/// use gt_test::straightforward_parallel;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let results = straightforward_parallel(&data, |x| x * 2);
/// assert_eq!(results, vec![2, 4, 6, 8, 10]);
/// ```
/// # Example
///
/// ```
/// use gt_test::flexible_parallel;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let results = flexible_parallel(&data, |x| x * 2);
/// assert_eq!(results, vec![2, 4, 6, 8, 10]);
/// ```

// Adjust the threshold
const THRESHOLD: usize = 128;

pub fn straightforward_parallel<T, R>(data: &[T], f: fn(t: T) -> R) -> Vec<R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(|item| f(item.clone())).collect()
    } else {
        // Split the work into threads
        let num_workers = available_parallelism().unwrap().get();
        let chunk_size = 1 + data.len() / num_workers;

        let mut handles = Vec::with_capacity(num_workers);
        let mut results = Vec::with_capacity(data.len());

        for chunk in data.chunks(chunk_size) {
            let chunk_data = chunk.to_vec();
            let handle = thread::spawn(move || chunk_data.into_iter().map(f).collect::<Vec<_>>());
            handles.push(handle);
        }

        // Collect results from threads
        for handle in handles {
            results.extend(handle.join().unwrap());
        }

        results
    }
}

pub fn flexible_parallel<T, R>(data: &[T], f: fn(t: T) -> R) -> Vec<R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(|item| f(item.clone())).collect()
    } else {
        let num_workers = available_parallelism().unwrap().get();
        let pool = ThreadPool::new(num_workers);

        let (tx, rx) = channel();
        for item in data {
            let tx = tx.clone();
            let val = item.clone();
            pool.execute(move || {
                let result = f(val);
                tx.send(result).expect("Channel will be open.");
            });
        }

        // Collect results from threads
        rx.iter().take(data.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn straightforward_positive_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(&data, |x| x * 2);
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn straightforward_negative_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(&data, |x| x * 2);
        assert_ne!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn straightforward_panic_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(&data, |x| {
            if x == 3 {
                panic!("Panic! Value 3!");
            }
            x * 2
        });
        assert_ne!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn straightforward_type_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(&data, f64::from);
        assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn straightforward_huge_test() {
        let limit = THRESHOLD * 100_000;
        let mut data = Vec::with_capacity(limit);
        let mut exp_result = Vec::with_capacity(limit);
        let mut rng = rand::thread_rng();

        for _ in 0..limit {
            let val = rng.gen::<i32>();
            data.push(val);
            exp_result.push(val / 2);
        }

        use std::time::Instant;
        let now = Instant::now();
        let results = straightforward_parallel(&data, |x| x / 2);
        let elapsed = now.elapsed();
        println!("Straightforward elapsed: {:.2?}", elapsed);

        assert_eq!(results, exp_result);
    }

    #[test]
    fn flexible_positive_test() {
        let data = vec![1, 2, 3, 4, 5];
        let exp_result = vec![2, 4, 6, 8, 10];
        let mut results = flexible_parallel(&data, |x| x * 2);
        results.sort();
        assert_eq!(results, exp_result);
    }

    #[test]
    fn flexible_negative_test() {
        let data = vec![1, 2, 3, 4, 5];
        let exp_result = vec![1, 2, 3, 4, 5];
        let mut results = flexible_parallel(&data, |x| x * 2);
        results.sort();
        assert_ne!(results, exp_result);
    }

    #[test]
    fn flexible_type_test() {
        let data = vec![1, 2, 3, 4, 5];
        let exp_result = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut results = flexible_parallel(&data, f64::from);
        results.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(results, exp_result);
    }

    #[test]
    fn flexible_huge_test() {
        let limit = THRESHOLD * 100_000;
        let mut data = Vec::with_capacity(limit);
        let mut exp_result = Vec::with_capacity(limit);
        let mut rng = rand::thread_rng();

        for _ in 0..limit {
            let val = rng.gen::<i32>();
            data.push(val);
            exp_result.push(val / 2);
        }

        use std::time::Instant;
        let now = Instant::now();
        let mut results = flexible_parallel(&data, |x| x / 2);
        let elapsed = now.elapsed();
        println!("Straightforward elapsed: {:.2?}", elapsed);

        results.sort();
        exp_result.sort();

        assert_eq!(results, exp_result);
    }
}
