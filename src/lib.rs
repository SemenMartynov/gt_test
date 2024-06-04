use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
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
/// let results = straightforward_parallel(data, |x| x * 2);
/// assert_eq!(results, vec![2, 4, 6, 8, 10]);
/// ```
/// # Example
///
/// ```
/// use gt_test::flexible_parallel;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let results = flexible_parallel(data, |x| x * 2);
/// assert_eq!(results, vec![2, 4, 6, 8, 10]);
/// ```

// Adjust the threshold
const THRESHOLD: usize = 128;

pub fn straightforward_parallel<T, R>(data: Vec<T>, f: fn(t: &T) -> R) -> Vec<R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(f).collect()
    } else {
        // Split the work into threads
        let num_workers = available_parallelism().unwrap().get();
        let chunk_size = 1 + data.len() / num_workers;

        let mut handles = Vec::with_capacity(num_workers);
        let mut results = Vec::with_capacity(data.len());

        for chunk in data.chunks(chunk_size) {
            let chunk_data = chunk.to_vec();
            let handle = thread::spawn(move || {
                chunk_data.iter().map(f).collect::<Vec<_>>()
            });
            handles.push(handle);
        }

        // Collect results from threads
        for handle in handles {
            results.extend(handle.join().unwrap());
        }

        results
    }
}

pub fn parallel_split_off<T, R>(data: Vec<T>, f: fn(t: &T) -> R) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(f).collect()
    } else {
        // Split the work into threads
        let num_workers = available_parallelism().unwrap().get();
        let chunk_size = 1 + data.len() / num_workers;

        let mut handles = Vec::with_capacity(num_workers);
        let mut results = Vec::with_capacity(data.len());

        let mut data = data;
        for _ in 0..num_workers {
            // I do expect that `copy` works as `move` here
            let chunk_data = data.split_off(data.len() - data.len().min(chunk_size));
            // If there's no data left to process, we're done
            if chunk_data.is_empty() {
                break;
            }

            let handle = thread::spawn(move || {
                chunk_data.iter().map(f).collect::<Vec<_>>()
            });
            handles.push(handle);
        }

        // Collect results from threads
        for handle in handles {
            results.extend(handle.join().unwrap());
        }

        results
    }
}

pub fn straightforward_parallel_arc<T, R>(data: Vec<T>, f: fn(t: &T) -> R) -> Vec<R>
where
    T: Send + Sync + 'static,
    R: Send + Clone + Default + Debug + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(f).collect()
    } else {
        let input = Arc::new(data);
        let output = Arc::new(Mutex::new(vec![R::default(); input.len()]));

        // Split the work into threads
        let num_workers = available_parallelism().unwrap().get();
        let chunk_size = 1 + input.len() / num_workers;

        let mut handles = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let input = Arc::clone(&input);
            let output = Arc::clone(&output);
            let handle = thread::spawn(move || {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, input.len());

                for j in start..end {
                    let result = f(&input[j]);
                    output.lock().unwrap()[j] = result;
                }
            });
            handles.push(handle);
        }

        // Collect results from threads
        for handle in handles {
            handle.join().unwrap();
        }

        Arc::try_unwrap(output).unwrap().into_inner().unwrap()
    }
}

pub fn straightforward_parallel_arc_nomutex<T, R>(data: Vec<T>, f: fn(t: &T) -> R) -> Vec<R>
where
    T: Send + Sync + 'static,
    R: Send + Clone + Default + Debug + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(f).collect()
    } else {
        let input = Arc::new(data);

        // Split the work into threads
        let num_workers = available_parallelism().unwrap().get();
        let chunk_size = 1 + input.len() / num_workers;

        let mut handles = Vec::with_capacity(num_workers);
        let mut results = Vec::with_capacity(input.len());

        for i in 0..num_workers {
            let input = Arc::clone(&input);
            let handle = thread::spawn(move || {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, input.len());

                let chunk = &input[start..end];
                chunk.iter().map(f).collect::<Vec<_>>()

                //let chunk_data = chunk.to_vec();
                //chunk_data.into_iter().map(f).collect::<Vec<_>>()
            });
            handles.push(handle);
        }

        // Collect results from threads
        for handle in handles {
            results.extend(handle.join().unwrap());
        }

        results
    }
}

pub fn prelude<T, R>(data: Vec<T>, f: fn(t: &T) -> R) -> Vec<R>
where
    T: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(f).collect()
    } else {
        data.par_iter().map(f).collect()
    }
}

pub fn flexible_parallel<T, R>(data: Vec<T>, f: fn(t: &T) -> R) -> Vec<R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    if data.len() == 0 {
        // Nothing to process
        vec![]
    } else if data.len() < THRESHOLD {
        // Process sequentially if below the threshold
        data.iter().map(f).collect()
    } else {
        let num_workers = available_parallelism().unwrap().get();
        let pool = ThreadPool::new(num_workers);

        let (tx, rx) = channel();
        for item in data.iter() {
            let tx = tx.clone();
            let val = item.clone();
            pool.execute(move || {
                let result = f(&val);
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
    use std::time::Instant;
    use super::*;

    #[test]
    fn straightforward_positive_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(data, |x| x * 2);
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn straightforward_negative_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(data, |x| x * 2);
        assert_ne!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn straightforward_panic_test() {
        let data = vec![1, 2, 3, 4, 5];
        let results = straightforward_parallel(data, |x| {
            if *x == 3 {
                panic!("Panic! Value 3!");
            }
            x * 2
        });
        assert_ne!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn straightforward_type_test() {
        let data = vec![1, 2, 3, 4, 5];
        let function = |x: &i32| *x as f64;
        let results = straightforward_parallel(data, function);
        assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn straightforward_huge_test() {
        let limit = 1_000;
        let inner_limit = 1_000_000;
        let mut data = Vec::with_capacity(limit);
        let mut exp_result = Vec::with_capacity(limit);
        let mut rng = rand::thread_rng();

        fn divide_vec_by_two(vec: &Vec<i32>) -> Vec<i32> {
            vec.iter().map(|&element| element / 2).collect()
        }
        let f: fn(&Vec<i32>) -> Vec<i32> = divide_vec_by_two;

        for i in 0..limit {
            let mut inner_data = Vec::with_capacity(inner_limit);
            let mut inner_exp_result = Vec::with_capacity(inner_limit);
            for _ in 0..inner_limit {
                let val = rng.gen::<i32>();
                inner_data.push(val);
                inner_exp_result.push(val / 2);
            }
            data.push(inner_data);
            exp_result.push(inner_exp_result);

            if i % 1000 == 0 {
                println!("Generating dataset...");
            }
        }
        println!("The dataset is generated!");

        let data_clone = data.clone();
        let now = Instant::now();
        let results = straightforward_parallel(data_clone, f);
        let elapsed = now.elapsed();
        println!("Straightforward elapsed: {:.2?}", elapsed);

        let data_clone = data.clone();
        let now = Instant::now();
        let results_arc = straightforward_parallel_arc(data_clone, f);
        let elapsed = now.elapsed();
        println!("Straightforward arc elapsed: {:.2?}", elapsed);

        let data_clone = data.clone();
        let now = Instant::now();
        let results_arc_nomutex = straightforward_parallel_arc_nomutex(data_clone, f);
        let elapsed = now.elapsed();
        println!("Straightforward arc nomutex elapsed: {:.2?}", elapsed);

        let data_clone = data.clone();
        let now = Instant::now();
        let results_prelude = prelude(data_clone, f);
        let elapsed = now.elapsed();
        println!("Prelude elapsed: {:.2?}", elapsed);

        assert_eq!(results, exp_result);
        assert_eq!(results_arc, exp_result);
        assert_eq!(results_arc_nomutex, exp_result);
        assert_eq!(results_prelude, exp_result);
    }

    #[test]
    fn flexible_positive_test() {
        let data = vec![1, 2, 3, 4, 5];
        let exp_result = vec![2, 4, 6, 8, 10];
        let mut results = flexible_parallel(data, |x| x * 2);
        results.sort();
        assert_eq!(results, exp_result);
    }

    #[test]
    fn flexible_negative_test() {
        let data = vec![1, 2, 3, 4, 5];
        let exp_result = vec![1, 2, 3, 4, 5];
        let mut results = flexible_parallel(data, |x| x * 2);
        results.sort();
        assert_ne!(results, exp_result);
    }

    #[test]
    fn flexible_type_test() {
        let data = vec![1, 2, 3, 4, 5];
        let exp_result = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let function = |x: &i32| *x as f64;
        let mut results = flexible_parallel(data, function);
        results.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(results, exp_result);
    }

    #[test]
    fn flexible_huge_test() {
        let limit = 1_000;
        let inner_limit = 1_000_000;
        let mut data = Vec::with_capacity(limit);
        let mut exp_result = Vec::with_capacity(limit);
        let mut rng = rand::thread_rng();

        fn divide_vec_by_two(vec: &Vec<i32>) -> Vec<i32> {
            vec.iter().map(|&element| element / 2).collect()
        }
        let f: fn(&Vec<i32>) -> Vec<i32> = divide_vec_by_two;

        for i in 0..limit {
            let mut inner_data = Vec::with_capacity(inner_limit);
            let mut inner_exp_result = Vec::with_capacity(inner_limit);
            for _ in 0..inner_limit {
                let val = rng.gen::<i32>();
                inner_data.push(val);
                inner_exp_result.push(val / 2);
            }
            data.push(inner_data);
            exp_result.push(inner_exp_result);

            if i % 1000 == 0 {
                println!("Generating dataset...");
            }
        }
        println!("The dataset is generated!");

        let data_clone = data.clone();
        let now = Instant::now();
        let mut results = flexible_parallel(data_clone, f);
        let elapsed = now.elapsed();
        println!("Flexible elapsed: {:.2?}", elapsed);

        let data_clone = data.clone();
        let now = Instant::now();
        let mut results_split_off = parallel_split_off(data_clone, f);
        let elapsed = now.elapsed();
        println!("Flexible split off elapsed: {:.2?}", elapsed);

        results.sort();
        exp_result.sort();
        results_split_off.sort();

        assert_eq!(results, exp_result);
        assert_eq!(results_split_off, exp_result);
    }
}
