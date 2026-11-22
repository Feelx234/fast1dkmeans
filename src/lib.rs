use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use oned_core::kmeans;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn sorted_slice<'a>(x: &'a PyReadonlyArray1<'_, f64>, k: usize) -> PyResult<&'a [f64]> {
    let x = x.as_slice().map_err(|_| PyValueError::new_err("x must be contiguous"))?;
    if k == 0 || k > x.len() {
        return Err(PyValueError::new_err(format!("k={k} must be in 1..={}", x.len())));
    }
    Ok(x)
}

/// Optimal k-means of a **sorted** 1d float array via binary search over lambda.
#[pyfunction]
#[pyo3(signature = (x, k, method=1, max_iter=200, epsilon=1e-10))]
fn binary_search<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    k: usize,
    method: u8,
    max_iter: usize,
    epsilon: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let x = sorted_slice(&x, k)?;
    if method > 1 {
        return Err(PyValueError::new_err("method must be 0 or 1"));
    }
    let out = py.detach(|| kmeans::binary_search(x, k, max_iter, epsilon, method == 1));
    Ok(out.into_pyarray(py))
}

/// Wu's matrix searching DP, O(kn) space.
#[pyfunction]
fn dynamic_programming_kn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let x = sorted_slice(&x, k)?;
    Ok(py.detach(|| kmeans::cluster_xi(x, k)).into_pyarray(py))
}

/// Wu's matrix searching DP with linear space.
#[pyfunction]
fn dynamic_programming_space<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let x = sorted_slice(&x, k)?;
    let out = py
        .detach(|| kmeans::cluster_xi_space(x, k))
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(out.into_pyarray(py))
}

/// Implicit solution of the lambda-regularized problem (testing hook).
#[pyfunction]
fn regularized<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    lambda: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let x = sorted_slice(&x, 1)?;
    let out = py.detach(|| {
        let seg = kmeans::KmeansSegment::new(x);
        kmeans::regularized(&seg, x.len(), lambda)
    });
    Ok(out.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_programming_kn, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_programming_space, m)?)?;
    m.add_function(wrap_pyfunction!(regularized, m)?)?;
    Ok(())
}
