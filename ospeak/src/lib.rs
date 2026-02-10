use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use optispeech::model::OptiSpeech; // Gebaseerd op de interne structuur van Mush42

#[pyclass]
struct OptiSpeechWrapper {
    inner: OptiSpeech,
}

#[pymethods]
impl OptiSpeechWrapper {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        // We laden het ONNX model via de interne Rust-logica
        let model = OptiSpeech::load(model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Model load error: {}", e)))?;
        Ok(OptiSpeechWrapper { inner: model })
    }

    fn predict(&self, text: String, d_factor: f32, p_factor: f32, e_factor: f32) -> PyResult<Py<PyArray1<f32>>> {
        let wav = self.inner.synthesize(&text, d_factor, p_factor, e_factor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Synthesis error: {}", e)))?;
        
        Python::with_gil(|py| {
            Ok(wav.to_pyarray(py).to_owned())
        })
    }
}

#[pymodule]
fn liboptispeech(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptiSpeechWrapper>()?;
    Ok(())
}
