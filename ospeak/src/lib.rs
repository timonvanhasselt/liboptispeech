use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
// We importeren de specifieke namen uit jouw lib.rs
use optispeech::OptiSpeechCNXModel; 

#[pyclass]
struct OptiSpeechWrapper {
    inner: OptiSpeechCNXModel,
}

#[pymethods]
impl OptiSpeechWrapper {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        // Gebruik de 'from_path' functie uit jouw lib.rs
        // We geven 'None' mee voor de config om de defaults te gebruiken
        let model = OptiSpeechCNXModel::from_path(model_path, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Model load error: {}", e)))?;
        Ok(OptiSpeechWrapper { inner: model })
    }

    fn predict(&self, input_ids: Vec<Vec<i64>>, d_factor: f64, p_factor: f64, e_factor: f64) -> PyResult<Py<PyArray1<f32>>> {
        // 1. Bereid de input voor (pad_sequences logica)
        let slice_ids: Vec<&[i64]> = input_ids.iter().map(|v| v.as_slice()).collect();
        let (inputs, lengths) = self.inner.prepare_input(&slice_ids)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Input prep error: {}", e)))?;

        // 2. Synthese (synthesise functie uit jouw lib.rs)
        let output = self.inner.synthesise(&inputs, &lengths, Some(d_factor), Some(p_factor), Some(e_factor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Synthesis error: {}", e)))?;
        
        // 3. Audio ophalen en converteren
        // We pakken de eerste audio sample uit de batch
        if let Some(samples) = output.audio_samples.first() {
            let wav_vec = samples.as_vec().to_vec();
            Python::with_gil(|py| {
                Ok(wav_vec.to_pyarray(py).to_owned())
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No audio generated"))
        }
    }
}

#[pymodule]
fn liboptispeech(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptiSpeechWrapper>()?;
    Ok(())
}
