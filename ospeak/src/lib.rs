use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use optispeech::OptiSpeechCNXModel; 

// De 'unsendable' flag lost de Send/Sync errors op.
// Dit is veilig omdat NVDA's synthesizer meestal op één thread blijft.
#[pyclass(unsendable)]
struct OptiSpeechWrapper {
    inner: OptiSpeechCNXModel,
}

#[pymethods]
impl OptiSpeechWrapper {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        // Laden van het model op de CPU
        let model = OptiSpeechCNXModel::from_path(model_path, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Model load error: {}", e)))?;
        Ok(OptiSpeechWrapper { inner: model })
    }

    fn predict(
        &self, 
        input_ids: Vec<Vec<i64>>, 
        d_factor: f64, 
        p_factor: f64, 
        e_factor: f64
    ) -> PyResult<Py<PyArray1<f32>>> {
        let id_slices: Vec<&[i64]> = input_ids.iter().map(|v| v.as_slice()).collect();
        
        let (inputs, lengths) = self.inner.prepare_input(&id_slices)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Input prep error: {}", e)))?;

        let output = self.inner.synthesise(
            &inputs, 
            &lengths, 
            Some(d_factor), 
            Some(p_factor), 
            Some(e_factor)
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Synthesis error: {}", e)))?;
        
        if let Some(samples) = output.audio_samples.first() {
            let wav_vec = samples.as_vec().to_vec();
            Python::with_gil(|py| {
                let bound_array = wav_vec.to_pyarray(py);
                Ok(bound_array.unbind())
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Geen audio gegenereerd"))
        }
    }
}

#[pymodule]
fn liboptispeech(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptiSpeechWrapper>()?;
    Ok(())
}
