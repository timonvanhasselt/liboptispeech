use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
// Importeer de exacte struct uit de optispeech crate
use optispeech::OptiSpeechCNXModel; 

#[pyclass]
struct OptiSpeechWrapper {
    inner: OptiSpeechCNXModel,
}

#[pymethods]
impl OptiSpeechWrapper {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        // We laden het model. 
        // None wordt meegegeven voor de config om de defaults uit het modelbestand te gebruiken.
        let model = OptiSpeechCNXModel::from_path(model_path, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Model load error: {}", e)))?;
        Ok(OptiSpeechWrapper { inner: model })
    }

    /// De predict functie die je vanuit Python aanroept.
    /// input_ids: Een lijst van lijsten met phoneme getallen (bijv. [[1, 5, 22]])
    fn predict(
        &self, 
        input_ids: Vec<Vec<i64>>, 
        d_factor: f64, 
        p_factor: f64, 
        e_factor: f64
    ) -> PyResult<Py<PyArray1<f32>>> {
        // 1. Converteer de Python input naar de juiste slices voor de engine
        let id_slices: Vec<&[i64]> = input_ids.iter().map(|v| v.as_slice()).collect();
        
        // 2. Bereid de tensors voor (padding en lengtes)
        let (inputs, lengths) = self.inner.prepare_input(&id_slices)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Input prep error: {}", e)))?;

        // 3. Voer de eigenlijke AI-synthese uit
        let output = self.inner.synthesise(
            &inputs, 
            &lengths, 
            Some(d_factor), 
            Some(p_factor), 
            Some(e_factor)
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Synthesis error: {}", e)))?;
        
        // 4. Converteer de audio-output naar een Numpy array voor Python
        if let Some(samples) = output.audio_samples.first() {
            let wav_vec = samples.as_vec().to_vec();
            Python::with_gil(|py| {
                // .to_pyarray geeft een Bound object (nieuw in PyO3 0.21+)
                let bound_array = wav_vec.to_pyarray(py);
                // .unbind() zet het om naar de Py<...> die we in de return-type beloven
                Ok(bound_array.unbind())
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Geen audio gegenereerd door het model"))
        }
    }
}

/// Definieer de Python module
#[pymodule]
fn liboptispeech(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptiSpeechWrapper>()?;
    Ok(())
}
