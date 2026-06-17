# !pip install onnxruntime

import numpy as np
import onnxruntime as rt
import os

def main():
    # 1. Define filenames and expected configurations
    model_path = "speech_model.onnx" # Changed to use the newly converted model
    target_files = {
        "Hello": "Hello_pre.npy",
        "Help_me": "Help_me_pre.npy",
        "Stop": "Stop_pre.npy"
    }

    # Check if the model file exists before booting the engine
    if not os.path.exists(model_path):
        print(f" Error: Model file '{model_path}' not found in the current directory.")
        return

    print("=== INITIALIZING ONNX INFERENCE ENGINE ===")
    # Load the optimized ONNX session, explicitly setting the CPUExecutionProvider as a fallback
    session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    inference_outputs = {}

    print("\n=== RUNNING PREPROCESSED DIRECT INFERENCE ===")
    for class_name, filename in target_files.items():
        if not os.path.exists(filename):
            print(f" Warning: Target file '{filename}' missing. Skipping...")
            continue

        # Load the 4D topographic tensor [32, 32, 3, 39]
        preprocessed_data = np.load(filename)

        # Add the batch dimension to create the expected 5D shape [1, 32, 32, 3, 39]
        processed_input = np.expand_dims(preprocessed_data, axis=0).astype(np.float32)

        # Execute the forward pass through the ONNX computation graph
        preds = session.run(None, {input_name: processed_input})[0]

        # Store raw model predictions (probabilities/logits)
        inference_outputs[class_name] = preds

        # Determine winning class index
        pred_class = np.argmax(preds, axis=1)[0]

        print(f"Sample: {class_name:<8} | Probabilities: {np.round(preds[0], 4)} | Winner Class Index: {pred_class}")

    # 2. Save results out to pre_results.npz for validation tracking
    np.savez("pre_results.npz", **inference_outputs)
    print("\n Success! Direct inference evaluations saved completely to 'pre_results.npz'")

if __name__ == "__main__":
    main()

