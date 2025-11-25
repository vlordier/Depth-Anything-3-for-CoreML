Depth Anything 3 → CoreML
=================================

Quick Start
-----------

1. **Install uv** (if you do not already have it): follow the instructions at <https://docs.astral.sh/uv/getting-started/installation/>.
2. **Install git-lfs** (required because the checkpoint is stored with large files). On macOS:

	```sh
	brew install git-lfs
	git lfs install
	```

3. **Download the pretrained weights** from Hugging Face, for example:

	```sh
	git clone https://huggingface.co/depth-anything/DA3-SMALL
	```

	- By default the converter expects this directory to live at `DA3-SMALL/` relative to the repo root. If you move it elsewhere or want to use a different checkpoint, update `DEFAULT_WEIGHTS_PATH` in `coreml_converter/convert2coreml.py` accordingly.
	- If you want to use a different model version (for example, da3-large), you will also need to update the parameters in `_build_head` and `_build_backbone` 
	
		**(TODO: add a dict to store parameters for all versions)**.

4. **Sync the Python environment** (installs the locked dependencies declared in `pyproject.toml`):

	```sh
	uv sync
	```

5. **Export the CoreML model** using the converter script. This runs inside the managed environment without needing to activate a virtualenv manually:

	```sh
	uv run coreml_converter/convert2coreml.py
	```

	- output `da3.mlpackage` by default.
	- Pass `--run-test` to also execute a quick inference on `assets/examples/SOH/000.jpg` prior to export (useful for verifying the PyTorch model setup):

	  ```sh
	  uv run coreml_converter/convert2coreml.py --run-test
	  ```

4. **Inspect the output**: the generated CoreML package lands at the project root unless you modify `DEFAULT_COREML_OUTPUT` inside `coreml_converter/convert2coreml.py`.
