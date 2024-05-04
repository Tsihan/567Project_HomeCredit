## Environment Settings
To run these notebook codes, you'd better use conda to manage a virtual environment. The recommended version of Python should be `3.11.X`. We assume you will use Anaconda, then the command is `conda create -n HomeCredit python=3.11 -y` and  `conda activate HomeCredit`(If you use some IDE like Pychram you can do it in a graphic way). 

And also remember to install the required libraries in the requirements.txt file. use this command: `pip install -r requirements.txt`. Some of the libraries are unnecessary, but for simplicity we don't filter them out.

## Run the code
To run the notebook, you can simply run them either by VS Code or Pycharm. If you want to run the code regarding to the Neuron Network model, you need to be careful of your OS. To comment and uncomment some code snippets to use `GPU` or `MPS` (Metal Performance Shaders). Or just use the CPU for a simple test.

## How to get the data
The data is available here: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data. If you don't want to register, we also provide them here: `https://drive.google.com/file/d/17u2HrtrU8T3aG50jjeKn5xqRwa-6B24w/view?usp=drive_link`. We use CSV files to do the competition, the complete size is 3GB or so when compressed, and 22GB or so when uncompressed. Take a look at the files' positions in the code, and make sure to put them in the correct place(so the test folder, the train folder, and these notebooks should be in parallel positions). The `feature_definitions.csv` and `sample_submission.csv` won't be used in the model, just to understand the feature defined in the CSV files and the submission form.
